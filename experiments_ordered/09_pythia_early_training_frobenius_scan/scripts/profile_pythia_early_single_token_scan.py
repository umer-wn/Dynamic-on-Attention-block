#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_utils import load_model_and_tokenizer
from src.single_token_dynamics import (
    SingleTokenOperator,
    exact_target_jacobian,
    fixed_projection_vectors,
    jacobian_summary,
    run_trajectory,
)


MODEL = "EleutherAI/pythia-70m"
CACHE = "/home/luohaoming/model_feature_cache/hf_cache"


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_memory() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def reset_peak() -> None:
    if torch.cuda.is_available():
        sync()
        torch.cuda.reset_peak_memory_stats()


def atomic_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def stage(payload: dict, output: Path, name: str, function):
    reset_peak()
    sync()
    started = time.perf_counter()
    value = function()
    sync()
    elapsed = time.perf_counter() - started
    payload["stages"][name] = {"seconds": elapsed, **gpu_memory()}
    atomic_dump(output, payload)
    print(json.dumps({"stage": name, **payload["stages"][name]}), flush=True)
    return value


def fixed_texts() -> tuple[list[str], str]:
    dataset = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test", cache_dir=CACHE,
        download_mode="reuse_dataset_if_exists",
    )
    texts = []
    for item in dataset:
        text = str(item.get("text", "")).strip()
        if text:
            texts.append(text)
        if len(texts) == 128:
            break
    digest = hashlib.sha256("\n\0\n".join(texts).encode()).hexdigest()
    return texts, digest


def sequential_loss(model, tokenizer, texts: list[str], device: torch.device) -> list[dict]:
    rows = []
    with torch.no_grad():
        for sample_index, text in enumerate(texts):
            encoded = tokenizer(text, truncation=True, max_length=64, padding="max_length", return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            labels = input_ids.clone(); labels[mask == 0] = -100
            count = int(mask[:, 1:].sum().item())
            if count:
                mean = float(model(input_ids=input_ids, attention_mask=mask, labels=labels, use_cache=False).loss.float().cpu())
                nll = mean * count
            else:
                mean, nll = None, 0.0
            rows.append({"sample_index": sample_index, "count": count, "nll": nll, "mean": mean})
    return rows


def batched_loss(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 16) -> list[dict]:
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size], truncation=True, max_length=64,
                padding="max_length", return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits.float()
            labels = input_ids[:, 1:]
            valid = mask[:, 1:].bool()
            token_nll = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]), labels.contiguous().view(-1), reduction="none"
            ).view(labels.shape)
            for local in range(len(labels)):
                count = int(valid[local].sum().item())
                nll = float(token_nll[local][valid[local]].sum().cpu()) if count else 0.0
                rows.append({"sample_index": start + local, "count": count, "nll": nll,
                             "mean": nll / count if count else None})
    return rows


def aggregate(rows: list[dict]) -> float:
    return sum(row["nll"] for row in rows) / sum(row["count"] for row in rows)


def direct_trajectory(operator: SingleTokenOperator, initial: torch.Tensor, steps: int, projections) -> tuple[list[torch.Tensor], list[dict]]:
    state = initial.detach().float()
    states = [state.detach().clone()]
    rows = []
    with torch.no_grad():
        for index in range(steps):
            state = operator.full_step(state).float()
            rows.append({
                "step": index + 1,
                **{f"projection_{k}": float(torch.dot(state, vector).cpu()) for k, vector in enumerate(projections)},
            })
            states.append(state.detach().clone())
    return states, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="step1000")
    parser.add_argument("--token-id", type=int, default=35408)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--jacobian-chunks", type=int, nargs="+", default=[16, 128])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default=(
        "/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/"
        "profiling/step1000_token35408.json"
    ))
    args = parser.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    output = Path(args.output)
    payload = {
        "status": "running",
        "revision": args.revision,
        "token_id": args.token_id,
        "steps": args.steps,
        "jacobian_chunks": args.jacobian_chunks,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "offline": args.offline,
        "stages": {},
    }
    atomic_dump(output, payload)

    model, tokenizer, device = stage(
        payload, output, "model_and_tokenizer_load",
        lambda: load_model_and_tokenizer(MODEL, args.revision, MODEL, "float32", "cuda", CACHE, "eager", args.offline),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts, dataset_hash = stage(payload, output, "dataset_load_first_128", fixed_texts)
    payload["dataset_text_sha256"] = dataset_hash

    def warmup():
        encoded = tokenizer(texts[0], truncation=True, max_length=64, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            model(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device), use_cache=False)
    stage(payload, output, "one_forward_warmup", warmup)

    sequential = stage(payload, output, "loss_sequential_128", lambda: sequential_loss(model, tokenizer, texts, device))
    batched = stage(payload, output, "loss_batched16_128", lambda: batched_loss(model, tokenizer, texts, device, 16))
    sequential_value, batched_value = aggregate(sequential), aggregate(batched)
    per_sample_max_abs = max(abs(a["mean"] - b["mean"]) for a, b in zip(sequential, batched) if a["mean"] is not None)
    payload["loss_parity"] = {
        "sequential": sequential_value,
        "batched16": batched_value,
        "aggregate_abs_difference": abs(sequential_value - batched_value),
        "per_sample_mean_max_abs_difference": per_sample_max_abs,
        "predicted_tokens_equal": [r["count"] for r in sequential] == [r["count"] for r in batched],
    }
    atomic_dump(output, payload)

    with torch.no_grad():
        initial = model.get_input_embeddings()(torch.tensor([args.token_id], device=device)).float()[0]
    operator = SingleTokenOperator(model, "isolated_token")
    projections = fixed_projection_vectors(initial, 2, 1234)
    double = stage(
        payload, output, "trajectory_current_with_nearby_768",
        lambda: run_trajectory(operator, initial, args.steps, 1e-3, 1234, projections),
    )
    direct_states, direct_rows = stage(
        payload, output, "trajectory_projection_only_768",
        lambda: direct_trajectory(operator, initial, args.steps, projections),
    )
    endpoint_difference = float((double.states[-1] - direct_states[-1]).abs().max().cpu())
    payload["trajectory_parity"] = {
        "endpoint_max_abs_difference": endpoint_difference,
        "current_state_count": len(double.states),
        "direct_state_count": len(direct_states),
        "direct_row_count": len(direct_rows),
    }
    atomic_dump(output, payload)

    matrices = {}
    for chunk in args.jacobian_chunks:
        matrix = stage(
            payload, output, f"exact_jacobian_chunk{chunk}",
            lambda chunk=chunk: exact_target_jacobian(operator, direct_states[767], chunk_size=chunk),
        )
        matrices[chunk] = matrix.cpu()
        payload.setdefault("jacobians", {})[str(chunk)] = jacobian_summary(matrix)
        atomic_dump(output, payload)
    reference_chunk = args.jacobian_chunks[0]
    payload["jacobian_parity"] = {}
    for chunk in args.jacobian_chunks[1:]:
        difference = (matrices[reference_chunk] - matrices[chunk]).abs()
        payload["jacobian_parity"][f"{reference_chunk}_vs_{chunk}"] = {
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "allclose_rtol1e-5_atol1e-6": bool(torch.allclose(matrices[reference_chunk], matrices[chunk], rtol=1e-5, atol=1e-6)),
        }
    payload["status"] = "complete"
    atomic_dump(output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
