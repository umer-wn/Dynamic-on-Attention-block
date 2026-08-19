#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_io import atomic_json, atomic_jsonl, read_jsonl, sha256_file

from src.model_utils import load_model_and_tokenizer
from src.single_token_dynamics import (
    SingleTokenOperator,
    exact_target_jacobian,
    fixed_projection_vectors,
)


MODEL_NAME = "EleutherAI/pythia-70m"
DEFAULT_CACHE = "/home/luohaoming/model_feature_cache/hf_cache"
DEFAULT_ROOT = "/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan"
DEFAULT_TOKENS = (
    "/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/"
    "frequency_audit/selected_tokens.jsonl"
)
DEFAULT_TEST_MANIFEST = (
    "/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/"
    "manifests/wikitext_test_first128.jsonl"
)


def load_fixed_test_texts(manifest_path: Path, count: int = 128) -> tuple[list[str], str]:
    rows = read_jsonl(manifest_path)
    if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("test manifest sample_index must be contiguous from zero")
    texts = [str(row["text"]) for row in rows]
    if len(texts) != int(count):
        raise RuntimeError(f"expected {count} manifest texts, got {len(texts)}")
    digest = hashlib.sha256("\n\0\n".join(texts).encode("utf-8")).hexdigest()
    return texts, digest


def compute_loss_rows(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    sequence_length: int,
    batch_size: int,
) -> list[dict]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[dict] = []
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_start in range(0, len(texts), int(batch_size)):
            batch_texts = texts[batch_start: batch_start + int(batch_size)]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=int(sequence_length),
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = input_ids[:, 1:].contiguous()
            valid = attention_mask[:, 1:].bool()
            token_nll = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                reduction="none",
            ).view(shifted_labels.shape)
            for local_index in range(len(batch_texts)):
                sample_index = batch_start + local_index
                predicted_tokens = int(valid[local_index].sum().item())
                nll_sum = float(token_nll[local_index][valid[local_index]].sum().cpu()) if predicted_tokens else 0.0
                mean_loss = nll_sum / predicted_tokens if predicted_tokens else None
                rows.append(
                    {
                        "sample_index": sample_index,
                        "nll_sum": nll_sum,
                        "predicted_token_count": predicted_tokens,
                        "mean_loss": mean_loss,
                    }
                )
                total_nll += nll_sum
                total_tokens += predicted_tokens
    loss = total_nll / max(total_tokens, 1)
    rows.append(
        {
            "sample_index": "aggregate",
            "nll_sum": total_nll,
            "predicted_token_count": total_tokens,
            "mean_loss": loss,
            "token_weighted_loss": loss,
            "token_weighted_perplexity": math.exp(min(loss, 20.0)),
        }
    )
    return rows


def initial_projection_row(state: torch.Tensor, projections: list[torch.Tensor]) -> dict:
    row = {
        "dynamics_step": 0,
        "state_norm": float(state.norm().cpu()),
        "step_delta": None,
        "relative_step_delta": None,
    }
    for index, vector in enumerate(projections):
        row[f"projection_{index}"] = float(torch.dot(state.float(), vector).cpu())
    return row


def run_projection_trajectory(
    operator: SingleTokenOperator,
    initial_state: torch.Tensor,
    steps: int,
    projections: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[dict]]:
    state = initial_state.detach().float()
    states = [state.detach().clone()]
    rows = [initial_projection_row(state, projections)]
    with torch.no_grad():
        for step in range(int(steps)):
            previous = state
            state = operator.full_step(state).float()
            delta = float((state - previous).norm().cpu())
            norm = float(state.norm().cpu())
            row = {
                "dynamics_step": step + 1,
                "state_norm": norm,
                "step_delta": delta,
                "relative_step_delta": delta / max(norm, 1e-12),
            }
            for index, vector in enumerate(projections):
                row[f"projection_{index}"] = float(torch.dot(state, vector).cpu())
            rows.append(row)
            states.append(state.detach().clone())
    return states, rows


def frobenius_only_summary(jacobian: torch.Tensor) -> dict:
    hidden = int(jacobian.shape[0])
    return {
        "shape": list(jacobian.shape),
        "normalized_frobenius": float(jacobian.norm().cpu()) / math.sqrt(hidden),
        "frobenius_norm": float(jacobian.norm().cpu()),
        "checksum_sha256": hashlib.sha256(jacobian.detach().cpu().numpy().tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sampling-stage", choices=["coarse", "adaptive", "sentinel"], required=True)
    parser.add_argument("--mode", choices=["full", "loss-only"], default="full")
    parser.add_argument("--output-root", default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--frequency-manifest", default=DEFAULT_TOKENS)
    parser.add_argument("--test-manifest", default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--test-samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--loss-batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--jacobian-step", type=int, default=767)
    parser.add_argument("--projection-seed", type=int, default=1234)
    parser.add_argument("--projection-count", type=int, default=2)
    parser.add_argument("--jacobian-chunk-size", type=int, default=128)
    parser.add_argument("--token-limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    checkpoint_dir = root / "raw" / args.revision
    complete_path = checkpoint_dir / "run_complete.json"
    if complete_path.exists() and not args.force:
        completed = json.loads(complete_path.read_text(encoding="utf-8"))
        if completed.get("mode") == "full" or completed.get("mode") == args.mode:
            print(json.dumps({"revision": args.revision, "status": "already_complete", "mode": completed.get("mode")}))
            return

    started = time.time()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (root / "jacobians" / args.revision).mkdir(parents=True, exist_ok=True)
    token_path = Path(args.frequency_manifest)
    token_rows = read_jsonl(token_path)
    if args.token_limit is not None:
        token_rows = token_rows[: args.token_limit]
    if args.mode == "full" and len(token_rows) != (args.token_limit or 16):
        raise RuntimeError(f"expected {(args.token_limit or 16)} tokens, got {len(token_rows)}")

    os.environ.setdefault("HF_HOME", args.cache_dir)
    model_load_started = time.perf_counter()
    model, tokenizer, device = load_model_and_tokenizer(
        MODEL_NAME,
        args.revision,
        MODEL_NAME,
        "float32",
        "cuda",
        cache_dir=args.cache_dir,
        attn_implementation="eager",
        local_files_only=not args.allow_network,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    test_manifest_path = Path(args.test_manifest)
    texts, dataset_hash = load_fixed_test_texts(test_manifest_path, args.test_samples)
    loss_started = time.perf_counter()
    loss_rows = compute_loss_rows(model, tokenizer, texts, device, args.sequence_length, args.loss_batch_size)
    loss_seconds = time.perf_counter() - loss_started
    common = {
        "model": MODEL_NAME,
        "checkpoint": args.revision,
        "training_step": int(args.revision.removeprefix("step")),
        "sampling_stage": args.sampling_stage,
        "test_sample_count": args.test_samples,
        "sequence_length": args.sequence_length,
        "loss_batch_size": args.loss_batch_size,
        "dataset": "wikitext/wikitext-2-raw-v1:test:first-128-nonempty",
        "test_manifest": str(test_manifest_path),
        "test_manifest_sha256": sha256_file(test_manifest_path),
        "dataset_text_sha256": dataset_hash,
    }
    atomic_jsonl(checkpoint_dir / "loss.jsonl", [{**common, **row} for row in loss_rows])

    trajectory_rows: list[dict] = []
    jacobian_rows: list[dict] = []
    trajectory_seconds = 0.0
    jacobian_seconds = 0.0
    if args.mode == "full":
        embedding = model.get_input_embeddings()
        for token_index, token_row in enumerate(token_rows):
            token_id = int(token_row["token_id"])
            with torch.no_grad():
                initial = embedding(torch.tensor([token_id], device=device)).float()[0]
            operator = SingleTokenOperator(model, "isolated_token")
            projections = fixed_projection_vectors(initial, args.projection_count, args.projection_seed)
            trajectory_started = time.perf_counter()
            states, token_trajectory_rows = run_projection_trajectory(operator, initial, args.steps, projections)
            trajectory_seconds += time.perf_counter() - trajectory_started
            base = {
                "checkpoint": args.revision,
                "training_step": common["training_step"],
                "sampling_stage": args.sampling_stage,
                "token_index": token_index,
                "token_id": token_id,
                "decoded": token_row["decoded"],
                "frequency_bin": int(token_row["frequency_bin"]),
                "frequency_count": int(token_row["count"]),
                "operator": "isolated_token_final_hidden_feedback",
                "hidden_size": int(initial.numel()),
                "projection_seed": args.projection_seed,
            }
            trajectory_rows.extend({**base, **row} for row in token_trajectory_rows)

            if not 0 <= args.jacobian_step < len(states):
                raise RuntimeError(f"jacobian step {args.jacobian_step} is outside saved states")
            jacobian_started = time.perf_counter()
            jacobian = exact_target_jacobian(
                operator,
                states[args.jacobian_step],
                chunk_size=args.jacobian_chunk_size,
            )
            jacobian_seconds += time.perf_counter() - jacobian_started
            if tuple(jacobian.shape) != (512, 512):
                raise RuntimeError(f"expected [512,512], got {tuple(jacobian.shape)}")
            matrix_path = root / "jacobians" / args.revision / f"token{token_id}__t{args.jacobian_step}.pt"
            torch.save(jacobian.cpu(), matrix_path)
            jacobian_rows.append(
                {
                    **base,
                    "trajectory_step": args.jacobian_step,
                    "matrix_path": str(matrix_path),
                    **frobenius_only_summary(jacobian),
                }
            )
            print(
                json.dumps(
                    {
                        "checkpoint": args.revision,
                        "token": token_id,
                        "rho": jacobian_rows[-1]["normalized_frobenius"],
                    }
                ),
                flush=True,
            )
        atomic_jsonl(checkpoint_dir / "trajectories.jsonl", trajectory_rows)
        atomic_jsonl(checkpoint_dir / "jacobians.jsonl", jacobian_rows)

    aggregate = loss_rows[-1]
    manifest = {
        **common,
        "mode": args.mode,
        "token_manifest": str(token_path),
        "token_manifest_sha256": sha256_file(token_path),
        "token_ids": [int(row["token_id"]) for row in token_rows] if args.mode == "full" else [],
        "dynamics_steps": args.steps if args.mode == "full" else 0,
        "jacobian_step": args.jacobian_step if args.mode == "full" else None,
        "jacobian_chunk_size": args.jacobian_chunk_size if args.mode == "full" else None,
        "jacobian_definition": "d final_hidden[0,0,:] / d inputs_embeds[0,0,:]",
        "normalized_frobenius_definition": "||J||_F/sqrt(512)",
        "projection_seed": args.projection_seed,
        "projection_count": args.projection_count,
        "dtype": "float32",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "token_weighted_loss": aggregate["token_weighted_loss"],
        "token_weighted_perplexity": aggregate["token_weighted_perplexity"],
        "predicted_token_count": aggregate["predicted_token_count"],
        "elapsed_seconds": time.time() - started,
        "model_load_seconds": model_load_seconds,
        "loss_seconds": loss_seconds,
        "trajectory_seconds": trajectory_seconds,
        "jacobian_seconds": jacobian_seconds,
        "local_files_only": not args.allow_network,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
    }
    atomic_json(checkpoint_dir / "manifest.json", manifest)
    atomic_json(complete_path, {"status": "complete", **manifest})
    print(json.dumps({"status": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
