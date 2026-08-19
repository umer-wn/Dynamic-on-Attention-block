#!/usr/bin/env python3
"""Compute low-dimensional projected Jacobian Frobenius norms for Exp. 20 tokens.

For the orthonormal projection rows B_d in R^{d x 512}, the reduced Jacobian is

    J_d = B_d J(x_1024) B_d^T

and the reported metric is ||J_d||_F / sqrt(d), directly analogous to the
full-dimensional normalized Frobenius metric ||J||_F / sqrt(512).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_EXP20 = Path("/data1/luohaoming/model_feature/experiments_ordered/20_100token_endpoint_jacobian")
SOURCE_EXPERIMENTS = SOURCE_EXP20.parent
EXP20_SCRIPT = SOURCE_EXP20 / "scripts/run_100token_endpoint_jacobian.py"
SPEC = importlib.util.spec_from_file_location("experiment20_base", EXP20_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {EXP20_SCRIPT}")
exp20 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp20)

PROJECTION = SOURCE_EXPERIMENTS / "16_frequency_stratified_window_jacobian/processed/projection_basis.pt"
PARTS = ROOT / "processed/projected_jacobian_parts"
SUMMARY = ROOT / "processed/projected_jacobian_checkpoint_summary.csv"
COMBINED = ROOT / "processed/endpoint100_with_projected_jacobian.csv"
DENSE_ENDPOINT = SOURCE_EXPERIMENTS / "25_dense_checkpoint_suite/processed/endpoint100_summary.csv"
BASE_ENDPOINT = SOURCE_EXP20 / "processed/checkpoint_metric_summary.csv"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_list() -> list[str]:
    early = [r["checkpoint"] for r in read_csv(BASE_ENDPOINT) if int(r["training_step"]) > 0]
    dense = [r["checkpoint"] for r in read_csv(DENSE_ENDPOINT)]
    return sorted(set(early + dense), key=lambda x: int(x.removeprefix("step")))


def projection_basis(device: torch.device) -> torch.Tensor:
    payload = torch.load(PROJECTION, map_location="cpu")
    basis = payload["basis"].float()
    if tuple(basis.shape) != (4, 512):
        raise RuntimeError(f"expected projection basis [4,512], found {tuple(basis.shape)}")
    gram = basis @ basis.T
    if not torch.allclose(gram, torch.eye(4), atol=2e-6, rtol=2e-6):
        raise RuntimeError(f"projection rows are not orthonormal: {gram}")
    return basis.to(device)


def compute_checkpoint(args: argparse.Namespace, checkpoint: str) -> None:
    part = args.parts / f"{checkpoint}.csv"
    if part.exists() and len(read_csv(part)) == args.num_tokens and not args.overwrite:
        print(json.dumps({"checkpoint": checkpoint, "status": "skip", "part": str(part)}), flush=True)
        return

    tokens = exp20.load_manifest(args.manifest, args.num_tokens)
    device = torch.device(args.device)
    basis = projection_basis(device)
    model = exp20.base.load_model(checkpoint, args.cache_dir, device)
    model.set_attn_implementation("eager")
    model.eval()

    ids = torch.tensor([row["token_id"] for row in tokens], device=device)
    state = model.get_input_embeddings()(ids).detach().float()
    attention_mask = torch.ones((len(tokens), 1), device=device, dtype=torch.long)
    position_ids = torch.zeros((1, 1), device=device, dtype=torch.long)

    def mapping(value: torch.Tensor) -> torch.Tensor:
        return model.gpt_neox(
            inputs_embeds=value.unsqueeze(1),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, -1, :].float()

    started = time.perf_counter()
    with torch.inference_mode():
        for dynamic_step in range(args.steps):
            state = mapping(state).detach()
            if (dynamic_step + 1) % 256 == 0:
                print(json.dumps({"checkpoint": checkpoint, "stage": "trajectory", "dynamic_step": dynamic_step + 1}), flush=True)

    # One batched JVP per input projection direction. Each token remains an
    # independent batch element, so reduced[token, output_axis, input_axis].
    columns = []
    for input_axis in range(4):
        tangent = basis[input_axis].expand_as(state).contiguous()
        _, jv = torch.func.jvp(mapping, (state,), (tangent,))
        columns.append(jv @ basis.T)
    reduced = torch.stack(columns, dim=2).detach().cpu().double()

    rows = []
    for index, token in enumerate(tokens):
        row = {
            "checkpoint": checkpoint,
            "training_step": int(checkpoint.removeprefix("step")),
            "dynamic_step": args.steps,
            **token,
            "hidden_dimension_N": 512,
            "projection_dimension_max": 4,
            "definition": "J_d=B_d J(x_1024) B_d^T; normalized=||J_d||_F/sqrt(d)",
        }
        for dimension in range(1, 5):
            block = reduced[index, :dimension, :dimension]
            row[f"projected_frobenius_norm_d{dimension}"] = float(torch.linalg.matrix_norm(block, ord="fro"))
            row[f"projected_normalized_frobenius_d{dimension}"] = float(
                torch.linalg.matrix_norm(block, ord="fro") / math.sqrt(dimension)
            )
        rows.append(row)
    write_csv(part, rows)
    print(json.dumps({
        "checkpoint": checkpoint,
        "status": "complete",
        "tokens": len(rows),
        "d4_mean": statistics.fmean(r["projected_normalized_frobenius_d4"] for r in rows),
        "seconds": time.perf_counter() - started,
        "part": str(part),
    }), flush=True)
    del model, state, reduced
    torch.cuda.empty_cache()


def metric_stats(values: list[float], prefix: str) -> dict:
    count = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if count > 1 else 0.0
    sem = std / math.sqrt(count)
    ordered = sorted(values)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": statistics.median(ordered),
        f"{prefix}_min": ordered[0],
        f"{prefix}_max": ordered[-1],
        f"{prefix}_sem": sem,
        f"{prefix}_ci95_low": mean - 1.96 * sem,
        f"{prefix}_ci95_high": mean + 1.96 * sem,
    }


def summarize(args: argparse.Namespace, checkpoints: list[str]) -> None:
    summary = []
    for checkpoint in checkpoints:
        rows = read_csv(args.parts / f"{checkpoint}.csv")
        if len(rows) != args.num_tokens:
            raise RuntimeError(f"{checkpoint}: expected {args.num_tokens} rows, found {len(rows)}")
        out = {
            "checkpoint": checkpoint,
            "training_step": int(checkpoint.removeprefix("step")),
            "dynamic_step": args.steps,
            "token_count": len(rows),
            "token_selection": "Experiment 20 paired WikiText-2 frequency-decile tokens",
            "token_seed": int(rows[0]["selection_seed"]),
            "projection_basis": str(PROJECTION),
            "projection_dimension_max": 4,
            "definition": "J_d=B_d J(x_1024) B_d^T; normalized=||J_d||_F/sqrt(d)",
        }
        for dimension in range(1, 5):
            key = f"projected_normalized_frobenius_d{dimension}"
            out.update(metric_stats([float(r[key]) for r in rows], key))
        summary.append(out)
    write_csv(args.summary, summary)

    # Prefer the dense-suite endpoint rows, then fill the early checkpoints from
    # the original Experiment 20 summary. Merge projected metrics by checkpoint.
    endpoint_by_checkpoint = {r["checkpoint"]: r for r in read_csv(BASE_ENDPOINT) if int(r["training_step"]) > 0}
    endpoint_by_checkpoint.update({r["checkpoint"]: r for r in read_csv(DENSE_ENDPOINT)})
    projected_by_checkpoint = {r["checkpoint"]: r for r in summary}
    combined = []
    for checkpoint in checkpoints:
        row = dict(endpoint_by_checkpoint[checkpoint])
        for key, value in projected_by_checkpoint[checkpoint].items():
            if key not in {"checkpoint", "training_step", "dynamic_step", "token_count", "token_selection", "token_seed"}:
                row[key] = value
        combined.append(row)
    write_csv(args.combined, combined)
    print(json.dumps({"status": "summary_complete", "checkpoints": len(summary), "summary": str(args.summary), "combined": str(args.combined)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compute", "summarize", "all"), default="all")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument("--manifest", type=Path, default=SOURCE_EXP20 / "manifests/tokens_100_frequency_deciles.csv")
    parser.add_argument("--cache-dir", type=Path, default=exp20.base.DEFAULT_CACHE)
    parser.add_argument("--parts", type=Path, default=PARTS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--combined", type=Path, default=COMBINED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoints = args.checkpoints or checkpoint_list()
    checkpoints = sorted(set(checkpoints), key=lambda x: int(x.removeprefix("step")))
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must satisfy 0 <= index < num-shards")
    selected = checkpoints[args.shard_index :: args.num_shards]
    if args.mode in {"compute", "all"}:
        for checkpoint in selected:
            compute_checkpoint(args, checkpoint)
    if args.mode in {"summarize", "all"}:
        if args.num_shards != 1 and args.mode == "all":
            print(json.dumps({"status": "worker_complete", "shard": args.shard_index, "checkpoints": len(selected)}), flush=True)
        else:
            summarize(args, checkpoints)


if __name__ == "__main__":
    main()
