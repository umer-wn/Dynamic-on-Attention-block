#!/usr/bin/env python3
"""Run paired original/perturbed single-token trajectories for Experiment 17."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
EXP17 = REPO / "experiments_ordered/17_visualize"
EXP16_SCRIPT = REPO / "experiments_ordered/16_frequency_stratified_window_jacobian/scripts/run_experiment16.py"
SPEC = importlib.util.spec_from_file_location("experiment16", EXP16_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {EXP16_SCRIPT}")
exp16 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:1"))
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--selection-index", type=int, default=0)
    parser.add_argument("--relative-scale", type=float, default=1e-6)
    parser.add_argument("--perturbation-seed", type=int, default=1904)
    parser.add_argument("--projection-seed", type=int, default=1616)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_checkpoint(args, checkpoint: str, token: dict, basis: torch.Tensor, output: Path):
    started = time.perf_counter()
    model = exp16.base.load_model(checkpoint, exp16.DEFAULT_CACHE, args.device)
    token_ids = torch.tensor([token["token_id"]], device=args.device)
    original = model.get_input_embeddings()(token_ids).detach().float()
    generator = torch.Generator(device="cpu").manual_seed(args.perturbation_seed)
    noise = torch.randn(original.shape, generator=generator).to(args.device)
    noise = torch.nn.functional.normalize(noise, dim=-1)
    state_scale = torch.linalg.vector_norm(original, dim=-1, keepdim=True).clamp_min(1.0)
    perturbed = original + args.relative_scale * state_scale * noise

    projected_original = torch.empty((args.steps + 1, 4), dtype=torch.float32)
    projected_perturbed = torch.empty((args.steps + 1, 4), dtype=torch.float32)
    distance = torch.empty((args.steps + 1,), dtype=torch.float32)
    relative_distance = torch.empty((args.steps + 1,), dtype=torch.float32)

    def record(step: int):
        o = original.detach().float().cpu()
        p = perturbed.detach().float().cpu()
        projected_original[step] = o @ basis.T
        projected_perturbed[step] = p @ basis.T
        delta = torch.linalg.vector_norm(p - o, dim=-1)[0]
        norm = torch.linalg.vector_norm(o, dim=-1)[0].clamp_min(1e-30)
        distance[step] = delta
        relative_distance[step] = delta / norm

    record(0)
    attention_mask = torch.ones((2, 1), device=args.device, dtype=torch.long)
    position_ids = torch.zeros((1, 1), device=args.device, dtype=torch.long)
    with torch.inference_mode():
        for step in range(1, args.steps + 1):
            joined = torch.cat([original, perturbed], dim=0)
            joined = model.gpt_neox(
                inputs_embeds=joined.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1, :].float()
            original, perturbed = joined.chunk(2, dim=0)
            record(step)

    payload = {
        "checkpoint": checkpoint,
        "steps": args.steps,
        "selection_index": token["selection_index"],
        "token_id": token["token_id"],
        "token": token["token"],
        "frequency_bin": token["frequency_bin"],
        "relative_scale": args.relative_scale,
        "perturbation_seed": args.perturbation_seed,
        "projection_seed": args.projection_seed,
        "original_projection": projected_original,
        "perturbed_projection": projected_perturbed,
        "full_state_distance": distance,
        "relative_distance": relative_distance,
    }
    torch.save(payload, output)
    seconds = time.perf_counter() - started
    print(json.dumps({
        "checkpoint": checkpoint,
        "status": "complete",
        "seconds": seconds,
        "initial_distance": float(distance[0]),
        "endpoint_distance": float(distance[-1]),
        "endpoint_relative_distance": float(relative_distance[-1]),
    }), flush=True)
    del model, original, perturbed
    torch.cuda.empty_cache()


def consolidate(args, checkpoints: list[str], raw_dir: Path):
    processed = EXP17 / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    csv_path = processed / "single_token_perturbation_projection_trajectories.csv"
    summary_path = processed / "single_token_perturbation_summary.json"
    fields = [
        "checkpoint", "dynamic_step", "selection_index", "token_id", "token",
        "frequency_bin", "relative_scale", "perturbation_seed", "projection_seed",
        "original_projection_1", "original_projection_2", "original_projection_3", "original_projection_4",
        "perturbed_projection_1", "perturbed_projection_2", "perturbed_projection_3", "perturbed_projection_4",
        "full_state_distance", "relative_distance",
    ]
    summaries = []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for checkpoint in checkpoints:
            payload = torch.load(raw_dir / f"{checkpoint}.pt", map_location="cpu", weights_only=True)
            o = payload["original_projection"]
            p = payload["perturbed_projection"]
            d = payload["full_state_distance"]
            rd = payload["relative_distance"]
            for step in range(args.steps + 1):
                row = {
                    "checkpoint": checkpoint,
                    "dynamic_step": step,
                    "selection_index": payload["selection_index"],
                    "token_id": payload["token_id"],
                    "token": payload["token"],
                    "frequency_bin": payload["frequency_bin"],
                    "relative_scale": payload["relative_scale"],
                    "perturbation_seed": payload["perturbation_seed"],
                    "projection_seed": payload["projection_seed"],
                    "full_state_distance": f"{float(d[step]):.10g}",
                    "relative_distance": f"{float(rd[step]):.10g}",
                }
                for index in range(4):
                    row[f"original_projection_{index + 1}"] = f"{float(o[step, index]):.10g}"
                    row[f"perturbed_projection_{index + 1}"] = f"{float(p[step, index]):.10g}"
                writer.writerow(row)
            summaries.append({
                "checkpoint": checkpoint,
                "initial_distance": float(d[0]),
                "endpoint_distance": float(d[-1]),
                "endpoint_relative_distance": float(rd[-1]),
                "max_distance": float(d.max()),
                "max_distance_step": int(torch.argmax(d)),
            })
    summary_path.write_text(json.dumps({
        "token_selection_index": args.selection_index,
        "relative_scale": args.relative_scale,
        "perturbation_seed": args.perturbation_seed,
        "projection_seed": args.projection_seed,
        "steps": args.steps,
        "checkpoints": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, summary_path, summaries


def main():
    args = parse_args()
    tokens = exp16.load_tokens(exp16.TOKEN_MANIFEST)
    matches = [token for token in tokens if token["selection_index"] == args.selection_index]
    if len(matches) != 1:
        raise ValueError(f"selection_index={args.selection_index} matched {len(matches)} tokens")
    token = matches[0]
    checkpoints = exp16.CHECKPOINTS
    basis = exp16.projection_basis(seed=args.projection_seed)
    raw_dir = EXP17 / "raw/single_token_perturbation_projection"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for index, checkpoint in enumerate(checkpoints):
        output = raw_dir / f"{checkpoint}.pt"
        if output.exists() and not args.overwrite:
            payload = torch.load(output, map_location="cpu", weights_only=True)
            compatible = (
                payload.get("steps") == args.steps
                and payload.get("selection_index") == args.selection_index
                and payload.get("relative_scale") == args.relative_scale
                and payload.get("perturbation_seed") == args.perturbation_seed
                and payload.get("projection_seed") == args.projection_seed
            )
            if compatible:
                print(json.dumps({"checkpoint": checkpoint, "status": "reuse", "completed": index + 1, "total": len(checkpoints)}), flush=True)
                continue
        run_checkpoint(args, checkpoint, token, basis, output)

    csv_path, summary_path, summaries = consolidate(args, checkpoints, raw_dir)
    print(json.dumps({
        "status": "complete",
        "checkpoints": len(checkpoints),
        "token": token,
        "csv": str(csv_path),
        "summary": str(summary_path),
        "endpoint_distance_range": [min(row["endpoint_distance"] for row in summaries), max(row["endpoint_distance"] for row in summaries)],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
