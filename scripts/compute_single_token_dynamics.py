#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpoint_utils import finite
from src.experiment_io import read_jsonl

from scripts._bootstrap import require_packages

require_packages(["torch", "transformers", "yaml"])

import torch

from src.io_utils import base_metadata, load_config, setup_storage_env, write_jsonl
from src.model_utils import load_model_and_tokenizer
from src.single_token_dynamics import (
    SingleTokenOperator,
    causal_cross_gradient_max,
    classify_convergence,
    estimate_conditional_lyapunov,
    estimate_hutchinson_frobenius,
    exact_target_jacobian,
    fixed_projection_vectors,
    jacobian_summary,
    run_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    setup_storage_env(config)
    output = Path(config["output_dir"])
    raw = output / "raw"
    matrices = output / "jacobians"
    states_dir = output / "states"
    for directory in (raw, matrices, states_dir):
        directory.mkdir(parents=True, exist_ok=True)

    selected = read_jsonl(Path(config["frequency_manifest"]))
    if bool(config.get("sampling", {}).get("one_per_bin", False)):
        seen_bins: set[int] = set()
        selected = [row for row in selected if not (int(row["frequency_bin"]) in seen_bins or seen_bins.add(int(row["frequency_bin"])))]
    limit = config.get("sampling", {}).get("token_limit")
    if limit is not None:
        selected = selected[: int(limit)]
    shard_count = int(args.shard_count or config.get("sampling", {}).get("shard_count", 1))
    shard_index = int(args.shard_index if args.shard_index is not None else config.get("sampling", {}).get("shard_index", 0))
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(f"invalid shard {shard_index}/{shard_count}")
    selected = [row for index, row in enumerate(selected) if index % shard_count == shard_index]
    if not selected:
        raise RuntimeError(f"shard {shard_index}/{shard_count} selected no tokens")
    model_cfg = config["models"][0]
    model, tokenizer, device = load_model_and_tokenizer(
        model_cfg["name"], model_cfg["revision"], model_cfg.get("tokenizer"),
        config.get("dtype", "float32"), config.get("device", "cuda"),
        cache_dir=config.get("cache_dir"), attn_implementation=config.get("attn_implementation"),
        local_files_only=bool(config.get("offline", False)),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding = model.get_input_embeddings()
    dynamics = config["dynamics"]
    metrics = config["metrics"]
    groups = list(dynamics.get("groups", ["isolated_token", "frozen_context", "dynamic_context"]))
    steps = int(dynamics["steps"])
    eval_start = int(dynamics["eval_start"])
    tail_size = int(dynamics.get("tail_size", 32))
    contexts_per_token = int(config.get("sampling", {}).get("contexts_per_token", 2))
    exact_steps = set(int(value) for value in metrics.get("exact_steps", []))
    summary_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    jacobian_rows: list[dict] = []
    manifest_rows: list[dict] = []
    start_all = time.perf_counter()

    for token_index, token_row in enumerate(selected):
        token_id = int(token_row["token_id"])
        contexts = token_row["contexts"][:contexts_per_token]
        with torch.no_grad():
            target0 = embedding(torch.tensor([token_id], device=device)).float()[0]
        projections = fixed_projection_vectors(
            target0, int(metrics.get("projection_count", 4)), int(metrics.get("projection_seed", 1234))
        )
        for group in groups:
            group_contexts = [None] if group == "isolated_token" else contexts
            for context_index, context_ids in enumerate(group_contexts):
                if context_ids is None:
                    prefix = None
                    effective_context_id = -1
                else:
                    ids = torch.tensor(context_ids[:-1], device=device, dtype=torch.long)
                    with torch.no_grad():
                        prefix = embedding(ids).float()
                    effective_context_id = context_index
                operator = SingleTokenOperator(model, group, prefix)
                initial = operator.initial_state(target0)
                seed_base = int(config.get("seed", 1234)) + token_index * 100 + (context_index + 1) * 10
                started = time.perf_counter()
                trajectory = run_trajectory(
                    operator, initial, steps, float(metrics.get("perturbation_epsilon", 1e-3)),
                    seed_base, projections,
                )
                identifier = f"{group}__token{token_id}__context{effective_context_id}"
                state_path = states_dir / f"{identifier}.pt"
                torch.save({"target_states": torch.stack([operator.target(state).cpu() for state in trajectory.states])}, state_path)
                base = {
                    **base_metadata(config, model_cfg["name"], model_cfg["revision"], model_cfg.get("tokenizer"),
                                    1 if group == "isolated_token" else len(context_ids)),
                    "group": group,
                    "token_id": token_id,
                    "decoded": token_row["decoded"],
                    "frequency_count": int(token_row["count"]),
                    "frequency_bin": int(token_row["frequency_bin"]),
                    "context_id": effective_context_id,
                    "context_ids": None if context_ids is None else context_ids,
                    "hidden_size": int(target0.numel()),
                    "jacobian_target": "target_output_wrt_target_input",
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                }
                for row in trajectory.rows:
                    trajectory_rows.append({**base, **row})

                eval_states = trajectory.states[eval_start : steps]
                hutchinson_states = int(metrics.get("hutchinson_states", 8))
                if hutchinson_states > 0 and eval_states:
                    indices = torch.linspace(0, len(eval_states) - 1, hutchinson_states).round().long().tolist()
                    selected_states = [eval_states[index] for index in indices]
                else:
                    selected_states = []
                hutchinson = estimate_hutchinson_frobenius(
                    operator, selected_states, int(metrics.get("hutchinson_probes", 0)), seed_base + 2000
                )
                lyapunov_states = eval_states[: int(metrics.get("lyapunov_steps", len(eval_states)))]
                lyapunov = estimate_conditional_lyapunov(
                    operator, lyapunov_states, int(metrics.get("lyapunov_probes", 2)), seed_base + 3000
                )
                exact_contexts = set(int(value) for value in metrics.get("exact_contexts", [0]))
                compute_exact = group == "isolated_token" or effective_context_id in exact_contexts
                for exact_step in sorted(exact_steps if compute_exact else set()):
                    if exact_step < 0 or exact_step >= len(trajectory.states):
                        continue
                    jacobian = exact_target_jacobian(
                        operator, trajectory.states[exact_step], int(metrics.get("jacobian_chunk_size", 16))
                    )
                    if jacobian.shape != (target0.numel(), target0.numel()):
                        raise RuntimeError(f"invalid Jacobian shape {tuple(jacobian.shape)}")
                    matrix_path = matrices / f"{identifier}__step{exact_step}.pt"
                    torch.save(jacobian.cpu(), matrix_path)
                    jacobian_rows.append({**base, "trajectory_step": exact_step, "matrix_path": str(matrix_path),
                                          **jacobian_summary(jacobian)})

                eval_rows = trajectory.rows[eval_start - 1 : steps - 1] if eval_start > 0 else trajectory.rows[:steps]
                tail = eval_rows[-min(tail_size, len(eval_rows)) :]
                tail_relative = sum(float(row["relative_step_delta"]) for row in tail) / max(len(tail), 1)
                if eval_rows:
                    d0 = float(eval_rows[0]["nearby_distance"])
                    d1 = float(eval_rows[-1]["nearby_distance"])
                    state_scale = max(float(eval_rows[-1]["state_norm"]), 1.0)
                    resolution_floor = 8.0 * torch.finfo(torch.float32).eps * state_scale
                    numerical_floor = d0 <= resolution_floor or d1 <= resolution_floor
                    nearby_growth = None if numerical_floor else math.log(max(d1, 1e-30) / max(d0, 1e-30)) / max(len(eval_rows) - 1, 1)
                else:
                    d0 = d1 = float("nan")
                    resolution_floor = float("nan")
                    numerical_floor = True
                    nearby_growth = None
                label = classify_convergence(tail_relative, nearby_growth, lyapunov, numerical_floor)
                cross_gradient = causal_cross_gradient_max(operator, trajectory.states[0]) if group == "dynamic_context" else 0.0
                summary = {
                    **base,
                    "steps": steps,
                    "eval_start": eval_start,
                    "tail_relative_step_delta_mean": tail_relative,
                    "eval_nearby_distance_initial": finite(d0),
                    "eval_nearby_distance_final": finite(d1),
                    "nearby_log_growth_per_step": nearby_growth,
                    "nearby_resolution_floor": finite(resolution_floor),
                    "nearby_numerical_floor": numerical_floor,
                    "lyapunov_exponents": [finite(value) for value in lyapunov],
                    "lyapunov_mean": finite(sum(lyapunov) / len(lyapunov)) if lyapunov else None,
                    "hutchinson_normalized_frobenius": hutchinson,
                    "convergence_label": label,
                    "causal_cross_gradient_max": cross_gradient,
                    "runtime_seconds": time.perf_counter() - started,
                    "state_path": str(state_path),
                }
                summary_rows.append(summary)
                manifest_rows.append({**base, "state_path": str(state_path)})
                print(json.dumps({"checkpoint": model_cfg["revision"], "id": identifier, "label": label,
                                  "relative": tail_relative, "nearby_growth": nearby_growth,
                                  "lyapunov": summary["lyapunov_mean"], "seconds": summary["runtime_seconds"]}), flush=True)

    prefix = f"{config['experiment_name']}__{model_cfg['revision']}"
    write_jsonl(raw / f"{prefix}__summary.jsonl", summary_rows)
    write_jsonl(raw / f"{prefix}__trajectory.jsonl", trajectory_rows)
    write_jsonl(raw / f"{prefix}__jacobians.jsonl", jacobian_rows)
    write_jsonl(raw / f"{prefix}__manifest.jsonl", manifest_rows)
    (output / "run_complete.json").write_text(json.dumps({"rows": len(summary_rows), "seconds": time.perf_counter() - start_all}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
