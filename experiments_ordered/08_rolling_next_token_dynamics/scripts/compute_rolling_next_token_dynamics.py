#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpoint_utils import finite

from scripts._bootstrap import require_packages

require_packages(["torch", "transformers", "datasets", "yaml"])

import torch

from src.io_utils import base_metadata, load_config, setup_storage_env, write_jsonl
from src.model_utils import load_model_and_tokenizer
from src.rolling_dynamics import (
    SoftNextTokenRollingOperator,
    estimate_innovation_frobenius,
    estimate_maximal_lyapunov,
    hard_argmax_rollout,
    run_soft_trajectory,
)


def reconstruct_documents(dataset: Any, text_column: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    current: list[str] = []
    title = "untitled"
    for row in dataset:
        text = str(row[text_column])
        stripped = text.strip()
        if stripped.startswith("=") and stripped.endswith("=") and len(stripped) > 2:
            if current:
                documents.append({"document_index": len(documents), "title": title, "text": "\n".join(current)})
            title = stripped
            current = [text]
        elif current or stripped:
            current.append(text)
    if current:
        documents.append({"document_index": len(documents), "title": title, "text": "\n".join(current)})
    return documents


def projection_vectors(reference: torch.Tensor, count: int, seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    full: list[torch.Tensor] = []
    newest: list[torch.Tensor] = []
    for _ in range(int(count)):
        q = torch.randn(reference.shape, device=reference.device, dtype=torch.float32, generator=generator)
        full.append(q / (q.norm() + 1e-12))
        qn = torch.randn(reference[:, -1, :].shape, device=reference.device, dtype=torch.float32, generator=generator)
        newest.append(qn / (qn.norm() + 1e-12))
    return full, newest


def add_projections(
    rows: list[dict[str, Any]],
    states: list[torch.Tensor],
    full_vectors: list[torch.Tensor],
    newest_vectors: list[torch.Tensor],
) -> None:
    for row, state in zip(rows, states):
        for idx, vector in enumerate(full_vectors):
            row[f"projection_full_{idx}"] = float((state.float() * vector).sum().cpu())
        newest_state = state[:, -1, :].float()
        for idx, vector in enumerate(newest_vectors):
            row[f"projection_newest_{idx}"] = float((newest_state * vector).sum().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_storage_env(config)
    # datasets reads HF_DATASETS_OFFLINE during import.  Import only after
    # setup_storage_env so an offline run never performs a network HEAD request.
    from datasets import load_dataset

    output_dir = Path(config["output_dir"])
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg = config["dataset"]
    rolling_cfg = config["rolling_operator"]
    metrics_cfg = config["metrics"]
    cache_dir = config.get("cache_dir")
    offline = bool(config.get("offline", False))
    dataset = load_dataset(
        dataset_cfg["name"],
        dataset_cfg.get("config"),
        split=dataset_cfg["split"],
        cache_dir=cache_dir,
    )
    documents = reconstruct_documents(dataset, dataset_cfg.get("text_column", "text"))
    document_indices = [int(v) for v in dataset_cfg["document_indices"]]
    anchor_offsets = [int(v) for v in dataset_cfg["anchor_offsets"]]
    window_length = int(dataset_cfg["window_length"])
    manifest_rows: list[dict[str, Any]] = []

    for model_cfg in config["models"]:
        model_name = str(model_cfg["name"])
        revision = str(model_cfg["revision"])
        tokenizer_name = str(model_cfg.get("tokenizer", model_name))
        model, tokenizer, device = load_model_and_tokenizer(
            model_name,
            revision,
            tokenizer_name,
            config.get("dtype", "float32"),
            config.get("device", "cuda"),
            cache_dir=cache_dir,
            attn_implementation=config.get("attn_implementation"),
            local_files_only=offline,
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        operator = SoftNextTokenRollingOperator(
            model,
            temperature=float(rolling_cfg.get("temperature", 1.0)),
            position_mode=str(rolling_cfg.get("position_mode", "reset")),
        )

        summary_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        hard_rows: list[dict[str, Any]] = []
        anchor_index = 0
        for document_index in document_indices:
            document = documents[document_index]
            token_ids = tokenizer(document["text"], add_special_tokens=False)["input_ids"]
            for anchor_offset in anchor_offsets:
                stop = anchor_offset + window_length
                if stop > len(token_ids):
                    raise ValueError(
                        f"document {document_index} has {len(token_ids)} tokens, cannot use anchor {anchor_offset}"
                    )
                ids = torch.tensor([token_ids[anchor_offset:stop]], device=device, dtype=torch.long)
                with torch.no_grad():
                    x0 = model.get_input_embeddings()(ids).float()
                row_base = {
                    **base_metadata(config, model_name, revision, tokenizer_name, window_length),
                    "document_index": int(document_index),
                    "document_title": document["title"],
                    "document_token_count": len(token_ids),
                    "anchor_index": anchor_index,
                    "anchor_offset": anchor_offset,
                    "window_length": window_length,
                    "hidden_size": int(x0.shape[-1]),
                    "active_dim": int(x0.shape[1] * x0.shape[2]),
                    "position_mode": rolling_cfg.get("position_mode", "reset"),
                    "temperature": float(rolling_cfg.get("temperature", 1.0)),
                    "initial_token_ids": [int(v) for v in ids[0].tolist()],
                }
                manifest_rows.append(row_base)
                torch.manual_seed(int(config.get("seed", 1234)) + anchor_index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(config.get("seed", 1234)) + anchor_index)
                start = time.perf_counter()
                trajectory = run_soft_trajectory(
                    operator,
                    x0,
                    burn_in_steps=int(rolling_cfg["burn_in_steps"]),
                    eval_steps=int(rolling_cfg["eval_steps"]),
                    epsilon=float(metrics_cfg.get("perturbation_epsilon", 1e-3)),
                    seed=int(config.get("seed", 1234)) + 1000 + anchor_index,
                )
                soft_seconds = time.perf_counter() - start
                full_vectors, newest_vectors = projection_vectors(
                    x0,
                    int(metrics_cfg.get("projection_count", 3)),
                    int(metrics_cfg.get("projection_seed", 1234)),
                )
                add_projections(trajectory.rows, trajectory.states, full_vectors, newest_vectors)
                for row in trajectory.rows:
                    trajectory_rows.append({**row_base, **row})

                frobenius: dict[str, Any] = {}
                frobenius_seconds = 0.0
                if bool(metrics_cfg.get("compute_frobenius", True)):
                    selected_states_count = int(metrics_cfg.get("frobenius_eval_states", 2))
                    selected_states = trajectory.states[-selected_states_count:]
                    start = time.perf_counter()
                    frobenius = estimate_innovation_frobenius(
                        operator,
                        selected_states,
                        probes=int(metrics_cfg.get("frobenius_probes", 4)),
                        seed=int(config.get("seed", 1234)) + 2000 + anchor_index,
                    )
                    frobenius_seconds = time.perf_counter() - start

                lyapunov: dict[str, Any] = {}
                lyapunov_seconds = 0.0
                if bool(metrics_cfg.get("compute_lyapunov", True)):
                    start = time.perf_counter()
                    lyapunov = estimate_maximal_lyapunov(
                        operator,
                        trajectory.states,
                        probes=int(metrics_cfg.get("lyapunov_probes", 1)),
                        seed=int(config.get("seed", 1234)) + 3000 + anchor_index,
                    )
                    lyapunov_seconds = time.perf_counter() - start

                hard: dict[str, Any] = {}
                hard_seconds = 0.0
                if bool(metrics_cfg.get("compute_hard", True)):
                    start = time.perf_counter()
                    hard = hard_argmax_rollout(model, ids, steps=int(rolling_cfg.get("hard_steps", 256)))
                    hard_seconds = time.perf_counter() - start

                tail = trajectory.rows[-min(16, len(trajectory.rows)) :]
                final_nearby = float(trajectory.rows[-1]["nearby_distance"])
                summary = {
                    **row_base,
                    "burn_in_steps": int(rolling_cfg["burn_in_steps"]),
                    "eval_steps": int(rolling_cfg["eval_steps"]),
                    "perturbation_epsilon": float(metrics_cfg.get("perturbation_epsilon", 1e-3)),
                    "tail_relative_step_delta_mean": sum(float(r["relative_step_delta"]) for r in tail) / len(tail),
                    "tail_soft_entropy_mean": sum(float(r["soft_entropy"]) for r in tail) / len(tail),
                    "tail_soft_top1_probability_mean": sum(float(r["soft_top1_probability"]) for r in tail) / len(tail),
                    "final_nearby_distance": final_nearby,
                    "final_to_initial_separation": final_nearby / max(trajectory.initial_nearby_distance, 1e-12),
                    "soft_seconds": soft_seconds,
                    "frobenius_seconds": frobenius_seconds,
                    "lyapunov_seconds": lyapunov_seconds,
                    "hard_seconds": hard_seconds,
                    **frobenius,
                    **lyapunov,
                    **{k: v for k, v in hard.items() if k != "generated_token_ids"},
                }
                if "maximal_lyapunov_mean" in summary:
                    summary["maximal_lyapunov_mean"] = finite(summary["maximal_lyapunov_mean"])
                    summary["maximal_lyapunov_max"] = finite(summary["maximal_lyapunov_max"])
                    summary["lyapunov_exponents"] = [finite(v) for v in summary["lyapunov_exponents"]]
                summary_rows.append(summary)
                if hard:
                    hard_rows.append({**row_base, **hard})
                print(
                    json.dumps(
                        {
                            "checkpoint": revision,
                            "anchor": anchor_index,
                            "total_frob": summary.get("total_geomean"),
                            "innovation": summary.get("innovation_geomean"),
                            "lyapunov": summary.get("maximal_lyapunov_mean"),
                            "hard_cycle": summary.get("hard_cycle_length"),
                            "seconds": soft_seconds + frobenius_seconds + lyapunov_seconds + hard_seconds,
                        }
                    ),
                    flush=True,
                )
                anchor_index += 1

        prefix = f"{config['experiment_name']}__{revision}"
        write_jsonl(raw_dir / f"{prefix}__summary.jsonl", summary_rows)
        write_jsonl(raw_dir / f"{prefix}__trajectory.jsonl", trajectory_rows)
        write_jsonl(raw_dir / f"{prefix}__hard.jsonl", hard_rows)
        write_jsonl(raw_dir / f"{prefix}__manifest.jsonl", manifest_rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
