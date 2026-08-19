#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts._bootstrap import require_packages

require_packages(["datasets", "matplotlib", "numpy", "torch", "transformers"])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analyze_convergence_neighbors import (
    DEFAULT_ARROW,
    DEFAULT_CACHE,
    confidence_label,
    decode_topk,
    prediction_description,
    wikitext_train_counts,
    write_csv,
)


# Equal-spacing targets are mapped to the nearest locally cached real revision.
CHECKPOINT_GRID = [
    {"checkpoint": "step0", "target_step": 0, "interval": "baseline"},
    {"checkpoint": "step1000", "target_step": 1000, "interval": "endpoint"},
    {"checkpoint": "step5000", "target_step": 4750, "interval": "1000_16000_interp1"},
    {"checkpoint": "step9000", "target_step": 8500, "interval": "1000_16000_interp2"},
    {"checkpoint": "step10000", "target_step": 10000, "interval": "local_fine_scan"},
    {"checkpoint": "step13000", "target_step": 12250, "interval": "1000_16000_interp3"},
    {"checkpoint": "step16000", "target_step": 16000, "interval": "endpoint"},
    {"checkpoint": "step37000", "target_step": 37167, "interval": "16000_143000_interp1"},
    {"checkpoint": "step57000", "target_step": 58333, "interval": "16000_143000_interp2"},
    {"checkpoint": "step81000", "target_step": 79500, "interval": "16000_143000_interp3"},
    {"checkpoint": "step101000", "target_step": 100667, "interval": "16000_143000_interp4"},
    {"checkpoint": "step121000", "target_step": 121833, "interval": "16000_143000_interp5"},
    {"checkpoint": "step143000", "target_step": 143000, "interval": "endpoint"},
]


def lexical(decoded: str) -> bool:
    return any(character.isalnum() for character in decoded)


def select_random_tokens(
    counts: Counter[int],
    tokenizer,
    count: int,
    seed: int,
) -> list[dict]:
    special = set(int(value) for value in tokenizer.all_special_ids)
    candidates = [
        int(token_id)
        for token_id in counts
        if int(token_id) not in special and lexical(tokenizer.decode([int(token_id)]))
    ]
    selected = random.Random(seed).sample(sorted(candidates), count)
    return [
        {
            "selection_index": index,
            "token_id": token_id,
            "token": tokenizer.decode([token_id]),
            "wikitext_train_count": int(counts[token_id]),
        }
        for index, token_id in enumerate(selected)
    ]


def fixed_projection(hidden_size: int, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vectors = torch.randn((count, hidden_size), generator=generator, dtype=torch.float32)
    return torch.nn.functional.normalize(vectors, dim=1)


def analyze_centers(
    centers: torch.Tensor,
    model,
    tokenizer,
    counts: Counter[int],
    selected: list[dict],
    checkpoint_row: dict,
    tail_radius_mean: torch.Tensor,
    tail_radius_max: torch.Tensor,
) -> list[dict]:
    input_weight = model.get_input_embeddings().weight.detach().float()
    output_weight = model.get_output_embeddings().weight.detach().float()
    normalized_centers = torch.nn.functional.normalize(centers, dim=1)
    normalized_input = torch.nn.functional.normalize(input_weight, dim=1)
    cosine = normalized_centers @ normalized_input.T
    special_ids = [int(value) for value in tokenizer.all_special_ids]
    if special_ids:
        cosine[:, special_ids] = -torch.inf
    cosine_values, cosine_ids = cosine.topk(5, dim=1)
    finite_cosine = torch.where(torch.isfinite(cosine), cosine, torch.nan)
    cosine_mean = torch.nanmean(finite_cosine, dim=1)
    cosine_std = torch.sqrt(
        torch.nanmean((finite_cosine - cosine_mean[:, None]).square(), dim=1)
    )

    logits = centers @ output_weight.T
    probabilities = torch.softmax(logits, dim=1)
    lm_values, lm_ids = probabilities.topk(5, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    normalized_entropy = -(probabilities * log_probabilities).sum(dim=1) / math.log(
        logits.shape[1]
    )
    logit_top2 = logits.topk(2, dim=1).values

    rows: list[dict] = []
    for index, source in enumerate(selected):
        neighbor_id = int(cosine_ids[index, 0])
        prediction_id = int(lm_ids[index, 0])
        cosine_top1 = float(cosine_values[index, 0])
        cosine_margin = cosine_top1 - float(cosine_values[index, 1])
        cosine_z = (cosine_top1 - float(cosine_mean[index])) / max(
            float(cosine_std[index]), 1e-12
        )
        probability = float(lm_values[index, 0])
        rows.append(
            {
                **checkpoint_row,
                **source,
                "center_norm": float(centers[index].norm()),
                "center_tail_radius_mean": float(tail_radius_mean[index]),
                "center_tail_radius_max": float(tail_radius_max[index]),
                "center_tail_relative_radius_mean": float(
                    tail_radius_mean[index] / centers[index].norm().clamp_min(1e-12)
                ),
                "cosine_neighbor_token_id": neighbor_id,
                "cosine_neighbor_token": tokenizer.decode([neighbor_id]),
                "cosine_neighbor_wikitext_train_count": int(counts[neighbor_id]),
                "cosine_similarity": cosine_top1,
                "cosine_top1_top2_margin": cosine_margin,
                "cosine_vocab_zscore": cosine_z,
                "geometry_confidence": confidence_label(cosine_z, cosine_margin),
                "cosine_top5": decode_topk(
                    tokenizer, cosine_ids[index].cpu(), cosine_values[index].cpu()
                ),
                "lm_top1_token_id": prediction_id,
                "lm_top1_token": tokenizer.decode([prediction_id]),
                "lm_top1_wikitext_train_count": int(counts[prediction_id]),
                "lm_top1_probability": probability,
                "lm_top1_top2_logit_margin": float(
                    logit_top2[index, 0] - logit_top2[index, 1]
                ),
                "lm_normalized_entropy": float(normalized_entropy[index]),
                "lm_top5": decode_topk(tokenizer, lm_ids[index].cpu(), lm_values[index].cpu()),
                "cosine_neighbor_matches_lm_top1": neighbor_id == prediction_id,
                "similarity_prediction_description": prediction_description(
                    neighbor_id, prediction_id, probability
                ),
            }
        )
    return rows


def final_vector_rows(center_style_rows: list[dict], final_step: int) -> list[dict]:
    output: list[dict] = []
    for row in center_style_rows:
        output.append(
            {
                "checkpoint": row["checkpoint"],
                "target_step": row["target_step"],
                "interval": row["interval"],
                "selection_index": row["selection_index"],
                "token_id": row["token_id"],
                "token": row["token"],
                "wikitext_train_count": row["wikitext_train_count"],
                "final_dynamic_step": final_step,
                "final_vector_norm": row["center_norm"],
                "final_cosine_neighbor_token_id": row["cosine_neighbor_token_id"],
                "final_cosine_neighbor_token": row["cosine_neighbor_token"],
                "final_cosine_neighbor_wikitext_train_count": row[
                    "cosine_neighbor_wikitext_train_count"
                ],
                "final_cosine_similarity": row["cosine_similarity"],
                "final_cosine_top1_top2_margin": row["cosine_top1_top2_margin"],
                "final_cosine_vocab_zscore": row["cosine_vocab_zscore"],
                "final_geometry_confidence": row["geometry_confidence"],
                "final_cosine_top5": row["cosine_top5"],
                "final_lm_top1_token_id": row["lm_top1_token_id"],
                "final_lm_top1_token": row["lm_top1_token"],
                "final_lm_top1_wikitext_train_count": row[
                    "lm_top1_wikitext_train_count"
                ],
                "final_lm_top1_probability": row["lm_top1_probability"],
                "final_lm_top1_top2_logit_margin": row["lm_top1_top2_logit_margin"],
                "final_lm_normalized_entropy": row["lm_normalized_entropy"],
                "final_lm_top5": row["lm_top5"],
                "final_cosine_neighbor_matches_lm_top1": row[
                    "cosine_neighbor_matches_lm_top1"
                ],
            }
        )
    return output


def projection_diagnostic_rows(
    checkpoint_row: dict,
    selected: list[dict],
    tail: torch.Tensor,
    centers: torch.Tensor,
    projection: torch.Tensor,
    tail_step_deltas: torch.Tensor,
) -> list[dict]:
    deviations = tail - centers.unsqueeze(0)
    full_rms = torch.sqrt(deviations.square().sum(dim=2).mean(dim=0))
    projected = torch.einsum("tbh,ph->tbp", deviations, projection)
    pair_count = projection.shape[0] // 2
    paired = projected.reshape(projected.shape[0], projected.shape[1], pair_count, 2)
    projected_rms = torch.sqrt(paired.square().sum(dim=3).mean(dim=0))
    ratio = projected_rms / full_rms[:, None].clamp_min(1e-30)
    expected_ratio = math.sqrt(2.0 / tail.shape[2])
    rows: list[dict] = []
    for index, token in enumerate(selected):
        rows.append(
            {
                **checkpoint_row,
                **token,
                "hidden_size": int(tail.shape[2]),
                "projection_pairs": pair_count,
                "full_tail_deviation_rms": float(full_rms[index]),
                "full_tail_step_delta_mean": float(tail_step_deltas[:, index].mean()),
                "full_tail_step_delta_final": float(tail_step_deltas[-1, index]),
                "expected_random_2d_capture_ratio": expected_ratio,
                "projection_capture_ratio_min": float(ratio[index].min()),
                "projection_capture_ratio_median": float(ratio[index].median()),
                "projection_capture_ratio_max": float(ratio[index].max()),
            }
        )
    return rows


def plot_checkpoint(
    checkpoint: str,
    trajectory_rows: list[dict],
    selected: list[dict],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    final_x = []
    final_y = []
    for token in selected:
        group = [
            row for row in trajectory_rows if int(row["token_id"]) == int(token["token_id"])
        ]
        group.sort(key=lambda row: int(row["dynamic_step"]))
        final_x.append(float(group[-1]["projection_0"]))
        final_y.append(float(group[-1]["projection_1"]))
    reference_x = float(np.mean(final_x))
    reference_y = float(np.mean(final_y))
    for index, token in enumerate(selected):
        group = [
            row for row in trajectory_rows if int(row["token_id"]) == int(token["token_id"])
        ]
        group.sort(key=lambda row: int(row["dynamic_step"]))
        x = np.asarray([float(row["projection_0"]) for row in group]) - reference_x
        y = np.asarray([float(row["projection_1"]) for row in group]) - reference_y
        color = cmap(index)
        ax.plot(
            x,
            y,
            color=color,
            lw=1.25,
            alpha=0.88,
            label=f"{token['token']!r} (count={token['wikitext_train_count']})",
        )
        ax.scatter(x[0], y[0], marker="o", s=45, color=color, edgecolor="black", zorder=4)
        ax.scatter(x[-1], y[-1], marker="X", s=62, color=color, edgecolor="black", zorder=5)
    ax.set_xlabel("Δ fixed projection 0 from mean final point")
    ax.set_ylabel("Δ fixed projection 1 from mean final point")
    ax.set_title(f"{checkpoint}: centered 2D projection; 8 random tokens; steps 512–768")
    ax.ticklabel_format(style="sci", scilimits=(-3, 3), useOffset=False)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_checkpoint(
    checkpoint_row: dict,
    selected: list[dict],
    tokenizer,
    counts: Counter[int],
    cache_dir: Path,
    device: torch.device,
    steps: int,
    eval_start: int,
    tail_size: int,
    projection: torch.Tensor,
    figures: Path,
) -> tuple[list[dict], list[dict], list[dict], list[dict], torch.Tensor, torch.Tensor, float]:
    checkpoint = str(checkpoint_row["checkpoint"])
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        revision=checkpoint,
        cache_dir=str(cache_dir),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    token_ids = torch.tensor([row["token_id"] for row in selected], device=device)
    state = model.get_input_embeddings()(token_ids).detach().float()
    attention_mask = torch.ones((len(selected), 1), device=device, dtype=torch.long)
    position_ids = torch.zeros((1, 1), device=device, dtype=torch.long)
    projection_device = projection.to(device)
    trajectory_rows: list[dict] = []
    tail_states: list[torch.Tensor] = []
    tail_step_deltas: list[torch.Tensor] = []
    previous_state = state.detach().clone()

    with torch.inference_mode():
        for dynamic_step in range(1, steps + 1):
            state = model.gpt_neox(
                inputs_embeds=state.to(dtype=next(model.parameters()).dtype).unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1, :].float()
            step_delta = torch.linalg.vector_norm(state - previous_state, dim=1)
            previous_state = state.detach().clone()
            if dynamic_step >= eval_start:
                projected = state @ projection_device.T
                for index, token in enumerate(selected):
                    trajectory_rows.append(
                        {
                            **checkpoint_row,
                            "selection_index": index,
                            "token_id": int(token["token_id"]),
                            "token": token["token"],
                            "wikitext_train_count": int(token["wikitext_train_count"]),
                            "dynamic_step": dynamic_step,
                            "projection_0": float(projected[index, 0]),
                            "projection_1": float(projected[index, 1]),
                        }
                    )
            if dynamic_step > steps - tail_size:
                tail_states.append(state.detach().cpu())
                tail_step_deltas.append(step_delta.detach().cpu())

    tail = torch.stack(tail_states)
    tail_deltas = torch.stack(tail_step_deltas)
    centers = tail.mean(dim=0)
    radii = torch.linalg.vector_norm(tail - centers.unsqueeze(0), dim=2)
    center_rows = analyze_centers(
        centers.to(device),
        model,
        tokenizer,
        counts,
        selected,
        checkpoint_row,
        radii.mean(dim=0).to(device),
        radii.max(dim=0).values.to(device),
    )
    zeros = torch.zeros(len(selected), device=device)
    final_style_rows = analyze_centers(
        state.detach(),
        model,
        tokenizer,
        counts,
        selected,
        checkpoint_row,
        zeros,
        zeros,
    )
    final_rows = final_vector_rows(final_style_rows, steps)
    diagnostic_rows = projection_diagnostic_rows(
        checkpoint_row,
        selected,
        tail,
        centers,
        projection,
        tail_deltas,
    )
    plot_checkpoint(
        checkpoint,
        trajectory_rows,
        selected,
        figures / f"random8_projection_{checkpoint}.png",
    )
    runtime = time.perf_counter() - started
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return trajectory_rows, center_rows, final_rows, diagnostic_rows, centers, state.detach().cpu(), runtime


def summarize_checkpoint(
    checkpoint_row: dict,
    rows: list[dict],
    centers: torch.Tensor,
    runtime: float,
) -> dict:
    centroid = centers.mean(dim=0)
    distances = torch.linalg.vector_norm(centers - centroid, dim=1)
    pairwise = torch.pdist(centers)
    return {
        **checkpoint_row,
        "n_tokens": len(rows),
        "median_center_tail_relative_radius": float(
            np.median([row["center_tail_relative_radius_mean"] for row in rows])
        ),
        "across_token_center_distance_relative_mean": float(
            distances.mean() / centroid.norm().clamp_min(1e-12)
        ),
        "across_token_pairwise_distance_max": float(pairwise.max()),
        "median_cosine_similarity": float(
            np.median([row["cosine_similarity"] for row in rows])
        ),
        "median_lm_top1_probability": float(
            np.median([row["lm_top1_probability"] for row in rows])
        ),
        "unique_cosine_neighbors": len({row["cosine_neighbor_token_id"] for row in rows}),
        "unique_lm_top1": len({row["lm_top1_token_id"] for row in rows}),
        "runtime_seconds": runtime,
    }


def plot_projection_diagnostic_summary(rows: list[dict], output: Path) -> None:
    checkpoints = [str(row["checkpoint"]) for row in CHECKPOINT_GRID]
    steps = [int(str(checkpoint).removeprefix("step")) for checkpoint in checkpoints]
    medians: dict[str, list[float]] = {
        "full_tail_deviation_rms": [],
        "full_tail_step_delta_mean": [],
        "projection_capture_ratio_median": [],
    }
    for checkpoint in checkpoints:
        group = [row for row in rows if row["checkpoint"] == checkpoint]
        for metric in medians:
            medians[metric].append(
                float(np.median([float(row[metric]) for row in group]))
            )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    axes[0].plot(steps, medians["full_tail_deviation_rms"], marker="o", lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Full-512D tail deviation RMS")
    axes[1].plot(steps, medians["full_tail_step_delta_mean"], marker="o", lw=1.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Full-512D tail step delta")
    axes[2].plot(steps, medians["projection_capture_ratio_median"], marker="o", lw=1.8)
    axes[2].axhline(math.sqrt(2.0 / 512.0), color="#9c3f35", ls="--", label="sqrt(2/512)")
    axes[2].set_ylabel("Median 2D/full capture ratio")
    axes[2].legend()
    for ax in axes:
        ax.set_xscale("symlog", linthresh=1000)
        ax.set_xlim(0, 150000)
        ax.set_xlabel("Training checkpoint step")
        ax.grid(alpha=0.25, which="both")
    fig.suptitle("Projection audit: high-dimensional motion versus 16 random 2D projections")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wikitext-train-arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--eval-start", type=int, default=512)
    parser.add_argument("--tail-size", type=int, default=64)
    args = parser.parse_args()

    processed = args.output_root / "processed"
    figures = args.output_root / "figures" / "random8_checkpoint_scan"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        revision="step100000",
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    counts = wikitext_train_counts(args.wikitext_train_arrow, tokenizer)
    selected = select_random_tokens(counts, tokenizer, args.token_count, args.seed)
    projection = fixed_projection(512, 32, seed=1234)
    all_trajectory: list[dict] = []
    all_centers: list[dict] = []
    all_finals: list[dict] = []
    all_projection_diagnostics: list[dict] = []
    summaries: list[dict] = []
    center_tensors: dict[str, torch.Tensor] = {}
    final_tensors: dict[str, torch.Tensor] = {}

    for checkpoint_row in CHECKPOINT_GRID:
        trajectory, center_rows, final_rows, diagnostic_rows, centers, finals, runtime = run_checkpoint(
            checkpoint_row,
            selected,
            tokenizer,
            counts,
            args.cache_dir,
            device,
            args.steps,
            args.eval_start,
            args.tail_size,
            projection,
            figures,
        )
        all_trajectory.extend(trajectory)
        all_centers.extend(center_rows)
        all_finals.extend(final_rows)
        all_projection_diagnostics.extend(diagnostic_rows)
        center_tensors[str(checkpoint_row["checkpoint"])] = centers
        final_tensors[str(checkpoint_row["checkpoint"])] = finals
        summaries.append(summarize_checkpoint(checkpoint_row, center_rows, centers, runtime))
        print(
            json.dumps(
                {
                    "checkpoint": checkpoint_row["checkpoint"],
                    "runtime_seconds": runtime,
                    "completed": len(summaries),
                    "total": len(CHECKPOINT_GRID),
                }
            ),
            flush=True,
        )

    write_csv(processed / "random8_selected_tokens.csv", selected)
    write_csv(processed / "random8_checkpoint_trajectory.csv", all_trajectory)
    write_csv(processed / "random8_convergence_center_neighbors.csv", all_centers)
    write_csv(processed / "random8_final_vector_neighbors.csv", all_finals)
    write_csv(processed / "random8_projection_diagnostics.csv", all_projection_diagnostics)
    write_csv(processed / "random8_checkpoint_summary.csv", summaries)
    plot_projection_diagnostic_summary(
        all_projection_diagnostics,
        args.output_root / "figures" / "random8_projection_audit.png",
    )
    torch.save(
        {
            "checkpoint_grid": CHECKPOINT_GRID,
            "selected_tokens": selected,
            "selection_seed": args.seed,
            "projection_seed": 1234,
            "centers": center_tensors,
            "final_vectors": final_tensors,
        },
        processed / "random8_convergence_centers.pt",
    )
    metadata = {
        "status": "complete",
        "checkpoint_grid": CHECKPOINT_GRID,
        "selection": "uniform random sample without replacement from observed lexical non-special token types",
        "selection_seed": args.seed,
        "selected_tokens": selected,
        "steps": args.steps,
        "eval_start": args.eval_start,
        "tail_size": args.tail_size,
        "device": str(device),
        "trajectory_rows": len(all_trajectory),
        "center_rows": len(all_centers),
        "final_vector_rows": len(all_finals),
        "projection_diagnostic_rows": len(all_projection_diagnostics),
    }
    (args.output_root / "random8_scan_complete.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
