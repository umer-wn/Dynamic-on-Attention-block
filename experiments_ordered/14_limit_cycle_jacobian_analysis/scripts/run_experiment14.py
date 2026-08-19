#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINTS = [
    "step5000",
    "step9000",
    "step13000",
    "step16000",
    "step21000",
    "step29000",
    "step37000",
    "step45000",
    "step53000",
    "step61000",
    "step69000",
    "step77000",
]
DEFAULT_CACHE = Path("/home/luohaoming/model_feature_cache/hf_cache")
DEFAULT_ARROW = DEFAULT_CACHE / (
    "wikitext/wikitext-2-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow"
)
DEFAULT_DATA_ROOT = Path(
    "/home/luohaoming/model_feature_experiments/"
    "experiment14_limit_cycle_jacobian/trace0_2048"
)
DEFAULT_REPORT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT13_TOKENS = Path(__file__).resolve().parents[2] / (
    "13_single_token_convergence_neighbors/processed/random8_selected_tokens.csv"
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_tokens(path: Path, count: int = 4) -> list[dict]:
    rows = read_csv(path)[:count]
    if len(rows) != count:
        raise RuntimeError(f"expected {count} tokens in {path}, got {len(rows)}")
    return [
        {
            "selection_index": int(row["selection_index"]),
            "token_id": int(row["token_id"]),
            "token": row["token"],
            "wikitext_train_count": int(row["wikitext_train_count"]),
        }
        for row in rows
    ]


def wikitext_counts(path: Path, tokenizer) -> Counter[int]:
    dataset = Dataset.from_file(str(path))
    text = "\n".join(str(row["text"]) for row in dataset)
    return Counter(int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"])


def projection_basis(hidden: int = 512, seed: int = 1414) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn((hidden, 3), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.float()


def sampled_orbit_scale(
    states: torch.Tensor,
    seed: int,
    pairs: int = 8192,
) -> dict[str, float]:
    length = states.shape[0]
    generator = torch.Generator(device=states.device)
    generator.manual_seed(seed)
    left = torch.randint(0, length, (pairs,), device=states.device, generator=generator)
    right = torch.randint(0, length, (pairs,), device=states.device, generator=generator)
    distance = torch.linalg.vector_norm(states[left] - states[right], dim=1)
    # Two-sweep farthest-point estimate supplements the sampled maximum.
    anchor = states[0]
    first = torch.argmax(torch.linalg.vector_norm(states - anchor, dim=1))
    second_distances = torch.linalg.vector_norm(states - states[first], dim=1)
    diameter = max(float(distance.max()), float(second_distances.max()))
    return {
        "orbit_diameter_approx": diameter,
        "orbit_pairwise_d95": float(torch.quantile(distance, 0.95)),
        "orbit_pairwise_rms": float(torch.sqrt(distance.square().mean())),
    }


def recurrence_scan(
    states: torch.Tensor,
    token: dict,
    checkpoint: str,
    seed: int,
    max_period: int = 256,
) -> tuple[list[dict], dict]:
    scale_metrics = sampled_orbit_scale(states, seed)
    state_norm = torch.linalg.vector_norm(states, dim=1)
    numerical_floor = 8.0 * torch.finfo(torch.float32).eps * max(
        float(state_norm.median()), 1.0
    )
    scale = max(scale_metrics["orbit_pairwise_d95"], numerical_floor, 1e-12)
    rows: list[dict] = []
    for period in range(1, max_period + 1):
        absolute = torch.linalg.vector_norm(states[period:] - states[:-period], dim=1)
        normalized = absolute / scale
        rows.append(
            {
                "checkpoint": checkpoint,
                **token,
                "candidate_period": period,
                "absolute_median": float(absolute.median()),
                "absolute_p95": float(torch.quantile(absolute, 0.95)),
                "absolute_max": float(absolute.max()),
                "normalized_median": float(normalized.median()),
                "normalized_p95": float(torch.quantile(normalized, 0.95)),
                "normalized_max": float(normalized.max()),
            }
        )

    best = min(rows[1:], key=lambda row: row["normalized_p95"])
    p1 = rows[0]
    d95 = scale_metrics["orbit_pairwise_d95"]

    def fundamental_candidate(threshold: float) -> dict | None:
        for candidate in rows[1:]:
            period = int(candidate["candidate_period"])
            if candidate["normalized_p95"] > threshold:
                continue
            multiples = [
                rows[multiple - 1]
                for multiple in (2 * period, 3 * period)
                if multiple <= max_period
            ]
            if multiples and not all(row["normalized_p95"] <= threshold for row in multiples):
                continue
            return candidate
        return None

    strict_candidate = fundamental_candidate(1e-4)
    approximate_candidate = fundamental_candidate(1e-2)
    if d95 <= numerical_floor:
        label = "numerical_fixed_point"
        period = 1
    elif p1["normalized_p95"] <= 1e-4:
        label = "fixed_point"
        period = 1
    elif strict_candidate is not None:
        label = "strict_cycle_candidate"
        period = int(strict_candidate["candidate_period"])
    elif approximate_candidate is not None:
        label = "approximate_cycle_candidate"
        period = int(approximate_candidate["candidate_period"])
    else:
        label = "nonperiodic_or_unresolved"
        period = None
    summary = {
        "checkpoint": checkpoint,
        **token,
        **scale_metrics,
        "median_state_norm": float(state_norm.median()),
        "numerical_floor": numerical_floor,
        "normalization_scale": scale,
        "classification": label,
        "estimated_period": period,
        "best_nontrivial_period": int(best["candidate_period"]),
        "best_nontrivial_normalized_p95": float(best["normalized_p95"]),
        "p1_normalized_p95": float(p1["normalized_p95"]),
    }
    return rows, summary


def topk_json(tokenizer, ids: torch.Tensor, values: torch.Tensor) -> str:
    return json.dumps(
        [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "value": float(value),
            }
            for token_id, value in zip(ids.cpu().tolist(), values.cpu().tolist())
        ],
        ensure_ascii=False,
    )


def final_token_rows(
    final_states: torch.Tensor,
    model,
    tokenizer,
    counts: Counter[int],
    tokens: list[dict],
    checkpoint: str,
    final_dynamic_step: int,
) -> list[dict]:
    input_weight = model.get_input_embeddings().weight.detach().float()
    output_weight = model.get_output_embeddings().weight.detach().float()
    special = [int(value) for value in tokenizer.all_special_ids]
    normalized_state = torch.nn.functional.normalize(final_states, dim=1)
    normalized_input = torch.nn.functional.normalize(input_weight, dim=1)
    cosine = normalized_state @ normalized_input.T
    if special:
        cosine[:, special] = -torch.inf
    cos_values, cos_ids = cosine.topk(5, dim=1)

    squared_distance = (
        final_states.square().sum(dim=1, keepdim=True)
        + input_weight.square().sum(dim=1).unsqueeze(0)
        - 2.0 * final_states @ input_weight.T
    )
    if special:
        squared_distance[:, special] = torch.inf
    euclidean_values, euclidean_ids = squared_distance.topk(5, dim=1, largest=False)
    euclidean_values = euclidean_values.clamp_min(0).sqrt()

    logits = final_states @ output_weight.T
    probabilities = torch.softmax(logits, dim=1)
    lm_values, lm_ids = probabilities.topk(5, dim=1)
    logit_values = logits.gather(1, lm_ids)
    entropy = -(probabilities * torch.log_softmax(logits, dim=1)).sum(dim=1) / math.log(
        logits.shape[1]
    )
    rows: list[dict] = []
    for index, token in enumerate(tokens):
        cosine_id = int(cos_ids[index, 0])
        euclidean_id = int(euclidean_ids[index, 0])
        lm_id = int(lm_ids[index, 0])
        d1 = float(euclidean_values[index, 0])
        d2 = float(euclidean_values[index, 1])
        rows.append(
            {
                "checkpoint": checkpoint,
                **token,
                "final_dynamic_step": final_dynamic_step,
                "final_vector_norm": float(final_states[index].norm()),
                "cosine_token_id": cosine_id,
                "cosine_token": tokenizer.decode([cosine_id]),
                "cosine_wikitext_count": int(counts[cosine_id]),
                "cosine_similarity": float(cos_values[index, 0]),
                "cosine_distance": 1.0 - float(cos_values[index, 0]),
                "cosine_margin": float(cos_values[index, 0] - cos_values[index, 1]),
                "cosine_top5": topk_json(tokenizer, cos_ids[index], cos_values[index]),
                "euclidean_token_id": euclidean_id,
                "euclidean_token": tokenizer.decode([euclidean_id]),
                "euclidean_wikitext_count": int(counts[euclidean_id]),
                "euclidean_distance": d1,
                "euclidean_absolute_margin": d2 - d1,
                "euclidean_relative_margin": (d2 - d1) / max(d1, 1e-12),
                "euclidean_top5": topk_json(
                    tokenizer, euclidean_ids[index], euclidean_values[index]
                ),
                "lm_head_token_id": lm_id,
                "lm_head_token": tokenizer.decode([lm_id]),
                "lm_head_wikitext_count": int(counts[lm_id]),
                "lm_head_probability": float(lm_values[index, 0]),
                "lm_head_probability_margin": float(lm_values[index, 0] - lm_values[index, 1]),
                "lm_head_logit_margin": float(logit_values[index, 0] - logit_values[index, 1]),
                "lm_head_normalized_entropy": float(entropy[index]),
                "lm_head_top5": topk_json(tokenizer, lm_ids[index], lm_values[index]),
                "cosine_equals_euclidean": cosine_id == euclidean_id,
                "cosine_equals_lm_head": cosine_id == lm_id,
                "euclidean_equals_lm_head": euclidean_id == lm_id,
            }
        )
    return rows


def dynamic_windows(start: int, end: int, width: int) -> list[tuple[int, int]]:
    if start < 0 or end <= start:
        raise ValueError(f"invalid projection interval: start={start}, end={end}")
    if width <= 0:
        raise ValueError(f"projection window must be positive, got {width}")
    if (end - start) % width != 0:
        raise ValueError(
            f"projection interval [{start}, {end}] is not divisible by window {width}"
        )
    return [(left, left + width) for left in range(start, end, width)]


def projection_window_metrics(
    checkpoint: str,
    tail: torch.Tensor,
    projected: torch.Tensor,
    recorded_start: int,
    window_start: int,
    window_end: int,
) -> dict:
    left = window_start - recorded_start
    right = window_end - recorded_start + 1
    window_states = tail[left:right].double()
    state_center = window_states.mean(dim=(0, 1), keepdim=True)
    state_radius = torch.linalg.vector_norm(window_states - state_center, dim=2).flatten()

    window_projected = projected[left:right, :, :2].double()
    projected_center = window_projected.mean(dim=(0, 1), keepdim=True)
    projected_radius = torch.linalg.vector_norm(
        window_projected - projected_center, dim=2
    ).flatten()
    return {
        "checkpoint": checkpoint,
        "window_start": window_start,
        "window_end": window_end,
        "window_transitions": window_end - window_start,
        "window_states": window_end - window_start + 1,
        "full_dimensional_radius_rms": float(torch.sqrt(state_radius.square().mean())),
        "full_dimensional_radius_p95": float(torch.quantile(state_radius, 0.95)),
        "full_dimensional_radius_max": float(state_radius.max()),
        "projection_2d_radius_rms": float(torch.sqrt(projected_radius.square().mean())),
        "projection_2d_radius_p95": float(torch.quantile(projected_radius, 0.95)),
        "projection_2d_radius_max": float(projected_radius.max()),
    }


def plot_checkpoint_window_grid(
    checkpoints: list[str],
    projected_by_checkpoint: dict[str, torch.Tensor],
    tokens: list[dict],
    window_start: int,
    window_end: int,
    recorded_start: int,
    display_floor: float,
    output: Path,
) -> None:
    left = window_start - recorded_start
    right = window_end - recorded_start + 1
    centered: dict[str, np.ndarray] = {}
    for checkpoint in checkpoints:
        # Use float64 for the long-axis mean. In float32 the centering residual can
        # be as large as the late-stage trajectory itself.
        values = projected_by_checkpoint[checkpoint][left:right, :, :2].double().numpy()
        values -= values.mean(axis=(0, 1), dtype=np.float64, keepdims=True)
        centered[checkpoint] = values

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(18, 13.5),
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.30, wspace=0.12)
    colors = plt.get_cmap("tab10")
    for ax, checkpoint in zip(axes.flat, checkpoints):
        values = centered[checkpoint]
        radii = np.linalg.norm(values, axis=2).reshape(-1)
        x_min, x_max = float(values[:, :, 0].min()), float(values[:, :, 0].max())
        y_min, y_max = float(values[:, :, 1].min()), float(values[:, :, 1].max())
        # Projection coordinates have the same units, so use one symmetric limit
        # per panel. This zooms each checkpoint independently without distorting
        # the trajectory geometry.
        panel_limit = max(
            abs(x_min),
            abs(x_max),
            abs(y_min),
            abs(y_max),
            display_floor,
        )
        panel_limit *= 1.08
        for token_index, token in enumerate(tokens):
            xy = values[:, token_index]
            color = colors(token_index)
            ax.plot(xy[:, 0], xy[:, 1], lw=1.0, alpha=0.9, color=color)
            ax.scatter(*xy[0], marker="o", s=24, color=color, edgecolor="black", linewidth=0.4)
            ax.scatter(*xy[-1], marker="X", s=34, color=color, edgecolor="black", linewidth=0.4)
        ax.set_title(f"{checkpoint} · projected r95={np.quantile(radii, 0.95):.3e}", fontsize=10)
        ax.text(
            0.02,
            0.98,
            (
                f"x∈[{x_min:.2e}, {x_max:.2e}]\n"
                f"y∈[{y_min:.2e}, {y_max:.2e}]"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "0.75"},
        )
        ax.set_xlim(-panel_limit, panel_limit)
        ax.set_ylim(-panel_limit, panel_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.ticklabel_format(style="sci", scilimits=(-3, 3), useOffset=False)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Δ projection 0")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ projection 1")
    handles = [
        plt.Line2D([0], [0], color=colors(index), label=repr(token["token"]))
        for index, token in enumerate(tokens)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(tokens),
        fontsize=9,
    )
    fig.suptitle(
        (
            f"Dynamic steps {window_start}–{window_end}: shared projection, adaptive axes\n"
            "Independent equal-unit zoom per checkpoint; centered in float64; ○ start, × end"
        ),
        fontsize=15,
        y=0.975,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_3d(checkpoint: str, projected: torch.Tensor, tokens: list[dict], output: Path) -> None:
    values = projected.double().numpy()
    values = values - values.mean(axis=(0, 1), dtype=np.float64)
    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("tab10")
    for index, token in enumerate(tokens):
        xyz = values[:, index]
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=0.9, color=colors(index), label=repr(token["token"]))
        ax.scatter(*xyz[0], marker="o", s=35, color=colors(index))
        ax.scatter(*xyz[-1], marker="X", s=48, color=colors(index))
    ax.set_xlabel("Δ projection 0")
    ax.set_ylabel("Δ projection 1")
    ax.set_zlabel("Δ projection 2")
    ax.set_title(f"{checkpoint}: shared fixed 3D projection")
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_model(checkpoint: str, cache: Path, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        revision=checkpoint,
        cache_dir=str(cache),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def dynamics_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    report_processed = args.report_root / "processed"
    figures = args.report_root / "figures"
    states_dir = args.data_root / "states"
    for directory in (report_processed, figures, states_dir):
        directory.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        revision="step100000",
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    counts = wikitext_counts(args.wikitext_train_arrow, tokenizer)
    basis = projection_basis()
    all_recurrence: list[dict] = []
    all_summary: list[dict] = []
    all_neighbors: list[dict] = []
    all_window_metrics: list[dict] = []
    projected_by_checkpoint: dict[str, torch.Tensor] = {}
    if not 0 <= args.record_start <= args.burn_in < args.steps:
        raise ValueError(
            "expected 0 <= record_start <= burn_in < steps, got "
            f"{args.record_start}, {args.burn_in}, {args.steps}"
        )
    windows = []
    if args.record_start < args.burn_in:
        windows.append((args.record_start, args.burn_in))
    windows.extend(dynamic_windows(args.burn_in, args.steps, args.projection_window))

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        started = time.perf_counter()
        model = load_model(checkpoint, args.cache_dir, args.device)
        ids = torch.tensor([row["token_id"] for row in tokens], device=args.device)
        state = model.get_input_embeddings()(ids).detach().float()
        attention_mask = torch.ones((len(tokens), 1), device=args.device, dtype=torch.long)
        position_ids = torch.zeros((1, 1), device=args.device, dtype=torch.long)
        recorded: list[torch.Tensor] = []
        if args.record_start == 0:
            recorded.append(state.detach().cpu())
        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                state = model.gpt_neox(
                    inputs_embeds=state.unsqueeze(1),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, -1, :].float()
                if step >= args.record_start:
                    recorded.append(state.detach().cpu())
        recorded_tensor = torch.stack(recorded)
        tail_offset = args.burn_in - args.record_start
        tail_tensor = recorded_tensor[tail_offset:]
        torch.save(
            {
                "checkpoint": checkpoint,
                "steps": args.steps,
                "burn_in": args.burn_in,
                "record_start": args.record_start,
                "token_ids": [row["token_id"] for row in tokens],
                "recorded_states": recorded_tensor,
                "tail_states": tail_tensor,
            },
            states_dir / f"{checkpoint}_tail_states.pt",
        )
        checkpoint_summaries: list[dict] = []
        for token_index, token in enumerate(tokens):
            recurrence, summary = recurrence_scan(
                tail_tensor[:, token_index].to(args.device),
                token,
                checkpoint,
                seed=args.seed + checkpoint_index * 100 + token_index,
                max_period=args.max_period,
            )
            all_recurrence.extend(recurrence)
            all_summary.append(summary)
            checkpoint_summaries.append(summary)
        all_neighbors.extend(
            final_token_rows(
                state,
                model,
                tokenizer,
                counts,
                tokens,
                checkpoint,
                final_dynamic_step=args.steps,
            )
        )
        projected = torch.einsum("tbh,ph->tbp", recorded_tensor, basis)
        projected_by_checkpoint[checkpoint] = projected
        for window_start, window_end in windows:
            all_window_metrics.append(
                projection_window_metrics(
                    checkpoint,
                    recorded_tensor,
                    projected,
                    args.record_start,
                    window_start,
                    window_end,
                )
            )
        if checkpoint in {"step9000", "step16000"}:
            plot_3d(
                checkpoint,
                projected,
                tokens,
                figures / f"{checkpoint}_projection_3d.png",
            )
        print(
            json.dumps(
                {
                    "stage": "dynamics",
                    "checkpoint": checkpoint,
                    "completed": checkpoint_index + 1,
                    "total": len(checkpoints),
                    "seconds": time.perf_counter() - started,
                    "classifications": [row["classification"] for row in checkpoint_summaries],
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    for window_start, window_end in windows:
        plot_checkpoint_window_grid(
            checkpoints,
            projected_by_checkpoint,
            tokens,
            window_start,
            window_end,
            args.record_start,
            args.projection_display_floor,
            figures
            / f"checkpoint_projection_window_step{window_start:04d}_{window_end:04d}.png",
        )

    write_csv(report_processed / "period_recurrence_rows.csv", all_recurrence)
    write_csv(report_processed / "trajectory_summary.csv", all_summary)
    write_csv(report_processed / "final_token_neighbors.csv", all_neighbors)
    write_csv(report_processed / "projection_window_summary.csv", all_window_metrics)


def exact_jacobian(model, state: torch.Tensor, chunk_size: int) -> torch.Tensor:
    position_ids = torch.zeros((1, 1), device=state.device, dtype=torch.long)
    attention_mask = torch.ones((1, 1), device=state.device, dtype=torch.long)

    def mapping(value: torch.Tensor) -> torch.Tensor:
        return model.gpt_neox(
            inputs_embeds=value.reshape(1, 1, -1),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0, -1].float()

    value = state.detach().float().requires_grad_(True)
    return torch.func.jacrev(mapping, chunk_size=chunk_size)(value).detach().float()


def jacobian_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    processed = args.report_root / "processed"
    parts = processed / "jacobian_parts"
    parts.mkdir(parents=True, exist_ok=True)
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        part_path = parts / f"{checkpoint}.csv"
        if part_path.exists() and not args.overwrite:
            print(json.dumps({"stage": "jacobian", "checkpoint": checkpoint, "status": "skip"}), flush=True)
            continue
        payload = torch.load(
            args.data_root / "states" / f"{checkpoint}_tail_states.pt",
            map_location="cpu",
            weights_only=True,
        )
        tail = payload["tail_states"].float()
        model = load_model(checkpoint, args.cache_dir, args.device)
        rows: list[dict] = []
        for token_index, token in enumerate(tokens):
            for offset in range(args.jacobian_steps, 0, -1):
                state = tail[-offset - 1, token_index].to(args.device)
                started = time.perf_counter()
                jacobian = exact_jacobian(model, state, args.jacobian_chunk_size)
                eigenvalues = torch.linalg.eigvals(jacobian)
                order = torch.argsort(eigenvalues.abs(), descending=True)
                first = eigenvalues[order[0]]
                second = eigenvalues[order[1]]
                singular = torch.linalg.svdvals(jacobian)
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        **token,
                        "dynamic_step_input": args.steps - offset,
                        "dynamic_step_output": args.steps - offset + 1,
                        "lambda1_abs": float(first.abs()),
                        "lambda1_real": float(first.real),
                        "lambda1_imag": float(first.imag),
                        "lambda2_abs": float(second.abs()),
                        "lambda2_real": float(second.real),
                        "lambda2_imag": float(second.imag),
                        "sigma1": float(singular[0]),
                        "sigma2": float(singular[1]),
                        "spectral_gap_abs": float(first.abs() - second.abs()),
                        "runtime_seconds": time.perf_counter() - started,
                    }
                )
                print(
                    json.dumps(
                        {
                            "stage": "jacobian",
                            "checkpoint": checkpoint,
                            "token": token["token"],
                            "input_step": args.steps - offset,
                            "lambda1": rows[-1]["lambda1_abs"],
                            "lambda2": rows[-1]["lambda2_abs"],
                            "seconds": rows[-1]["runtime_seconds"],
                        }
                    ),
                    flush=True,
                )
        write_csv(part_path, rows)
        del model
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "stage": "jacobian",
                    "checkpoint": checkpoint,
                    "completed": checkpoint_index + 1,
                    "total": len(checkpoints),
                }
            ),
            flush=True,
        )
    combined: list[dict] = []
    for checkpoint in checkpoints:
        combined.extend(read_csv(parts / f"{checkpoint}.csv"))
    write_csv(processed / "final10_jacobian_spectrum.csv", combined)


def report_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    processed = args.report_root / "processed"
    figures = args.report_root / "figures"
    summary = read_csv(processed / "trajectory_summary.csv")
    recurrence = read_csv(processed / "period_recurrence_rows.csv")
    jacobian = read_csv(processed / "final10_jacobian_spectrum.csv")

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    matrix = np.full((len(tokens), len(checkpoints)), np.nan)
    for token_index, token in enumerate(tokens):
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            row = next(
                item
                for item in summary
                if item["checkpoint"] == checkpoint
                and int(item["token_id"]) == int(token["token_id"])
            )
            matrix[token_index, checkpoint_index] = float(row["best_nontrivial_normalized_p95"])
    image = ax.imshow(np.log10(np.maximum(matrix, 1e-12)), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(checkpoints)), checkpoints, rotation=45, ha="right")
    ax.set_yticks(range(len(tokens)), [repr(row["token"]) for row in tokens])
    ax.set_title("Best nontrivial full-512D recurrence (log10 normalized P95)")
    fig.colorbar(image, ax=ax)
    fig.savefig(figures / "period_recurrence_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for token_index, token in enumerate(tokens):
        group = [
            row for row in jacobian if int(row["token_id"]) == int(token["token_id"])
        ]
        for ax, metric, label in [
            (axes[0], "lambda1_abs", "|lambda1|"),
            (axes[1], "lambda2_abs", "|lambda2|"),
        ]:
            medians = []
            for checkpoint in checkpoints:
                values = [
                    float(row[metric]) for row in group if row["checkpoint"] == checkpoint
                ]
                medians.append(float(np.median(values)))
            ax.plot(
                [int(value.removeprefix("step")) for value in checkpoints],
                medians,
                marker="o",
                label=repr(token["token"]),
            )
            ax.set_ylabel(f"Median final-10 {label}")
            ax.set_xlabel("Training checkpoint step")
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.savefig(figures / "final10_jacobian_top2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "complete",
        "checkpoints": checkpoints,
        "tokens": tokens,
        "trajectory_summary_rows": len(summary),
        "recurrence_rows": len(recurrence),
        "jacobian_rows": len(jacobian),
    }
    (args.report_root / "run_complete.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dynamics", "jacobian", "report", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wikitext-train-arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--token-manifest", type=Path, default=EXPERIMENT13_TOKENS)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:7"))
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--burn-in", type=int, default=512)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--projection-window", type=int, default=128)
    parser.add_argument("--projection-display-floor", type=float, default=1e-6)
    parser.add_argument("--max-period", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1414)
    parser.add_argument("--jacobian-steps", type=int, default=10)
    parser.add_argument("--jacobian-chunk-size", type=int, default=16)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoints = CHECKPOINTS[: args.max_checkpoints]
    tokens = load_tokens(args.token_manifest, args.max_tokens)
    if args.stage in {"dynamics", "all"}:
        dynamics_stage(args, checkpoints, tokens)
    if args.stage in {"jacobian", "all"}:
        jacobian_stage(args, checkpoints, tokens)
    if args.stage in {"report", "all"}:
        report_stage(args, checkpoints, tokens)


if __name__ == "__main__":
    main()
