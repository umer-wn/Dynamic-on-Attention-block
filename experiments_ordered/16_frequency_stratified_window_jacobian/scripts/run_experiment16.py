#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts._bootstrap import require_packages

require_packages(["datasets", "matplotlib", "numpy", "torch", "transformers"])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
EXP15_SCRIPT = REPO / (
    "experiments_ordered/15_window_jacobian_token_projection/"
    "scripts/run_experiment15.py"
)
SPEC = importlib.util.spec_from_file_location("experiment15_base", EXP15_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {EXP15_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

CHECKPOINTS = [
    "step0",
    "step2000",
    "step3000",
    "step4000",
    "step5000",
    "step7000",
    "step8000",
    "step9000",
    "step10000",
    "step13000",
    "step21000",
    "step25000",
    "step29000",
    "step33000",
    "step37000",
    "step41000",
    "step53000",
    "step57000",
    "step61000",
]
PERIOD_CHECKPOINTS = [
    "step9000",
    "step29000",
    "step41000",
    "step53000",
    "step57000",
]
base.CHECKPOINTS = CHECKPOINTS
DEFAULT_CACHE = Path("/home/luohaoming/model_feature_cache/hf_cache")
DEFAULT_DATA_ROOT = Path(
    "/home/luohaoming/model_feature_experiments/"
    "experiment16_frequency_stratified_window_jacobian"
)
TOKEN_MANIFEST = ROOT / "manifests/frequency_stratified_tokens.csv"
LOSS_ROOT = Path(
    "/home/luohaoming/model_feature_experiments/"
    "pythia_validation_corpus_loss_rescan/raw/the_pile_test"
)


def load_tokens(path: Path) -> list[dict]:
    rows = base.read_csv(path)
    return [
        {
            "selection_index": int(row["selection_index"]),
            "token_id": int(row["token_id"]),
            "token": row["token"],
            "wikitext_train_count": int(row["wikitext_train_count"]),
            "frequency_bin": int(row["frequency_bin"]),
        }
        for row in rows
    ]


def projection_basis(hidden: int = 512, seed: int = 1616) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn((hidden, 4), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.float()


def projection_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    processed = args.report_root / "processed"
    figures = args.report_root / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    basis = projection_basis(seed=args.projection_seed)
    torch.save({"seed": args.projection_seed, "basis": basis}, processed / "projection_basis.pt")
    projected: dict[str, torch.Tensor] = {}
    rows: list[dict] = []
    triples: list[str] = []
    for checkpoint in checkpoints:
        states = base.load_states(args, checkpoint, tokens)
        values = torch.einsum("tbh,ph->tbp", states, basis)
        projected[checkpoint] = values
        for step in range(values.shape[0]):
            for token_index, token in enumerate(tokens):
                coords = [float(value) for value in values[step, token_index]]
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "dynamic_step": step,
                        **token,
                        **{f"projection_{i + 1}": coords[i] for i in range(4)},
                    }
                )
                triples.append(
                    json.dumps(
                        [
                            [checkpoint, step],
                            coords,
                            {
                                "token_id": token["token_id"],
                                "token": token["token"],
                                "frequency_bin": token["frequency_bin"],
                            },
                        ],
                        ensure_ascii=False,
                    )
                )
    base.write_csv(processed / "projection_trajectory.csv", rows)
    (processed / "projection_triples.jsonl").write_text(
        "\n".join(triples) + "\n", encoding="utf-8"
    )
    for end in base.window_endpoints(args.steps, args.window):
        start = end - args.window
        for pair in ((0, 1), (2, 3)):
            plot_projection_grid(
                projected,
                checkpoints,
                tokens,
                start,
                end,
                pair,
                figures
                / f"projection_p{pair[0] + 1}_p{pair[1] + 1}_step{start:04d}_{end:04d}.png",
            )


def plot_projection_grid(
    projected: dict[str, torch.Tensor],
    checkpoints: list[str],
    tokens: list[dict],
    start: int,
    end: int,
    pair: tuple[int, int],
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 5, figsize=(21, 12))
    colors = plt.get_cmap("tab10")
    for ax, checkpoint in zip(axes.flat, checkpoints):
        xy = projected[checkpoint][start : end + 1, :, list(pair)].double().numpy()
        xy -= xy.mean(axis=(0, 1), keepdims=True)
        x0, x1 = float(xy[..., 0].min()), float(xy[..., 0].max())
        y0, y1 = float(xy[..., 1].min()), float(xy[..., 1].max())
        span = max(abs(x0), abs(x1), abs(y0), abs(y1), 1e-12) * 1.08
        for index, token in enumerate(tokens):
            curve = xy[:, index]
            ax.plot(curve[:, 0], curve[:, 1], lw=0.9, color=colors(index))
            ax.scatter(*curve[0], s=18, marker="o", color=colors(index))
            ax.scatter(*curve[-1], s=28, marker="X", color=colors(index))
        ax.set(xlim=(-span, span), ylim=(-span, span), title=checkpoint)
        ax.set_aspect("equal", adjustable="box")
        ax.ticklabel_format(style="sci", scilimits=(-3, 3), useOffset=False)
        ax.text(
            0.02,
            0.98,
            f"x∈[{x0:.2e},{x1:.2e}]\ny∈[{y0:.2e},{y1:.2e}]",
            transform=ax.transAxes,
            va="top",
            fontsize=6.5,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.8"},
        )
        ax.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], color=colors(i), label=f"{token['token']!r} (bin {token['frequency_bin']})")
        for i, token in enumerate(tokens)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle(
        f"Experiment 16 · steps {start}–{end} · centered fixed P{pair[0] + 1}/P{pair[1] + 1}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def lyapunov_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    processed = args.report_root / "processed"
    parts = processed / "lyapunov_parts"
    parts.mkdir(parents=True, exist_ok=True)
    endpoints = base.window_endpoints(args.steps, args.window)
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        part = parts / f"{checkpoint}.csv"
        if part.exists() and not args.overwrite:
            existing = base.read_csv(part)
            if len(existing) == len(tokens) * len(endpoints):
                print(json.dumps({"stage": "lyapunov", "checkpoint": checkpoint, "status": "skip"}), flush=True)
                continue
        states = base.load_states(args, checkpoint, tokens).to(args.device)
        model = base.load_model(checkpoint, args.cache_dir, args.device)
        # PyTorch SDPA does not currently implement forward-mode AD. Eager
        # attention is mathematically equivalent here (sequence length is one)
        # and permits the exact JVP required by tangent propagation.
        model.set_attn_implementation("eager")
        batch = len(tokens)
        attention_mask = torch.ones((batch, 1), device=args.device, dtype=torch.long)
        position_ids = torch.zeros((1, 1), device=args.device, dtype=torch.long)

        def mapping(value: torch.Tensor) -> torch.Tensor:
            return model.gpt_neox(
                inputs_embeds=value.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1, :].float()

        generator = torch.Generator(device="cpu").manual_seed(args.lyapunov_seed)
        tangent = torch.randn((batch, states.shape[-1]), generator=generator).to(args.device)
        tangent = tangent / tangent.norm(dim=1, keepdim=True)
        logs: list[torch.Tensor] = []
        started = time.perf_counter()
        for step in range(args.steps):
            _, next_tangent = torch.func.jvp(mapping, (states[step],), (tangent,))
            growth = next_tangent.norm(dim=1).clamp_min(args.lyapunov_epsilon)
            logs.append(growth.log().detach().cpu())
            tangent = (next_tangent / growth.unsqueeze(1)).detach()
        log_growth = torch.stack(logs)
        rows: list[dict] = []
        for end in endpoints:
            start = end - args.window
            for token_index, token in enumerate(tokens):
                value = float(log_growth[start:end, token_index].mean())
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "window_start": start,
                        "window_end": end,
                        "dynamic_step": end,
                        **token,
                        "lyapunov_exponent_per_step": value,
                        "geometric_tangent_factor_per_step": math.exp(value),
                        "overall_lyapunov_0_1024": float(log_growth[:, token_index].mean()),
                    }
                )
        base.write_csv(part, rows)
        print(
            json.dumps(
                {
                    "stage": "lyapunov",
                    "checkpoint": checkpoint,
                    "completed": checkpoint_index + 1,
                    "total": len(checkpoints),
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del model, states
        torch.cuda.empty_cache()
    all_rows = [row for checkpoint in checkpoints for row in base.read_csv(parts / f"{checkpoint}.csv")]
    base.write_csv(processed / "lyapunov_by_token_window.csv", all_rows)
    overall_rows: list[dict] = []
    summary_rows: list[dict] = []
    for checkpoint in checkpoints:
        group = [row for row in all_rows if row["checkpoint"] == checkpoint]
        token_values = []
        for token in tokens:
            row = next(row for row in group if int(row["token_id"]) == token["token_id"])
            value = float(row["overall_lyapunov_0_1024"])
            token_values.append(value)
            overall_rows.append(
                {
                    "checkpoint": checkpoint,
                    **token,
                    "lyapunov_exponent_0_1024": value,
                    "geometric_tangent_factor_per_step": math.exp(value),
                }
            )
        summary_rows.append(
            {
                "checkpoint": checkpoint,
                "lyapunov_median_4tokens": float(np.median(token_values)),
                "lyapunov_min_4tokens": min(token_values),
                "lyapunov_max_4tokens": max(token_values),
                "geometric_factor_from_median": math.exp(float(np.median(token_values))),
            }
        )
    base.write_csv(processed / "lyapunov_by_token_overall.csv", overall_rows)
    base.write_csv(processed / "lyapunov_checkpoint_summary.csv", summary_rows)
    plot_lyapunov(summary_rows, args.report_root / "figures/lyapunov_by_checkpoint.png")


def plot_lyapunov(rows: list[dict], output: Path) -> None:
    x = np.arange(len(rows))
    med = np.array([float(row["lyapunov_median_4tokens"]) for row in rows])
    low = np.array([float(row["lyapunov_min_4tokens"]) for row in rows])
    high = np.array([float(row["lyapunov_max_4tokens"]) for row in rows])
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, med, marker="o", label="median over 4 tokens")
    ax.fill_between(x, low, high, alpha=0.2, label="token min–max")
    ax.axhline(0.0, color="black", ls="--", lw=0.9)
    ax.set_xticks(x, [row["checkpoint"].replace("step", "") for row in rows], rotation=35)
    ax.set(xlabel="training checkpoint", ylabel="largest FTLE per dynamic step")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def sampled_orbit_scale(states: torch.Tensor, seed: int, pairs: int = 8192) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    length = states.shape[0]
    left = torch.randint(0, length, (pairs,), generator=generator)
    right = torch.randint(0, length, (pairs,), generator=generator)
    sampled = torch.linalg.vector_norm(states[left] - states[right], dim=1)
    first = torch.argmax(torch.linalg.vector_norm(states - states[0], dim=1))
    farthest = torch.linalg.vector_norm(states - states[first], dim=1)
    state_norm = torch.linalg.vector_norm(states, dim=1)
    numerical_floor = 8.0 * torch.finfo(torch.float32).eps * max(
        float(state_norm.median()), 1.0
    )
    d95 = float(torch.quantile(sampled, 0.95))
    return {
        "orbit_diameter_approx": max(float(sampled.max()), float(farthest.max())),
        "orbit_pairwise_d95": d95,
        "orbit_pairwise_rms": float(torch.sqrt(sampled.square().mean())),
        "median_state_norm": float(state_norm.median()),
        "numerical_floor": numerical_floor,
        "normalization_scale": max(d95, numerical_floor, 1e-12),
    }


def substep_minimum(values: list[float], index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index + 1)
    left, center, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * center + right
    if denominator <= 0:
        return float(index + 1)
    offset = 0.5 * (left - right) / denominator
    return float(index + 1 + np.clip(offset, -0.5, 0.5))


def period_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    selected = [checkpoint for checkpoint in PERIOD_CHECKPOINTS if checkpoint in checkpoints]
    if not selected:
        raise ValueError("period stage needs at least one configured period checkpoint")
    processed = args.report_root / "processed"
    figures = args.report_root / "figures"
    recurrence_rows: list[dict] = []
    token_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    tail_start = args.steps - args.period_tail_steps
    if tail_start < 0 or args.period_max * 4 > args.period_tail_steps:
        raise ValueError(
            "period tail must cover at least four repeats of the maximum candidate: "
            f"tail={args.period_tail_steps}, max={args.period_max}"
        )
    for checkpoint_index, checkpoint in enumerate(selected):
        states = base.load_states(args, checkpoint, tokens)[tail_start : args.steps + 1]
        curves: list[list[float]] = []
        checkpoint_token_rows: list[dict] = []
        for token_index, token in enumerate(tokens):
            trajectory = states[:, token_index].float()
            scale = sampled_orbit_scale(
                trajectory, args.period_seed + checkpoint_index * 100 + token_index
            )
            curve: list[float] = []
            raw_rows: list[dict] = []
            for period in range(1, args.period_max + 1):
                absolute = torch.linalg.vector_norm(
                    trajectory[period:] - trajectory[:-period], dim=1
                )
                normalized = absolute / scale["normalization_scale"]
                row = {
                    "checkpoint": checkpoint,
                    "tail_start": tail_start,
                    "tail_end": args.steps,
                    **token,
                    "candidate_period": period,
                    "absolute_median": float(absolute.median()),
                    "absolute_p95": float(torch.quantile(absolute, 0.95)),
                    "absolute_max": float(absolute.max()),
                    "normalized_median": float(normalized.median()),
                    "normalized_p95": float(torch.quantile(normalized, 0.95)),
                    "normalized_max": float(normalized.max()),
                }
                raw_rows.append(row)
                recurrence_rows.append(row)
                curve.append(row["normalized_p95"])
            best_index = int(np.argmin(curve[1:])) + 1
            best = raw_rows[best_index]
            classification = (
                "strict_cycle_candidate"
                if best["normalized_p95"] <= 1e-4
                else "approximate_cycle_candidate"
                if best["normalized_p95"] <= 1e-2
                else "dominant_recurrence_only"
            )
            token_row = {
                "checkpoint": checkpoint,
                "tail_start": tail_start,
                "tail_end": args.steps,
                **token,
                **scale,
                "estimated_period_integer": int(best["candidate_period"]),
                "estimated_period_quadratic": substep_minimum(curve, best_index),
                "best_absolute_median": best["absolute_median"],
                "best_absolute_p95": best["absolute_p95"],
                "best_absolute_max": best["absolute_max"],
                "best_normalized_median": best["normalized_median"],
                "best_normalized_p95": best["normalized_p95"],
                "best_normalized_max": best["normalized_max"],
                "classification": classification,
            }
            token_rows.append(token_row)
            checkpoint_token_rows.append(token_row)
            curves.append(curve)
        median_curve = np.median(np.asarray(curves), axis=0)
        best_index = int(np.argmin(median_curve[1:])) + 1
        best_period = best_index + 1
        token_periods = [int(row["estimated_period_integer"]) for row in checkpoint_token_rows]
        norm_values = [float(row["best_normalized_p95"]) for row in checkpoint_token_rows]
        absolute_values = [float(row["best_absolute_p95"]) for row in checkpoint_token_rows]
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "tail_start": tail_start,
                "tail_end": args.steps,
                "estimated_period_integer": best_period,
                "estimated_period_quadratic": substep_minimum(
                    median_curve.tolist(), best_index
                ),
                "token_period_min": min(token_periods),
                "token_period_max": max(token_periods),
                "all_tokens_same_integer_period": len(set(token_periods)) == 1,
                "normalized_p95_mean_4tokens": float(np.mean(norm_values)),
                "normalized_p95_min_4tokens": min(norm_values),
                "normalized_p95_max_4tokens": max(norm_values),
                "absolute_p95_mean_4tokens": float(np.mean(absolute_values)),
                "absolute_p95_min_4tokens": min(absolute_values),
                "absolute_p95_max_4tokens": max(absolute_values),
                "classification": (
                    "strict_cycle_candidate"
                    if max(norm_values) <= 1e-4
                    else "approximate_cycle_candidate"
                    if max(norm_values) <= 1e-2
                    else "dominant_recurrence_only"
                ),
            }
        )
    base.write_csv(processed / "period_recurrence_rows.csv", recurrence_rows)
    base.write_csv(processed / "period_by_token.csv", token_rows)
    base.write_csv(processed / "period_checkpoint_summary.csv", checkpoint_rows)

    lyapunov_rows = base.read_csv(processed / "lyapunov_by_token_window.csv")
    final_window_rows = [
        row for row in lyapunov_rows if int(row["dynamic_step"]) == args.steps
    ]
    lyapunov_summary: list[dict] = []
    for checkpoint in checkpoints:
        values = np.asarray(
            [
                float(row["lyapunov_exponent_per_step"])
                for row in final_window_rows
                if row["checkpoint"] == checkpoint
            ],
            dtype=np.float64,
        )
        if values.size != len(tokens):
            raise RuntimeError(f"expected {len(tokens)} final-window values for {checkpoint}")
        lyapunov_summary.append(
            {
                "checkpoint": checkpoint,
                "window_start": args.steps - args.window,
                "window_end": args.steps,
                "token_count": int(values.size),
                "lyapunov_mean": float(values.mean()),
                "lyapunov_population_variance": float(values.var(ddof=0)),
                "lyapunov_sample_variance": float(values.var(ddof=1)),
                "lyapunov_lower_bound_min": float(values.min()),
                "lyapunov_upper_bound_max": float(values.max()),
            }
        )
    base.write_csv(processed / "lyapunov_last256_summary.csv", lyapunov_summary)
    plot_period_curves(
        recurrence_rows,
        selected,
        figures / "period_recurrence_last512.png",
    )
    plot_last_window_lyapunov(
        lyapunov_summary,
        figures / "lyapunov_last256_checkpoint_summary.png",
    )


def plot_period_curves(rows: list[dict], checkpoints: list[str], output: Path) -> None:
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(20, 4.3), sharey=True)
    for ax, checkpoint in zip(axes, checkpoints):
        group = [row for row in rows if row["checkpoint"] == checkpoint]
        periods = sorted({int(row["candidate_period"]) for row in group})
        values = [
            np.median(
                [
                    float(row["normalized_p95"])
                    for row in group
                    if int(row["candidate_period"]) == period
                ]
            )
            for period in periods
        ]
        best = int(np.argmin(values[1:])) + 1
        ax.semilogy(periods, values, lw=1.1)
        ax.scatter(periods[best], values[best], color="red", zorder=3)
        ax.set_title(f"{checkpoint} · p≈{periods[best]}")
        ax.set_xlabel("candidate lag p")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("median over tokens: normalized P95 recurrence")
    fig.suptitle("Experiment 16 · full-512D recurrence over dynamic steps 512–1024")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_last_window_lyapunov(rows: list[dict], output: Path) -> None:
    x = np.arange(len(rows))
    means = np.asarray([float(row["lyapunov_mean"]) for row in rows])
    lower = np.asarray([float(row["lyapunov_lower_bound_min"]) for row in rows])
    upper = np.asarray([float(row["lyapunov_upper_bound_max"]) for row in rows])
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.errorbar(
        x,
        means,
        yerr=np.vstack((means - lower, upper - means)),
        fmt="o-",
        capsize=3,
        label="mean with token min–max",
    )
    ax.axhline(0.0, color="black", ls="--", lw=0.9)
    ax.set_xticks(
        x, [row["checkpoint"].replace("step", "") for row in rows], rotation=35
    )
    ax.set(
        xlabel="training checkpoint",
        ylabel="last-window Lyapunov exponent / dynamic step",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def loss_stage(args, checkpoints: list[str]) -> None:
    rows = []
    missing = []
    for checkpoint in checkpoints:
        path = args.loss_root / checkpoint / "loss_complete.json"
        if not path.exists():
            missing.append(checkpoint)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "checkpoint": checkpoint,
                "training_step": int(checkpoint.removeprefix("step")),
                "loss": float(payload.get("token_weighted_loss", payload.get("loss"))),
                "corpus": "the_pile_test",
                "sample_count": int(payload.get("sample_count", payload.get("num_samples", 512))),
                "sequence_length": int(payload.get("sequence_length", 64)),
                "source": str(path),
            }
        )
    if missing:
        raise FileNotFoundError(f"missing fixed-corpus loss for: {missing}")
    base.write_csv(args.report_root / "processed/checkpoint_loss.csv", rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([row["training_step"] for row in rows], [row["loss"] for row in rows], marker="o")
    ax.set(xlabel="training checkpoint step", ylabel="token-weighted causal CE loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.report_root / "figures/checkpoint_loss.png", dpi=180)
    plt.close(fig)


def analysis_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    processed = args.report_root / "processed"
    figures = args.report_root / "figures"
    parts = processed / "endpoint_parts"
    parts.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        revision="step100000",
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    counts = base.wikitext_counts(args.wikitext_train_arrow, tokenizer)
    basis = projection_basis(seed=args.projection_seed).double()
    loss_map = {row["checkpoint"]: float(row["loss"]) for row in base.read_csv(processed / "checkpoint_loss.csv")}
    lyap_rows = base.read_csv(processed / "lyapunov_by_token_window.csv")
    lyap_map = {
        (row["checkpoint"], int(row["dynamic_step"]), int(row["token_id"])): float(row["lyapunov_exponent_per_step"])
        for row in lyap_rows
    }
    endpoints = base.window_endpoints(args.steps, args.window)
    expected = len(endpoints) * len(tokens)
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        part = parts / f"{checkpoint}.csv"
        rows = [] if args.overwrite or not part.exists() else base.read_csv(part)
        row_index = {
            (int(row["dynamic_step"]), int(row["token_id"])): index
            for index, row in enumerate(rows)
        }
        done = {
            key
            for key, index in row_index.items()
            if rows[index].get("jacobian_frobenius_norm", "") != ""
        }
        if len(done) == expected:
            print(json.dumps({"stage": "analysis", "checkpoint": checkpoint, "status": "skip"}), flush=True)
            continue
        states = base.load_states(args, checkpoint, tokens)
        model = base.load_model(checkpoint, args.cache_dir, args.device)
        model.set_attn_implementation("eager")
        for endpoint in endpoints:
            for token_index, token in enumerate(tokens):
                key = (endpoint, token["token_id"])
                if key in done:
                    continue
                started = time.perf_counter()
                state = states[endpoint, token_index].to(args.device)
                jacobian = base.exact_jacobian(model, state, args.jacobian_chunk_size)
                frobenius_norm = torch.linalg.matrix_norm(jacobian, ord="fro")
                eigenvalues = torch.linalg.eigvals(jacobian)
                abs_values = eigenvalues.abs()
                radius, radius_index = abs_values.max(dim=0)
                abscissa, abscissa_index = eigenvalues.real.max(dim=0)
                singular = torch.linalg.svdvals(jacobian).topk(2).values
                radius_eigenvalue = eigenvalues[radius_index]
                abscissa_eigenvalue = eigenvalues[abscissa_index]
                nearest = base.endpoint_token_metrics(state, model, tokenizer, counts)
                coords = basis @ state.detach().cpu().double()
                result = {
                        "checkpoint": checkpoint,
                        "window_start": endpoint - args.window,
                        "window_end": endpoint,
                        "dynamic_step": endpoint,
                        **token,
                        **{f"projection_{i + 1}": float(coords[i]) for i in range(4)},
                        "checkpoint_loss": loss_map[checkpoint],
                        "lyapunov_exponent_window": lyap_map[(checkpoint, endpoint, token["token_id"])],
                        "spectral_radius": float(radius),
                        "spectral_radius_eigenvalue_real": float(radius_eigenvalue.real),
                        "spectral_radius_eigenvalue_imag": float(radius_eigenvalue.imag),
                        "spectral_abscissa": float(abscissa),
                        "max_real_eigenvalue_imag": float(abscissa_eigenvalue.imag),
                        "max_real_eigenvalue_abs": float(abscissa_eigenvalue.abs()),
                        "operator_norm_2": float(singular[0]),
                        "operator_norm_sigma2": float(singular[1]),
                        "jacobian_frobenius_norm": float(frobenius_norm),
                        "operator_norm_over_spectral_radius": float(
                            singular[0] / radius.clamp_min(1e-12)
                        ),
                        **nearest,
                        "runtime_seconds": time.perf_counter() - started,
                    }
                if key in row_index:
                    rows[row_index[key]].update(result)
                    latest = rows[row_index[key]]
                else:
                    rows.append(result)
                    row_index[key] = len(rows) - 1
                    latest = result
                done.add(key)
                base.write_csv(part, rows)
                print(
                    json.dumps(
                        {
                            "stage": "analysis",
                            "checkpoint": checkpoint,
                            "dynamic_step": endpoint,
                            "token": token["token"],
                            "rho": latest["spectral_radius"],
                            "alpha": latest["spectral_abscissa"],
                            "norm": latest["operator_norm_2"],
                            "frobenius": latest["jacobian_frobenius_norm"],
                            "seconds": latest["runtime_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"stage": "analysis", "completed": checkpoint_index + 1, "total": len(checkpoints)}), flush=True)
    combined = [row for checkpoint in checkpoints for row in base.read_csv(parts / f"{checkpoint}.csv")]
    base.write_csv(processed / "window_endpoint_metrics.csv", combined)
    plot_metric_grid(combined, checkpoints, tokens, figures)


def plot_metric_grid(rows: list[dict], checkpoints: list[str], tokens: list[dict], figures: Path) -> None:
    definitions = [
        ("spectral_radius", "spectral radius ρ(J)"),
        ("spectral_abscissa", "max Re(eigenvalue) α(J)"),
        ("operator_norm_2", "operator norm ||J||₂"),
        ("jacobian_frobenius_norm", "Frobenius norm ||J||F"),
        ("lyapunov_exponent_window", "window FTLE / step"),
    ]
    colors = plt.get_cmap("tab10")
    for field, label in definitions:
        fig, axes = plt.subplots(3, 5, figsize=(20, 11), sharex=True)
        for ax, checkpoint in zip(axes.flat, checkpoints):
            for index, token in enumerate(tokens):
                subset = sorted(
                    [
                        row
                        for row in rows
                        if row["checkpoint"] == checkpoint
                        and int(row["token_id"]) == token["token_id"]
                    ],
                    key=lambda row: int(row["dynamic_step"]),
                )
                ax.plot(
                    [int(row["dynamic_step"]) for row in subset],
                    [float(row[field]) for row in subset],
                    marker="o",
                    color=colors(index),
                )
            ax.set_title(checkpoint)
            ax.grid(alpha=0.25)
        handles = [
            plt.Line2D([0], [0], color=colors(i), marker="o", label=token["token"])
            for i, token in enumerate(tokens)
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4)
        fig.suptitle(label, fontsize=15)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(figures / f"{field}_by_window.png", dpi=180)
        plt.close(fig)
    plot_jacobian_by_checkpoint(rows, checkpoints, figures)


def plot_jacobian_by_checkpoint(
    rows: list[dict],
    checkpoints: list[str],
    figures: Path,
) -> None:
    definitions = [
        ("spectral_radius", "spectral radius ρ(J)"),
        ("spectral_abscissa", "max Re(eigenvalue) α(J)"),
        ("operator_norm_2", "operator norm ||J||₂"),
        ("jacobian_frobenius_norm", "Frobenius norm ||J||F"),
    ]
    endpoints = sorted({int(row["dynamic_step"]) for row in rows})
    checkpoint_steps = [int(checkpoint.removeprefix("step")) for checkpoint in checkpoints]
    colors = plt.get_cmap("viridis")(
        np.linspace(0.12, 0.88, len(endpoints))
    )

    def series(field: str, endpoint: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        median, lower, upper = [], [], []
        for checkpoint in checkpoints:
            values = np.asarray(
                [
                    float(row[field])
                    for row in rows
                    if row["checkpoint"] == checkpoint
                    and int(row["dynamic_step"]) == endpoint
                ],
                dtype=np.float64,
            )
            median.append(float(np.median(values)))
            lower.append(float(values.min()))
            upper.append(float(values.max()))
        return np.asarray(median), np.asarray(lower), np.asarray(upper)

    for field, ylabel in definitions:
        fig, ax = plt.subplots(figsize=(12.5, 5.5))
        for color, endpoint in zip(colors, endpoints):
            median, lower, upper = series(field, endpoint)
            ax.plot(
                checkpoint_steps,
                median,
                marker="o",
                ms=4,
                lw=1.25,
                color=color,
                label=f"dynamic step {endpoint}",
            )
            ax.fill_between(
                checkpoint_steps, lower, upper, color=color, alpha=0.12
            )
        if field != "jacobian_frobenius_norm":
            ax.axhline(1.0, color="black", ls="--", lw=0.9, alpha=0.75)
        ax.set_xticks(
            checkpoint_steps,
            [checkpoint.replace("step", "") for checkpoint in checkpoints],
            rotation=35,
        )
        ax.set(
            xlabel="training checkpoint",
            ylabel=ylabel,
            title=f"Experiment 16 · {ylabel} by checkpoint",
        )
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(figures / f"{field}_by_checkpoint.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(13, 17), sharex=True)
    for ax, (field, ylabel) in zip(axes, definitions):
        for color, endpoint in zip(colors, endpoints):
            median, lower, upper = series(field, endpoint)
            ax.plot(
                checkpoint_steps,
                median,
                marker="o",
                ms=3.5,
                lw=1.15,
                color=color,
                label=f"dynamic step {endpoint}",
            )
            ax.fill_between(
                checkpoint_steps, lower, upper, color=color, alpha=0.11
            )
        if field != "jacobian_frobenius_norm":
            ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1].set_xticks(
        checkpoint_steps,
        [checkpoint.replace("step", "") for checkpoint in checkpoints],
        rotation=35,
    )
    axes[-1].set_xlabel("training checkpoint")
    axes[0].legend(ncol=4, fontsize=9)
    fig.suptitle(
        "Experiment 16 · Jacobian metrics by training checkpoint\n"
        "lines: median across 4 tokens; bands: token min–max",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "jacobian_four_metrics_by_checkpoint.png", dpi=180)
    plt.close(fig)
    export_jacobian_loss_visualization_table(
        rows,
        checkpoints,
        figures.parent / "processed/jacobian_three_metrics_with_loss.csv",
    )


def export_jacobian_loss_visualization_table(
    rows: list[dict],
    checkpoints: list[str],
    output: Path,
) -> None:
    endpoints = sorted({int(row["dynamic_step"]) for row in rows})
    output_rows: list[dict] = []
    for checkpoint in checkpoints:
        for endpoint in endpoints:
            group = [
                row
                for row in rows
                if row["checkpoint"] == checkpoint
                and int(row["dynamic_step"]) == endpoint
            ]
            if not group:
                raise RuntimeError(f"missing Jacobian rows for {checkpoint} / {endpoint}")
            output_row = {
                "checkpoint": checkpoint,
                "training_step": int(checkpoint.removeprefix("step")),
                "dynamic_step": endpoint,
                "window_start": int(group[0]["window_start"]),
                "window_end": int(group[0]["window_end"]),
                "checkpoint_loss": float(group[0]["checkpoint_loss"]),
                "token_count": len(group),
            }
            for field in (
                "spectral_radius",
                "spectral_abscissa",
                "operator_norm_2",
                "jacobian_frobenius_norm",
            ):
                values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
                output_row[f"{field}_median"] = float(np.median(values))
                output_row[f"{field}_min"] = float(values.min())
                output_row[f"{field}_max"] = float(values.max())
            output_rows.append(output_row)
    base.write_csv(output, output_rows)
    base.write_csv(
        output.with_name("jacobian_four_metrics_with_loss.csv"),
        output_rows,
    )


def report_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    metrics = base.read_csv(args.report_root / "processed/window_endpoint_metrics.csv")
    losses = base.read_csv(args.report_root / "processed/checkpoint_loss.csv")
    lyapunov = base.read_csv(args.report_root / "processed/lyapunov_checkpoint_summary.csv")
    endpoints = base.window_endpoints(args.steps, args.window)
    lines = [
        "# 实验16报告：词频分层的窗口动力学",
        "",
        "## 1. 协议",
        "",
        f"- checkpoint：{', '.join(checkpoints)}。",
        f"- dynamic step：0–{args.steps}；窗口宽度 {args.window}；端点 {endpoints}。",
        "- 4个token来自分离的WikiText-2词频bin；所有checkpoint使用完全相同的token。",
        "- Projection 1–4来自同一个固定随机正交基；CSV与JSONL均保存"
        "`(checkpoint, dynamicstep, projection(1,2,3,4))`。",
        "- Lyapunov使用JVP传播单个扰动向量，并在每一步重新归一化。"
        "`λ=(1/T)Σ log(||J_t v_t||₂)`，单位是每个dynamic step的自然对数增长率；"
        "`λ>0`表示该有限时间轨道附近扰动平均增长，`λ<0`表示平均衰减。",
        "",
        "### 起始token",
        "",
        "| token | id | WikiText count | frequency bin |",
        "|---|---:|---:|---:|",
    ]
    for token in tokens:
        lines.append(
            f"| {token['token']!r} | {token['token_id']} | "
            f"{token['wikitext_train_count']} | {token['frequency_bin']} |"
        )
    lines += [
        "",
        "## 2. Checkpoint loss",
        "",
        "loss沿用实验11的固定The Pile test协议：512个固定样本、sequence length 64、"
        "token-weighted causal cross entropy。",
        "",
        "| checkpoint | loss |",
        "|---|---:|",
    ]
    for row in losses:
        lines.append(f"| {row['checkpoint']} | {float(row['loss']):.7f} |")
    lines += [
        "",
        "## 3. 0–1024有限时间最大Lyapunov指数",
        "",
        "下表中位数及范围来自4个词频分层初始token。`exp(λ)`是每一步的几何平均"
        "扰动倍率；这不是某一个端点Jacobian的谱半径。",
        "",
        "| checkpoint | median λ | token range | exp(median λ) |",
        "|---|---:|---:|---:|",
    ]
    for row in lyapunov:
        lines.append(
            f"| {row['checkpoint']} | {float(row['lyapunov_median_4tokens']):.6g} | "
            f"[{float(row['lyapunov_min_4tokens']):.6g}, "
            f"{float(row['lyapunov_max_4tokens']):.6g}] | "
            f"{float(row['geometric_factor_from_median']):.6g} |"
        )
    last_window_path = args.report_root / "processed/lyapunov_last256_summary.csv"
    period_path = args.report_root / "processed/period_checkpoint_summary.csv"
    if last_window_path.exists():
        last_window = base.read_csv(last_window_path)
        lines += [
            "",
            "### 3.1 最后256步（768–1024）Lyapunov统计",
            "",
            "统计样本是每个checkpoint的4个词频分层token。总体方差使用分母 `N=4`；"
            "样本方差使用分母 `N-1=3`。上下界是4个观测值的min/max，"
            "不是置信区间。",
            "",
            "| checkpoint | mean | population variance | sample variance | lower=min | upper=max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in last_window:
            lines.append(
                f"| {row['checkpoint']} | {float(row['lyapunov_mean']):.7g} | "
                f"{float(row['lyapunov_population_variance']):.7g} | "
                f"{float(row['lyapunov_sample_variance']):.7g} | "
                f"{float(row['lyapunov_lower_bound_min']):.7g} | "
                f"{float(row['lyapunov_upper_bound_max']):.7g} |"
            )
    if period_path.exists():
        periods = base.read_csv(period_path)
        lines += [
            "",
            "### 3.2 指定checkpoint的动态周期估计",
            "",
            "周期在完整512维hidden state上估计，不使用投影坐标。分析区间是"
            "`dynamic step 512–1024`，候选整数时滞为1–128，因此每个候选周期"
            "至少可观察4次。距离用轨道pairwise-distance P95归一化；"
            "`normalized P95`越接近0，跨周期闭合越好。",
            "",
            "`quadratic period`是在最佳整数时滞及其左右邻点上进行抛物线插值，"
            "只用于给出亚步近似。实验14预注册阈值为：strict `≤1e-4`、"
            "approximate `≤1e-2`。未通过阈值时仅称为主回归时滞，不能据此确认"
            "严格极限环。",
            "",
            "| checkpoint | integer period | quadratic period | token period range | normalized P95 range | absolute P95 range | classification |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in periods:
            lines.append(
                f"| {row['checkpoint']} | {int(row['estimated_period_integer'])} | "
                f"{float(row['estimated_period_quadratic']):.4f} | "
                f"[{int(row['token_period_min'])}, {int(row['token_period_max'])}] | "
                f"[{float(row['normalized_p95_min_4tokens']):.5g}, "
                f"{float(row['normalized_p95_max_4tokens']):.5g}] | "
                f"[{float(row['absolute_p95_min_4tokens']):.5g}, "
                f"{float(row['absolute_p95_max_4tokens']):.5g}] | "
                f"{row['classification']} |"
            )
    lines += [
        "",
        "## 4. Jacobian四个尺度",
        "",
        "- `spectral radius = max|λᵢ|`：取特征值的模，反映局部渐近模态尺度。",
        "- `spectral abscissa = max Re(λᵢ)`：取特征值实部；伴随列记录该特征值"
        "的虚部和模，不能用它替代离散系统的谱半径。",
        "- `operator norm = σ₁`：最大奇异值，无实部或虚部，表示单步最强扰动放大。",
        "- `Frobenius norm = sqrt(ΣᵢⱼJᵢⱼ²)=sqrt(Σₖσₖ²)`："
        "所有Jacobian元素平方和开根号，也等于全部奇异值平方和开根号；"
        "它衡量整体线性响应能量，不是最强单一方向的放大率。",
        "",
        "| checkpoint | spectral radius range | max Re range | operator norm range | Frobenius norm range | norm/radius range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in checkpoints:
        group = [row for row in metrics if row["checkpoint"] == checkpoint]
        def span(field: str) -> str:
            values = [float(row[field]) for row in group]
            return f"[{min(values):.5g}, {max(values):.5g}]"
        lines.append(
            f"| {checkpoint} | {span('spectral_radius')} | {span('spectral_abscissa')} | "
            f"{span('operator_norm_2')} | {span('jacobian_frobenius_norm')} | "
            f"{span('operator_norm_over_spectral_radius')} |"
        )
    lines += [
        "",
        "Jacobian图提供两种组织方式：`by_window`以每个checkpoint为子图、"
        "横轴为dynamic step；`by_checkpoint`以training checkpoint为横轴、"
        "每条线对应一个窗口端点。中心线是4个token的中位数，阴影是min–max。",
        "",
        "- [`spectral_radius_by_checkpoint.png`](figures/spectral_radius_by_checkpoint.png)",
        "- [`spectral_abscissa_by_checkpoint.png`](figures/spectral_abscissa_by_checkpoint.png)",
        "- [`operator_norm_2_by_checkpoint.png`](figures/operator_norm_2_by_checkpoint.png)",
        "- [`jacobian_frobenius_norm_by_checkpoint.png`](figures/jacobian_frobenius_norm_by_checkpoint.png)",
        "- [`jacobian_three_metrics_by_checkpoint.png`](figures/jacobian_three_metrics_by_checkpoint.png)",
        "- [`jacobian_four_metrics_by_checkpoint.png`](figures/jacobian_four_metrics_by_checkpoint.png)",
        "- [`jacobian_three_metrics_with_loss.csv`](processed/jacobian_three_metrics_with_loss.csv)",
        "- [`jacobian_four_metrics_with_loss.csv`](processed/jacobian_four_metrics_with_loss.csv)",
    ]
    lines += [
        "",
        "## 5. 三类最近词与可信度",
        "",
        "- cosine：input embedding方向相似度；可信度为top1-top2 similarity margin。",
        "- Euclidean：input embedding绝对距离；可信度为`(d2-d1)/d1`。",
        "- LM-head：输出头logit/softmax的预测；可信度同时保存概率差与logit差。",
        "",
        "### 5.1 Cosine最近词",
        "",
        base.neighbor_table(metrics, tokens, endpoints, "cosine"),
        "",
        "### 5.2 Euclidean最近词",
        "",
        base.neighbor_table(metrics, tokens, endpoints, "euclidean"),
        "",
        "### 5.3 LM-head最近词",
        "",
        base.neighbor_table(metrics, tokens, endpoints, "lm_head"),
        "",
        "完整机器可读结果见"
        "[`window_endpoint_metrics.csv`](processed/window_endpoint_metrics.csv)，"
        "其中三种方法各自的top5也以JSON保存。",
        "",
        "## 6. 输出",
        "",
        "- [`projection_triples.jsonl`](processed/projection_triples.jsonl)",
        "- [`lyapunov_by_token_window.csv`](processed/lyapunov_by_token_window.csv)",
        "- [`lyapunov_by_token_overall.csv`](processed/lyapunov_by_token_overall.csv)",
        "- [`lyapunov_checkpoint_summary.csv`](processed/lyapunov_checkpoint_summary.csv)",
        "- [`lyapunov_last256_summary.csv`](processed/lyapunov_last256_summary.csv)",
        "- [`period_checkpoint_summary.csv`](processed/period_checkpoint_summary.csv)",
        "- [`period_by_token.csv`](processed/period_by_token.csv)",
        "- [`period_recurrence_rows.csv`](processed/period_recurrence_rows.csv)",
        "- [`checkpoint_loss.csv`](processed/checkpoint_loss.csv)",
        "- [`window_endpoint_metrics.csv`](processed/window_endpoint_metrics.csv)",
        "- [`lyapunov_by_checkpoint.png`](figures/lyapunov_by_checkpoint.png)",
    ]
    (args.report_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "dynamics",
            "projection",
            "lyapunov",
            "period",
            "loss",
            "analysis",
            "report",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--upstream-states-root", type=Path, default=Path("/nonexistent"))
    parser.add_argument("--report-root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wikitext-train-arrow", type=Path, default=base.DEFAULT_ARROW)
    parser.add_argument("--token-manifest", type=Path, default=TOKEN_MANIFEST)
    parser.add_argument("--loss-root", type=Path, default=LOSS_ROOT)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--checkpoints", nargs="+", choices=CHECKPOINTS)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--projection-seed", type=int, default=1616)
    parser.add_argument("--lyapunov-seed", type=int, default=1617)
    parser.add_argument("--lyapunov-epsilon", type=float, default=1e-30)
    parser.add_argument("--period-tail-steps", type=int, default=512)
    parser.add_argument("--period-max", type=int, default=128)
    parser.add_argument("--period-seed", type=int, default=1618)
    parser.add_argument("--jacobian-chunk-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoints = (args.checkpoints or CHECKPOINTS)[: args.max_checkpoints]
    tokens = load_tokens(args.token_manifest)
    base.window_endpoints(args.steps, args.window)
    for directory in (args.report_root / "processed", args.report_root / "figures", args.data_root):
        directory.mkdir(parents=True, exist_ok=True)
    if args.stage in {"dynamics", "all"}:
        base.dynamics_stage(args, checkpoints, tokens)
    if args.stage in {"projection", "all"}:
        projection_stage(args, checkpoints, tokens)
    if args.stage in {"lyapunov", "all"}:
        lyapunov_stage(args, checkpoints, tokens)
    if args.stage in {"period", "all"}:
        period_stage(args, checkpoints, tokens)
    if args.stage in {"loss", "all"}:
        loss_stage(args, checkpoints)
    if args.stage in {"analysis", "all"}:
        analysis_stage(args, checkpoints, tokens)
    if args.stage in {"report", "all"}:
        report_stage(args, checkpoints, tokens)


if __name__ == "__main__":
    main()
