#!/usr/bin/env python
"""Aggregate and visualize the rolling next-token pilot without mutating raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot"
)
CHECKPOINT_ORDER = ["step0", "step143000"]
COLORS = {"step0": "#377eb8", "step143000": "#e68613"}
MARKERS = {"step0": "o", "step143000": "^"}


def read_jsonl(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    raw_paths = sorted(root.glob("*/raw/*__summary.jsonl"))
    trajectory_paths = sorted(root.glob("*/raw/*__trajectory.jsonl"))
    if not raw_paths or not trajectory_paths:
        raise FileNotFoundError(f"pilot JSONL files not found under {root}")

    summary = read_jsonl(raw_paths)
    trajectory = read_jsonl(trajectory_paths)
    summary["anchor_label"] = summary.apply(
        lambda r: f"doc{int(r.document_index)}@{int(r.anchor_offset)}", axis=1
    )
    summary["innovation_share_of_total_squared"] = 1.0 - summary[
        "shift_fraction_of_total_squared"
    ]
    summary["lyapunov_sign"] = np.where(
        summary["maximal_lyapunov_mean"] > 0, "positive", "negative"
    )
    summary["hard_cycle_detected"] = summary["hard_cycle_length"].notna()
    summary["hard_cycle_length_numeric"] = summary["hard_cycle_length"].fillna(np.nan)

    keep = [
        "checkpoint",
        "anchor_index",
        "anchor_label",
        "document_index",
        "anchor_offset",
        "total_geomean",
        "shift_normalized_frobenius",
        "innovation_geomean",
        "innovation_output_geomean",
        "shift_fraction_of_total_squared",
        "innovation_share_of_total_squared",
        "maximal_lyapunov_mean",
        "final_to_initial_separation",
        "tail_relative_step_delta_mean",
        "tail_soft_entropy_mean",
        "tail_soft_top1_probability_mean",
        "hard_cycle_detected",
        "hard_cycle_length_numeric",
        "hard_cycle_start",
        "hard_unique_token_ratio",
        "hard_unique_windows",
        "soft_seconds",
        "frobenius_seconds",
        "lyapunov_seconds",
        "hard_seconds",
    ]
    anchor = summary[keep].sort_values(["checkpoint", "anchor_index"])
    grouped = summary.groupby("checkpoint", sort=False)
    checkpoint = grouped.agg(
        anchors=("anchor_index", "count"),
        total_frobenius_mean=("total_geomean", "mean"),
        total_frobenius_std=("total_geomean", "std"),
        innovation_total_mean=("innovation_geomean", "mean"),
        innovation_total_std=("innovation_geomean", "std"),
        innovation_output_mean=("innovation_output_geomean", "mean"),
        innovation_output_std=("innovation_output_geomean", "std"),
        shift_fraction_mean=("shift_fraction_of_total_squared", "mean"),
        lyapunov_mean=("maximal_lyapunov_mean", "mean"),
        lyapunov_std=("maximal_lyapunov_mean", "std"),
        positive_lyapunov_fraction=("lyapunov_sign", lambda x: float((x == "positive").mean())),
        final_separation_median=("final_to_initial_separation", "median"),
        tail_relative_delta_mean=("tail_relative_step_delta_mean", "mean"),
        entropy_mean=("tail_soft_entropy_mean", "mean"),
        top1_probability_mean=("tail_soft_top1_probability_mean", "mean"),
        hard_cycle_fraction=("hard_cycle_detected", "mean"),
        hard_cycle_length_median=("hard_cycle_length_numeric", "median"),
        total_seconds=(
            "soft_seconds",
            lambda x: float(
                summary.loc[x.index, ["soft_seconds", "frobenius_seconds", "lyapunov_seconds", "hard_seconds"]]
                .sum(axis=1)
                .sum()
            ),
        ),
    ).reset_index()
    checkpoint["shift_normalized_frobenius"] = summary[
        "shift_normalized_frobenius"
    ].iloc[0]
    order = {name: i for i, name in enumerate(CHECKPOINT_ORDER)}
    checkpoint["_order"] = checkpoint["checkpoint"].map(order)
    checkpoint = checkpoint.sort_values("_order").drop(columns="_order")

    processed = root / "processed"
    figures = root / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    anchor_path = processed / "pilot_anchor_metrics.csv"
    checkpoint_path = processed / "pilot_checkpoint_summary.csv"
    anchor.to_csv(anchor_path, index=False)
    checkpoint.to_csv(checkpoint_path, index=False)

    configure_style()
    checkpoints = [c for c in CHECKPOINT_ORDER if c in set(summary["checkpoint"])]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(summary["anchor_label"].unique()))
    labels = list(summary.sort_values("anchor_index")["anchor_label"].unique())
    offsets = np.linspace(-0.12, 0.12, len(checkpoints))
    for offset, ckpt in zip(offsets, checkpoints):
        part = summary[summary.checkpoint == ckpt].sort_values("anchor_index")
        ax.scatter(
            x + offset,
            part.total_geomean,
            color=COLORS[ckpt],
            marker=MARKERS[ckpt],
            label=f"{ckpt}: full J",
            zorder=3,
        )
    shift = float(summary.shift_normalized_frobenius.iloc[0])
    ax.axhline(shift, color="#555555", linestyle="--", label=f"shift-only = {shift:.6f}")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel(r"Normalized Frobenius $\|J\|_F/\sqrt{LH}$")
    ax.set_title("Rolling operator: full Jacobian versus analytic shift baseline")
    ax.legend(fontsize=8)
    frob_fig = figures / "pilot_total_frobenius_vs_shift.png"
    save_figure(fig, frob_fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for ckpt in checkpoints:
        part = summary[summary.checkpoint == ckpt]
        for _, row in part.iterrows():
            ax.scatter(
                row.innovation_output_geomean,
                row.maximal_lyapunov_mean,
                color=COLORS[ckpt],
                marker=MARKERS[ckpt],
                s=55,
                zorder=3,
            )
            ax.annotate(
                row.anchor_label,
                (row.innovation_output_geomean, row.maximal_lyapunov_mean),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.scatter([], [], color=COLORS[ckpt], marker=MARKERS[ckpt], label=ckpt)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel(r"Innovation Jacobian $\|J_{new}\|_F/\sqrt{H}$")
    ax.set_ylabel("Benettin maximal Lyapunov exponent (per step)")
    ax.set_title("Innovation sensitivity and long-horizon expansion")
    ax.legend()
    scatter_fig = figures / "pilot_innovation_vs_lyapunov.png"
    save_figure(fig, scatter_fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ckpt in checkpoints:
        part = summary[summary.checkpoint == ckpt].sort_values("anchor_index")
        axes[0].scatter(
            part.anchor_label,
            part.tail_relative_step_delta_mean,
            color=COLORS[ckpt],
            marker=MARKERS[ckpt],
            label=ckpt,
        )
        cycle_y = part.hard_cycle_length_numeric.fillna(300.0)
        axes[1].scatter(
            part.anchor_label,
            cycle_y,
            color=COLORS[ckpt],
            marker=MARKERS[ckpt],
            label=ckpt,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tail relative step delta")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Soft rolling trajectory")
    axes[1].axhline(256, color="#777777", linestyle="--", linewidth=1)
    axes[1].text(0.02, 0.95, "300 = no cycle within 256 steps", transform=axes[1].transAxes, va="top", fontsize=7)
    axes[1].set_ylabel("Exact hard full-window cycle length")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_title("Hard argmax rollout")
    axes[0].legend(fontsize=8)
    behavior_fig = figures / "pilot_soft_hard_behavior.png"
    save_figure(fig, behavior_fig)

    manifest = {
        "experiment": "pythia_rolling_next_token_pilot",
        "raw_summary_files": [str(path.resolve()) for path in raw_paths],
        "raw_trajectory_files": [str(path.resolve()) for path in trajectory_paths],
        "processed_files": [str(anchor_path.resolve()), str(checkpoint_path.resolve())],
        "figures": [
            {
                "path": str(frob_fig.resolve()),
                "question": "Is normalized full-J Frobenius near one because of the rolling shift?",
                "caveat": "Four anchors; stochastic Hutchinson estimate with four probes at two states.",
            },
            {
                "path": str(scatter_fig.resolve()),
                "question": "Does innovation sensitivity track long-horizon expansion?",
                "caveat": "One Benettin probe and 128 measured steps per anchor.",
            },
            {
                "path": str(behavior_fig.resolve()),
                "question": "Do soft convergence and hard exact cycles differ across checkpoints and anchors?",
                "caveat": "Hard cycles are exact full-window repeats detected only within 256 steps.",
            },
        ],
    }
    manifest_path = figures / "pilot_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("ANCHOR_METRICS")
    print(anchor.to_string(index=False))
    print("CHECKPOINT_SUMMARY")
    print(checkpoint.to_string(index=False))
    print("OUTPUTS")
    for path in [anchor_path, checkpoint_path, frob_fig, scatter_fig, behavior_fig, manifest_path]:
        print(path.resolve())


if __name__ == "__main__":
    main()
