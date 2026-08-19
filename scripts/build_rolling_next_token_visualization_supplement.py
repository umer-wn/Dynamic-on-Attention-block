#!/usr/bin/env python
"""Build read-only visualization supplements for rolling next-token dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/home/luohaoming/model_feature_experiments/rolling_next_token_criticality")
CHECKPOINTS = ["step0", "step1000", "step16000", "step143000"]
COLORS = {
    "step0": "#6a6a6a",
    "step1000": "#377eb8",
    "step16000": "#4daf4a",
    "step143000": "#e68613",
}
SCOPES = {
    "full": ["projection_full_0", "projection_full_1", "projection_full_2"],
    "newest": ["projection_newest_0", "projection_newest_1", "projection_newest_2"],
}


def read_jsonl(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def quantile_rows(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for (checkpoint, step), group in frame.groupby(["checkpoint", "step_index"]):
        for metric in metrics:
            values = group[metric].astype(float)
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "step_index": int(step),
                    "metric": metric,
                    "n": int(values.notna().sum()),
                    "q25": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q75": float(values.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def interpolated_crossings(
    frame: pd.DataFrame,
    columns: list[str],
    section: float,
    centered: bool,
) -> list[dict]:
    ordered = frame.sort_values("step_index").reset_index(drop=True)
    values = ordered[columns].astype(float).to_numpy()
    centers = np.median(values, axis=0) if centered else np.zeros(3)
    projected = values - centers
    local_section = 0.0 if centered else float(section)
    rows: list[dict] = []
    for idx in range(len(ordered) - 1):
        z0_a = float(projected[idx, 0])
        z0_b = float(projected[idx + 1, 0])
        if z0_a <= local_section < z0_b and z0_b != z0_a:
            alpha = (local_section - z0_a) / (z0_b - z0_a)
            point = projected[idx] + alpha * (projected[idx + 1] - projected[idx])
            rows.append(
                {
                    "checkpoint": ordered.loc[idx, "checkpoint"],
                    "document_index": int(ordered.loc[idx, "document_index"]),
                    "anchor_offset": int(ordered.loc[idx, "anchor_offset"]),
                    "scope": "full" if columns[0].startswith("projection_full") else "newest",
                    "section_mode": "trajectory_centered_zero" if centered else "shared_absolute",
                    "section_value": float(local_section),
                    "absolute_section_value": float(centers[0] + local_section),
                    "step_before": int(ordered.loc[idx, "step_index"]),
                    "step_after": int(ordered.loc[idx + 1, "step_index"]),
                    "alpha": float(alpha),
                    "poincare_z1": float(point[1]),
                    "poincare_z2": float(point[2]),
                    "center_z0": float(centers[0]),
                    "center_z1": float(centers[1]),
                    "center_z2": float(centers[2]),
                    "crossing_order": len(rows),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--document-index", type=int, default=264)
    parser.add_argument("--anchor-offset", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    behavior_summary_paths = sorted((root / "main_behavior").glob("*/raw/*__summary.jsonl"))
    behavior_trajectory_paths = sorted((root / "main_behavior").glob("*/raw/*__trajectory.jsonl"))
    tangent_summary_paths = sorted((root / "main_tangent").glob("*/raw/*__summary.jsonl"))
    if not (len(behavior_summary_paths) == len(behavior_trajectory_paths) == len(tangent_summary_paths) == 4):
        raise FileNotFoundError("expected four behavior summaries/trajectories and four tangent summaries")

    behavior = read_jsonl(behavior_summary_paths)
    trajectory = read_jsonl(behavior_trajectory_paths)
    tangent = read_jsonl(tangent_summary_paths)
    expected_projection_fields = [name for columns in SCOPES.values() for name in columns]
    if trajectory[expected_projection_fields].isna().any().any():
        raise AssertionError("projection fields contain missing values")
    if behavior.groupby("checkpoint").size().to_dict() != {checkpoint: 32 for checkpoint in CHECKPOINTS}:
        raise AssertionError("behavior summary counts differ from 32/checkpoint")
    if tangent.groupby("checkpoint").size().to_dict() != {checkpoint: 8 for checkpoint in CHECKPOINTS}:
        raise AssertionError("tangent summary counts differ from 8/checkpoint")
    trajectory_counts = trajectory.groupby("checkpoint").size().to_dict()
    if trajectory_counts != {checkpoint: 32 * 256 for checkpoint in CHECKPOINTS}:
        raise AssertionError(f"trajectory counts differ from 8192/checkpoint: {trajectory_counts}")

    selected = trajectory[
        (trajectory.document_index.astype(int) == int(args.document_index))
        & (trajectory.anchor_offset.astype(int) == int(args.anchor_offset))
    ].copy()
    if selected.groupby("checkpoint").size().to_dict() != {checkpoint: 256 for checkpoint in CHECKPOINTS}:
        raise AssertionError("selected matched anchor does not have 256 rows/checkpoint")
    token_variants = behavior[
        (behavior.document_index.astype(int) == int(args.document_index))
        & (behavior.anchor_offset.astype(int) == int(args.anchor_offset))
    ].initial_token_ids.map(tuple).nunique()
    if token_variants != 1:
        raise AssertionError("selected initial token window differs across checkpoints")

    out = root / "visualization_supplement"
    processed = out / "processed"
    figures = out / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    metrics = ["nearby_distance", "relative_step_delta", "soft_entropy", "soft_top1_probability"]
    quantiles = quantile_rows(trajectory, metrics)
    quantiles.to_csv(processed / "soft_trajectory_quantiles.csv", index=False)
    hard_columns = [
        "checkpoint",
        "anchor_index",
        "document_index",
        "anchor_offset",
        "hard_cycle_length",
        "hard_cycle_start",
        "hard_unique_token_ratio",
        "hard_adjacent_repeat_fraction",
    ]
    hard = behavior[hard_columns].copy()
    hard["hard_cycle_detected"] = hard.hard_cycle_length.notna()
    hard.to_csv(processed / "hard_cycle_anchor_metrics.csv", index=False)
    selected.to_csv(processed / "selected_anchor_projection_rows.csv", index=False)

    span_rows: list[dict] = []
    for scope, columns in SCOPES.items():
        for checkpoint in CHECKPOINTS:
            part = selected[selected.checkpoint == checkpoint]
            values = part[columns].astype(float)
            span_rows.append(
                {
                    "scope": scope,
                    "checkpoint": checkpoint,
                    "document_index": int(args.document_index),
                    "anchor_offset": int(args.anchor_offset),
                    "z0_min": float(values.iloc[:, 0].min()),
                    "z0_max": float(values.iloc[:, 0].max()),
                    "z0_span": float(values.iloc[:, 0].max() - values.iloc[:, 0].min()),
                    "z1_span": float(values.iloc[:, 1].max() - values.iloc[:, 1].min()),
                    "z2_span": float(values.iloc[:, 2].max() - values.iloc[:, 2].min()),
                    "coordinate_rms_about_median": float(
                        np.sqrt(np.mean(np.square(values.to_numpy() - np.median(values.to_numpy(), axis=0))))
                    ),
                }
            )
    spans = pd.DataFrame(span_rows)
    spans.to_csv(processed / "selected_anchor_projection_spans.csv", index=False)

    shared_crossing_rows: list[dict] = []
    centered_crossing_rows: list[dict] = []
    shared_sections: dict[str, float] = {}
    for scope, columns in SCOPES.items():
        shared_section = float(selected[columns[0]].median())
        shared_sections[scope] = shared_section
        for checkpoint in CHECKPOINTS:
            part = selected[selected.checkpoint == checkpoint]
            shared_crossing_rows.extend(
                interpolated_crossings(part, columns, section=shared_section, centered=False)
            )
            centered_crossing_rows.extend(
                interpolated_crossings(part, columns, section=0.0, centered=True)
            )
    shared_crossings = pd.DataFrame(shared_crossing_rows)
    centered_crossings = pd.DataFrame(centered_crossing_rows)
    shared_crossings.to_csv(processed / "projected_poincare_shared_absolute.csv", index=False)
    centered_crossings.to_csv(processed / "projected_poincare_centered.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
        }
    )

    # Soft trajectory diagnostics across all 32 anchors/checkpoint.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True)
    metric_specs = [
        ("nearby_distance", "Nearby distance", True),
        ("relative_step_delta", "Relative step delta", True),
        ("soft_entropy", "Soft token entropy (nats)", False),
        ("soft_top1_probability", "Top-1 probability", False),
    ]
    for ax, (metric, ylabel, logy) in zip(axes.flat, metric_specs):
        for checkpoint in CHECKPOINTS:
            part = quantiles[(quantiles.checkpoint == checkpoint) & (quantiles.metric == metric)].sort_values(
                "step_index"
            )
            x = part.step_index.to_numpy(dtype=float)
            median = part["median"].to_numpy(dtype=float)
            q25 = part.q25.to_numpy(dtype=float)
            q75 = part.q75.to_numpy(dtype=float)
            if logy:
                median = np.clip(median, 1e-12, None)
                q25 = np.clip(q25, 1e-12, None)
                q75 = np.clip(q75, 1e-12, None)
                ax.set_yscale("log")
            ax.plot(x, median, color=COLORS[checkpoint], label=checkpoint, linewidth=1.8)
            ax.fill_between(x, q25, q75, color=COLORS[checkpoint], alpha=0.16)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}: median and IQR over 32 anchors")
    axes[1, 0].set_xlabel("Eval step after burn-in 512")
    axes[1, 1].set_xlabel("Eval step after burn-in 512")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Rolling next-token soft trajectory diagnostics (behavior layer)")
    soft_path = figures / "rolling_soft_trajectory_diagnostics.png"
    save_figure(fig, soft_path)

    # Hard cycle metrics; only detected cycles are placed on cycle axes.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    rng = np.random.default_rng(1234)
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        part = hard[hard.checkpoint == checkpoint]
        detected = part[part.hard_cycle_detected]
        jitter = rng.uniform(-0.12, 0.12, len(detected))
        axes[0].scatter(
            checkpoint_index + jitter,
            detected.hard_cycle_length,
            color=COLORS[checkpoint],
            alpha=0.7,
            s=24,
        )
        axes[1].scatter(
            checkpoint_index + jitter,
            detected.hard_cycle_start,
            color=COLORS[checkpoint],
            alpha=0.7,
            s=24,
        )
        label = f"{len(detected)}/32 detected"
        axes[0].text(checkpoint_index, 0.03, label, rotation=90, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    for ax in axes:
        ax.set_xticks(range(4), CHECKPOINTS, rotation=18)
        ax.set_yscale("log")
    axes[0].set_ylabel("Exact full-window cycle length")
    axes[0].set_title("Detected cycle lengths within 512 hard steps")
    axes[1].set_ylabel("First cycle start step")
    axes[1].set_title("Transient length before first exact repeat")
    fig.suptitle("Hard argmax rollout, 32 anchors/checkpoint; missing cycles are not assigned a length")
    hard_path = figures / "rolling_hard_cycle_distributions.png"
    save_figure(fig, hard_path)

    # Full-window and newest-token 3D trajectories for one matched anchor only.
    fig = plt.figure(figsize=(16.0, 10.0))
    for row_index, (scope, columns) in enumerate(SCOPES.items()):
        for column_index, checkpoint in enumerate(CHECKPOINTS):
            ax = fig.add_subplot(2, 4, row_index * 4 + column_index + 1, projection="3d")
            part = selected[selected.checkpoint == checkpoint].sort_values("step_index")
            values = part[columns].astype(float).to_numpy()
            steps = part.step_index.to_numpy(dtype=float)
            ax.plot(values[:, 0], values[:, 1], values[:, 2], color="#777777", linewidth=0.8, alpha=0.65)
            scatter = ax.scatter(
                values[::4, 0], values[::4, 1], values[::4, 2], c=steps[::4], cmap="viridis", s=9, alpha=0.8
            )
            ax.scatter(*values[0], color="#1b9e77", marker="o", s=35, label="start")
            ax.scatter(*values[-1], color="#d95f02", marker="x", s=45, label="end")
            span = np.ptp(values, axis=0)
            ax.set_title(
                f"{scope}, {checkpoint}\nspans=({span[0]:.2e},{span[1]:.2e},{span[2]:.2e})",
                fontsize=9,
                pad=10,
            )
            ax.set_xlabel("z0")
            ax.set_ylabel("z1")
            ax.set_zlabel("z2")
            if row_index == 0 and column_index == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        f"Matched rolling trajectories: doc{args.document_index}@{args.anchor_offset}; time increases with viridis color",
        y=0.97,
    )
    projection_path = figures / "rolling_projection_trajectories_full_vs_newest.png"
    fig.subplots_adjust(top=0.90, bottom=0.06, hspace=0.14, wspace=0.10)
    fig.savefig(projection_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Return maps for the same matched anchor, never pooled across samples.
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.3))
    for row_index, (scope, columns) in enumerate(SCOPES.items()):
        for column_index, checkpoint in enumerate(CHECKPOINTS):
            ax = axes[row_index, column_index]
            part = selected[selected.checkpoint == checkpoint].sort_values("step_index")
            z = part[columns[0]].to_numpy(dtype=float)
            steps = part.step_index.to_numpy(dtype=float)
            ax.scatter(z[:-1], z[1:], c=steps[:-1], cmap="viridis", s=11, alpha=0.7)
            lo = float(min(z[:-1].min(), z[1:].min()))
            hi = float(max(z[:-1].max(), z[1:].max()))
            ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=0.8)
            ax.set_title(f"{scope}, {checkpoint}")
            ax.set_xlabel("z0(t)")
            ax.set_ylabel("z0(t+1)")
    fig.suptitle(f"Rolling return maps for one matched anchor only: doc{args.document_index}@{args.anchor_offset}")
    return_path = figures / "rolling_projection_return_maps_full_vs_newest.png"
    save_figure(fig, return_path)

    def plot_poincare(frame: pd.DataFrame, title: str, path: Path, shared: bool) -> None:
        fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2))
        for row_index, scope in enumerate(SCOPES):
            for column_index, checkpoint in enumerate(CHECKPOINTS):
                ax = axes[row_index, column_index]
                if frame.empty:
                    part = frame
                else:
                    part = frame[(frame.scope == scope) & (frame.checkpoint == checkpoint)]
                if part.empty:
                    ax.text(0.5, 0.5, "No upward crossing", ha="center", va="center", transform=ax.transAxes)
                else:
                    ax.scatter(
                        part.poincare_z1,
                        part.poincare_z2,
                        c=part.crossing_order,
                        cmap="viridis",
                        s=24,
                        alpha=0.8,
                    )
                section_text = f"c={shared_sections[scope]:.3e}" if shared else "centered c=0"
                ax.set_title(f"{scope}, {checkpoint}\n{section_text}; n={len(part)}")
                ax.set_xlabel("z1* at crossing")
                ax.set_ylabel("z2* at crossing")
        fig.suptitle(title)
        save_figure(fig, path)

    shared_poincare_path = figures / "rolling_projected_poincare_shared_absolute.png"
    centered_poincare_path = figures / "rolling_projected_poincare_centered.png"
    plot_poincare(
        shared_crossings,
        f"Projected Poincaré: one absolute section shared across checkpoints, doc{args.document_index}@{args.anchor_offset}",
        shared_poincare_path,
        shared=True,
    )
    plot_poincare(
        centered_crossings,
        f"Centered diagnostic Poincaré: each trajectory median-centered, doc{args.document_index}@{args.anchor_offset}",
        centered_poincare_path,
        shared=False,
    )

    figure_paths = [
        soft_path,
        hard_path,
        projection_path,
        return_path,
        shared_poincare_path,
        centered_poincare_path,
    ]
    processed_paths = [
        processed / "soft_trajectory_quantiles.csv",
        processed / "hard_cycle_anchor_metrics.csv",
        processed / "selected_anchor_projection_rows.csv",
        processed / "selected_anchor_projection_spans.csv",
        processed / "projected_poincare_shared_absolute.csv",
        processed / "projected_poincare_centered.csv",
    ]
    crossing_counts = {
        mode: {
            scope: {
                checkpoint: int(
                    len(
                        frame[
                            (frame.scope == scope) & (frame.checkpoint == checkpoint)
                        ]
                    )
                )
                if not frame.empty
                else 0
                for checkpoint in CHECKPOINTS
            }
            for scope in SCOPES
        }
        for mode, frame in [("shared_absolute", shared_crossings), ("centered", centered_crossings)]
    }
    manifest = {
        "experiment": "rolling_next_token_visualization_supplement",
        "source_behavior_summary_files": [str(path.resolve()) for path in behavior_summary_paths],
        "source_behavior_trajectory_files": [str(path.resolve()) for path in behavior_trajectory_paths],
        "source_tangent_summary_files": [str(path.resolve()) for path in tangent_summary_paths],
        "selected_anchor": {"document_index": int(args.document_index), "anchor_offset": int(args.anchor_offset)},
        "projection_scope": {
            "full": "unit-random projection of the complete [1,64,512] rolling state",
            "newest": "unit-random projection of only the newest [1,512] expected embedding",
        },
        "shared_absolute_sections": shared_sections,
        "crossing_counts": crossing_counts,
        "processed_files": [str(path.resolve()) for path in processed_paths],
        "figures": [str(path.resolve()) for path in figure_paths],
        "caveats": [
            "Three random projections are non-injective and cannot establish chaos by themselves.",
            "3D axes autoscale independently; coordinate spans are printed in every panel.",
            "Projection figures use exactly one matched anchor and never pool samples.",
            "Shared-absolute Poincare sections may have no crossings; centered sections are diagnostic and lose absolute location.",
            "No Frobenius or Lyapunov JVP was recomputed in this read-only visualization pass.",
        ],
    }
    manifest_path = out / "visualization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("VALIDATION behavior=32/checkpoint tangent=8/checkpoint trajectory=8192/checkpoint")
    print("SELECTED_ANCHOR", args.document_index, args.anchor_offset)
    print("SHARED_SECTIONS", json.dumps(shared_sections, sort_keys=True))
    print("CROSSING_COUNTS", json.dumps(crossing_counts, sort_keys=True))
    print("PROJECTION_SPANS")
    print(spans.to_string(index=False))
    print("OUTPUT", out)


if __name__ == "__main__":
    main()
