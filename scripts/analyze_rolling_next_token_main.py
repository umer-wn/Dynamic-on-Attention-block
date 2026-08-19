#!/usr/bin/env python
"""Validate, aggregate, and visualize the four-checkpoint rolling main experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/home/luohaoming/model_feature_experiments/rolling_next_token_criticality")
CHECKPOINTS = ["step0", "step1000", "step16000", "step143000"]
STEPS = {name: int(name.removeprefix("step")) for name in CHECKPOINTS}
COLORS = {
    "step0": "#6a6a6a",
    "step1000": "#377eb8",
    "step16000": "#4daf4a",
    "step143000": "#e68613",
}


def read_jsonl(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def format_training_axis(ax: plt.Axes) -> None:
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(-20, 200000)
    ax.set_xticks([0, 1000, 16000, 143000], ["0", "1k", "16k", "143k"])


def validate(behavior: pd.DataFrame, tangent: pd.DataFrame) -> dict[str, object]:
    errors: list[str] = []
    bcounts = behavior.groupby("checkpoint").size().to_dict()
    tcounts = tangent.groupby("checkpoint").size().to_dict()
    for checkpoint in CHECKPOINTS:
        if bcounts.get(checkpoint) != 32:
            errors.append(f"behavior {checkpoint}: expected 32, got {bcounts.get(checkpoint)}")
        if tcounts.get(checkpoint) != 8:
            errors.append(f"tangent {checkpoint}: expected 8, got {tcounts.get(checkpoint)}")

    key = ["document_index", "anchor_offset"]
    for name, frame in [("behavior", behavior), ("tangent", tangent)]:
        expected: set[tuple[int, int]] | None = None
        for checkpoint in CHECKPOINTS:
            part = frame[frame.checkpoint == checkpoint]
            current = set(map(tuple, part[key].astype(int).to_numpy()))
            if expected is None:
                expected = current
            elif current != expected:
                errors.append(f"{name}: anchor keys differ at {checkpoint}")
        token_variants = frame.groupby(key)["initial_token_ids"].apply(
            lambda values: len({tuple(v) for v in values})
        )
        if int(token_variants.max()) != 1:
            errors.append(f"{name}: initial token windows differ across checkpoints")

    for name, frame, burn, eval_steps in [
        ("behavior", behavior, 512, 256),
        ("tangent", tangent, 256, 128),
    ]:
        if set(frame.burn_in_steps.astype(int)) != {burn}:
            errors.append(f"{name}: inconsistent burn-in")
        if set(frame.eval_steps.astype(int)) != {eval_steps}:
            errors.append(f"{name}: inconsistent eval steps")
        if set(frame.sequence_length.astype(int)) != {64}:
            errors.append(f"{name}: inconsistent window length")
        if set(frame.temperature.astype(float)) != {1.0}:
            errors.append(f"{name}: inconsistent temperature")
    if errors:
        raise AssertionError("; ".join(errors))
    return {
        "status": "passed",
        "behavior_counts": bcounts,
        "tangent_counts": tcounts,
        "same_anchor_keys_across_checkpoints": True,
        "same_initial_token_ids_across_checkpoints": True,
        "window_length": 64,
        "temperature": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    behavior_paths = sorted((root / "main_behavior").glob("*/raw/*__summary.jsonl"))
    tangent_paths = sorted((root / "main_tangent").glob("*/raw/*__summary.jsonl"))
    if len(behavior_paths) != 4 or len(tangent_paths) != 4:
        raise FileNotFoundError(
            f"expected 4 behavior and 4 tangent summary files; got {len(behavior_paths)}, {len(tangent_paths)}"
        )
    behavior = read_jsonl(behavior_paths)
    tangent = read_jsonl(tangent_paths)
    validation = validate(behavior, tangent)

    behavior["hard_cycle_detected"] = behavior.hard_cycle_length.notna()
    behavior["log10_final_separation"] = np.log10(
        behavior.final_to_initial_separation.clip(lower=1e-30)
    )
    tangent["positive_lyapunov"] = tangent.maximal_lyapunov_mean > 0
    tangent["innovation_share_of_total_squared"] = 1.0 - tangent.shift_fraction_of_total_squared

    bsummary = behavior.groupby("checkpoint", as_index=False).agg(
        behavior_anchors=("anchor_index", "count"),
        tail_relative_delta_mean=("tail_relative_step_delta_mean", "mean"),
        tail_relative_delta_std=("tail_relative_step_delta_mean", "std"),
        final_separation_median=("final_to_initial_separation", "median"),
        log10_final_separation_mean=("log10_final_separation", "mean"),
        entropy_mean=("tail_soft_entropy_mean", "mean"),
        entropy_std=("tail_soft_entropy_mean", "std"),
        top1_probability_mean=("tail_soft_top1_probability_mean", "mean"),
        hard_cycle_fraction=("hard_cycle_detected", "mean"),
        hard_cycle_length_median=("hard_cycle_length", "median"),
        hard_unique_token_ratio_mean=("hard_unique_token_ratio", "mean"),
        behavior_seconds=("soft_seconds", lambda x: float(behavior.loc[x.index, ["soft_seconds", "hard_seconds"]].sum().sum())),
    )
    tsummary = tangent.groupby("checkpoint", as_index=False).agg(
        tangent_anchors=("anchor_index", "count"),
        total_frobenius_mean=("total_geomean", "mean"),
        total_frobenius_std=("total_geomean", "std"),
        innovation_total_mean=("innovation_geomean", "mean"),
        innovation_total_std=("innovation_geomean", "std"),
        innovation_output_mean=("innovation_output_geomean", "mean"),
        innovation_output_std=("innovation_output_geomean", "std"),
        shift_fraction_mean=("shift_fraction_of_total_squared", "mean"),
        lyapunov_mean=("maximal_lyapunov_mean", "mean"),
        lyapunov_std=("maximal_lyapunov_mean", "std"),
        positive_lyapunov_fraction=("positive_lyapunov", "mean"),
        tangent_seconds=("soft_seconds", lambda x: float(tangent.loc[x.index, ["soft_seconds", "frobenius_seconds", "lyapunov_seconds"]].sum().sum())),
    )
    merged = bsummary.merge(tsummary, on="checkpoint", validate="one_to_one")
    merged["training_step"] = merged.checkpoint.map(STEPS)
    merged = merged.sort_values("training_step")

    match = tangent.merge(
        behavior,
        on=["checkpoint", "document_index", "anchor_offset"],
        suffixes=("_tangent", "_behavior"),
        validate="one_to_one",
    )
    correlation_rows = []
    for xcol in ["innovation_output_geomean", "maximal_lyapunov_mean"]:
        for ycol in [
            "tail_relative_step_delta_mean_behavior",
            "log10_final_separation",
            "hard_cycle_detected",
            "hard_unique_token_ratio",
        ]:
            xvalues = match[xcol].astype(float)
            yvalues = match[ycol].astype(float)
            correlation_rows.append(
                {
                    "x": xcol,
                    "y": ycol,
                    "pearson": xvalues.corr(yvalues, method="pearson"),
                    # Spearman is Pearson correlation of average ranks.  Compute
                    # it directly so the analysis does not require SciPy.
                    "spearman": xvalues.rank(method="average").corr(
                        yvalues.rank(method="average"), method="pearson"
                    ),
                    "n": len(match),
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    processed = root / "main_processed"
    figures = root / "main_figures"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    behavior.to_csv(processed / "main_behavior_anchor_metrics.csv", index=False)
    tangent.to_csv(processed / "main_tangent_anchor_metrics.csv", index=False)
    merged.to_csv(processed / "main_checkpoint_summary.csv", index=False)
    correlations.to_csv(processed / "main_cross_metric_correlations.csv", index=False)
    (processed / "main_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    steps = merged.training_step.to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    ax = axes[0, 0]
    for checkpoint in CHECKPOINTS:
        part = tangent[tangent.checkpoint == checkpoint]
        x = np.full(len(part), STEPS[checkpoint], dtype=float)
        ax.scatter(x, part.maximal_lyapunov_mean, color=COLORS[checkpoint], alpha=0.65, s=25)
    ax.plot(steps, merged.lyapunov_mean, color="black", marker="D", label="checkpoint mean")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    format_training_axis(ax)
    ax.set_ylabel("Maximal Lyapunov / step")
    ax.set_title("Sample-level Lyapunov across training")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    for checkpoint in CHECKPOINTS:
        part = tangent[tangent.checkpoint == checkpoint]
        ax.scatter(
            np.full(len(part), STEPS[checkpoint], dtype=float),
            part.innovation_output_geomean,
            color=COLORS[checkpoint],
            alpha=0.65,
            s=25,
        )
    ax.plot(steps, merged.innovation_output_mean, color="black", marker="D")
    format_training_axis(ax)
    ax.set_ylabel(r"Innovation $\|J_{new}\|_F/\sqrt{H}$")
    ax.set_title("Innovation sensitivity")

    ax = axes[1, 0]
    ax.plot(steps, merged.hard_cycle_fraction, color="#984ea3", marker="o")
    format_training_axis(ax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Hard full-window cycle fraction")
    ax.set_xlabel("Training step")
    ax.set_title("Exact cycles within 512 hard steps")

    ax = axes[1, 1]
    ax.plot(steps, merged.entropy_mean, color="#377eb8", marker="o", label="entropy")
    ax2 = ax.twinx()
    ax2.plot(steps, merged.top1_probability_mean, color="#e68613", marker="^", label="top-1 p")
    format_training_axis(ax)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Soft entropy", color="#377eb8")
    ax2.set_ylabel("Top-1 probability", color="#e68613")
    ax.set_title("Soft distribution sharpness")
    overview_path = figures / "main_training_dynamics_overview.png"
    save(fig, overview_path)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    shift = float(tangent.shift_normalized_frobenius.iloc[0])
    for checkpoint in CHECKPOINTS:
        part = tangent[tangent.checkpoint == checkpoint]
        axes[0].scatter(
            part.maximal_lyapunov_mean,
            part.total_geomean,
            color=COLORS[checkpoint],
            label=checkpoint,
            alpha=0.75,
        )
        axes[1].scatter(
            part.innovation_output_geomean,
            part.maximal_lyapunov_mean,
            color=COLORS[checkpoint],
            label=checkpoint,
            alpha=0.75,
        )
    axes[0].axhline(shift, color="#555555", linestyle="--", label="shift-only")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Maximal Lyapunov / step")
    axes[0].set_ylabel(r"Total $\|J\|_F/\sqrt{LH}$")
    axes[0].set_title("Total Frobenius is shift-dominated")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel(r"Innovation $\|J_{new}\|_F/\sqrt{H}$")
    axes[1].set_ylabel("Maximal Lyapunov / step")
    axes[1].set_title("Innovation versus long-horizon expansion")
    axes[1].legend(fontsize=8)
    tangent_path = figures / "main_tangent_decomposition.png"
    save(fig, tangent_path)

    manifest = {
        "experiment": "pythia_rolling_next_token_main",
        "validation": validation,
        "raw_behavior_files": [str(p.resolve()) for p in behavior_paths],
        "raw_tangent_files": [str(p.resolve()) for p in tangent_paths],
        "processed_dir": str(processed.resolve()),
        "figures": [
            {
                "path": str(overview_path.resolve()),
                "question": "How do rolling dynamics and hard cycles vary across training checkpoints?",
                "caveat": "Four checkpoints; behavior n=32 and tangent n=8 per checkpoint.",
            },
            {
                "path": str(tangent_path.resolve()),
                "question": "Does shift-dominated total Frobenius agree with innovation and Lyapunov?",
                "caveat": "Frobenius is a Hutchinson estimate; Lyapunov uses two Benettin probes.",
            },
        ],
    }
    manifest_path = figures / "main_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("VALIDATION", json.dumps(validation, sort_keys=True))
    print("CHECKPOINT_SUMMARY")
    print(merged.to_string(index=False))
    print("CORRELATIONS")
    print(correlations.to_string(index=False))
    print("OUTPUTS", processed, figures)


if __name__ == "__main__":
    main()
