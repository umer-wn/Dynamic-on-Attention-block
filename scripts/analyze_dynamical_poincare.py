#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import require_packages

require_packages(["matplotlib", "numpy", "pandas", "yaml"])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.io_utils import load_config, sanitize_name


def _path_for(config: dict[str, Any], suffix: str, seq_len: int, model_name: str, revision: str) -> Path:
    raw_dir = Path(config.get("output_dir", "results")) / "raw"
    return raw_dir / f"{config['experiment_name']}__{sanitize_name(model_name)}__{revision}__seq{seq_len}__{suffix}.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _all_rows(config: dict[str, Any], suffix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        revisions = model_cfg.get("revisions") or [model_cfg.get("revision", "main")]
        for revision in revisions:
            for seq_len in config["dataset"].get("sequence_lengths", [128]):
                rows.extend(_load_jsonl(_path_for(config, suffix, int(seq_len), model_name, str(revision))))
    return rows


def _poincare_points(traj: pd.DataFrame) -> pd.DataFrame:
    points: list[dict[str, Any]] = []
    group_cols = ["model", "checkpoint", "sequence_length", "sample_index"]
    for keys, group in traj.sort_values("step_index").groupby(group_cols, dropna=False):
        median_norm = float(group["state_norm"].median())
        prev = None
        crossing_order = 0
        for _, row in group.iterrows():
            value = float(row["state_norm"]) - median_norm
            if prev is not None and prev["section_value"] <= 0.0 and value > 0.0:
                out = row.to_dict()
                out["section_value"] = value
                out["crossing_order"] = crossing_order
                points.append(out)
                crossing_order += 1
            prev = {"section_value": value}
    return pd.DataFrame(points)


def _return_map_points(traj: pd.DataFrame) -> pd.DataFrame:
    if "projection_value" not in traj.columns or "projection_next" not in traj.columns:
        return pd.DataFrame()
    rows = traj.dropna(subset=["projection_next"]).copy()
    return rows


def _pca_return_map_points(traj: pd.DataFrame) -> pd.DataFrame:
    if "projection_value" not in traj.columns:
        return pd.DataFrame()
    value_cols = ["state_norm", "step_delta", "nearby_distance"]
    if not all(col in traj.columns for col in value_cols):
        return pd.DataFrame()
    rows = []
    for keys, group in traj.sort_values("step_index").groupby(["model", "checkpoint", "sequence_length", "sample_index"], dropna=False):
        if len(group) < 3:
            continue
        x = group[value_cols].astype(float)
        centered = x - x.mean(axis=0)
        _, _, vt = np.linalg.svd(centered.to_numpy(), full_matrices=False)
        pc1 = centered.to_numpy() @ vt[0]
        out = group.copy()
        out["pca_summary_value"] = pc1
        out["pca_summary_next"] = pd.Series(pc1).shift(-1).to_numpy()
        rows.append(out.dropna(subset=["pca_summary_next"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _save_phase_plot(traj: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        traj["state_norm"],
        traj["step_delta"],
        c=traj["step_index"],
        s=20,
        alpha=0.75,
        cmap="viridis",
    )
    ax.set_xlabel("state norm")
    ax.set_ylabel("step delta")
    ax.set_title("Phase Projection")
    fig.colorbar(scatter, ax=ax, label="step")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_return_map_plot(points: pd.DataFrame, path: Path, x_col: str, y_col: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if points.empty:
        ax.text(0.5, 0.5, "No return-map points", ha="center", va="center")
        ax.set_axis_off()
    else:
        scatter = ax.scatter(
            points[x_col],
            points[y_col],
            c=points["step_index"],
            s=24,
            alpha=0.78,
            cmap="viridis",
        )
        ax.set_xlabel("z_t")
        ax.set_ylabel("z_{t+1}")
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, label="step")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_poincare_plot(points: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if points.empty:
        ax.text(0.5, 0.5, "No section crossings", ha="center", va="center")
        ax.set_axis_off()
    else:
        scatter = ax.scatter(
            points["step_delta"],
            points["nearby_distance"],
            c=points["crossing_order"],
            s=32,
            alpha=0.8,
            cmap="plasma",
        )
        ax.set_xlabel("step delta at crossing")
        ax.set_ylabel("nearby distance at crossing")
        ax.set_title("Approximate Poincare Section")
        fig.colorbar(scatter, ax=ax, label="crossing order")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_nearby_plot(traj: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for sample_index, group in traj.groupby("sample_index"):
        group = group.sort_values("step_index")
        ax.plot(group["step_index"], group["nearby_distance"], marker="o", linewidth=1.2, label=f"sample {sample_index}")
    ax.set_xlabel("step")
    ax.set_ylabel("nearby distance")
    ax.set_title("Nearby Trajectory Distance")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_lag_plot(distance: pd.DataFrame, path: Path) -> None:
    if distance.empty or "lag_window" not in distance.columns:
        return
    summary = distance.groupby("lag_window", dropna=False)["lag_distance_mean"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["lag_window"], summary["lag_distance_mean"], marker="o")
    ax.set_xlabel("lag window")
    ax.set_ylabel("mean lag distance")
    ax.set_title("Lagged State-Space Distance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_product_plot(product: pd.DataFrame, path: Path) -> None:
    if product.empty or "product_window" not in product.columns:
        return
    summary = product.groupby("product_window", dropna=False)["product_log_gain_max"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.plot(summary["product_window"], summary["product_log_gain_max"], marker="o")
    ax.set_xlabel("product window")
    ax.set_ylabel("mean max log gain per step")
    ax.set_title("Multi-Step Jacobian Product Gain")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_figure_note(
    figure_path: Path,
    title: str,
    source: str,
    x_axis: str,
    y_axis: str,
    explanation: str,
    caveat: str,
) -> dict[str, str]:
    note_path = figure_path.with_suffix(".md")
    text = "\n".join(
        [
            f"# {title}",
            "",
            f"- Figure: `{figure_path.name}`",
            f"- Source data: `{source}`",
            f"- X axis: {x_axis}",
            f"- Y axis: {y_axis}",
            f"- Meaning: {explanation}",
            f"- Caution: {caveat}",
            "",
        ]
    )
    note_path.write_text(text, encoding="utf-8")
    return {
        "figure": figure_path.name,
        "note": note_path.name,
        "title": title,
        "source": source,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "meaning": explanation,
        "caution": caveat,
    }


def _write_figure_manifest(experiment: str, figures_dir: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    manifest_csv = figures_dir / f"{experiment}__figure_manifest.csv"
    manifest_md = figures_dir / f"{experiment}__figure_manifest.md"
    pd.DataFrame(rows).to_csv(manifest_csv, index=False)
    lines = [f"# {experiment} Figure Manifest", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['title']}",
                "",
                f"- Figure: `{row['figure']}`",
                f"- Note: `{row['note']}`",
                f"- Source data: `{row['source']}`",
                f"- Axes: {row['x_axis']} / {row['y_axis']}",
                f"- Meaning: {row['meaning']}",
                f"- Caution: {row['caution']}",
                "",
            ]
        )
    manifest_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {manifest_csv}")
    print(f"wrote {manifest_md}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    experiment = config["experiment_name"]
    output_dir = Path(config.get("output_dir", "results"))
    processed_dir = output_dir / "processed"
    figures_dir = output_dir / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    traj = pd.DataFrame(_all_rows(config, "dynamics_trajectory"))
    if traj.empty:
        raise SystemExit("no trajectory rows found")
    distance = pd.DataFrame(_all_rows(config, "state_distance_metrics"))
    product = pd.DataFrame(_all_rows(config, "product_jacobian_metrics"))
    points = _poincare_points(traj)
    return_points = _return_map_points(traj)
    pca_return_points = _pca_return_map_points(traj)

    traj_path = processed_dir / f"{experiment}__trajectory_summary.csv"
    points_path = processed_dir / f"{experiment}__poincare_points.csv"
    return_path = processed_dir / f"{experiment}__return_map_points.csv"
    pca_return_path = processed_dir / f"{experiment}__pca_summary_return_map_points.csv"
    distance_path = processed_dir / f"{experiment}__state_distance_metrics.csv"
    product_path = processed_dir / f"{experiment}__product_jacobian_metrics.csv"
    traj.to_csv(traj_path, index=False)
    points.to_csv(points_path, index=False)
    return_points.to_csv(return_path, index=False)
    pca_return_points.to_csv(pca_return_path, index=False)
    if not distance.empty:
        distance.to_csv(distance_path, index=False)
    if not product.empty:
        product.to_csv(product_path, index=False)

    figure_notes: list[dict[str, str]] = []
    phase_path = figures_dir / f"{experiment}__phase_projection.png"
    poincare_path = figures_dir / f"{experiment}__poincare_section.png"
    return_map_path = figures_dir / f"{experiment}__return_map_projection.png"
    pca_return_path_fig = figures_dir / f"{experiment}__return_map_pca_summary.png"
    nearby_path = figures_dir / f"{experiment}__nearby_distance_by_step.png"
    lag_path = figures_dir / f"{experiment}__lag_distance_by_window.png"
    product_path_fig = figures_dir / f"{experiment}__product_log_gain_by_window.png"

    _save_phase_plot(traj, phase_path)
    figure_notes.append(
        _write_figure_note(
            phase_path,
            "Phase Projection",
            traj_path.name,
            "state norm",
            "step delta",
            "Shows the feedback trajectory in a two-scalar phase view; movement toward low step delta indicates convergence of the iterated hidden state.",
            "This is a diagnostic projection, not a full-dimensional phase portrait.",
        )
    )
    _save_poincare_plot(points, poincare_path)
    figure_notes.append(
        _write_figure_note(
            poincare_path,
            "Approximate Poincare Section",
            points_path.name,
            "step delta at crossing",
            "nearby distance at crossing",
            "Samples crossings of an empirical section defined by state norm passing above its per-sample median.",
            "The section is approximate and scalar-defined; it is useful for recurrence diagnostics but is not identical to a hand-selected low-dimensional map in the paper.",
        )
    )
    _save_return_map_plot(
        return_points,
        return_map_path,
        "projection_value",
        "projection_next",
        "Return Map: z_t vs z_{t+1}",
    )
    figure_notes.append(
        _write_figure_note(
            return_map_path,
            "Fixed-Projection Return Map",
            return_path.name,
            "z_t",
            "z_{t+1}",
            "Plots consecutive iterates after projecting every state in a trajectory onto one fixed direction, matching the paper-style return-map idea more closely than changing projections over time.",
            "It remains a one-dimensional projection of the full hidden state; conclusions depend on the chosen fixed projection direction.",
        )
    )
    _save_return_map_plot(
        pca_return_points,
        pca_return_path_fig,
        "pca_summary_value",
        "pca_summary_next",
        "Return Map: PCA-summary z_t vs z_{t+1}",
    )
    figure_notes.append(
        _write_figure_note(
            pca_return_path_fig,
            "PCA-Summary Return Map",
            pca_return_path.name,
            "PCA-summary z_t",
            "PCA-summary z_{t+1}",
            "Uses the first principal component of scalar trajectory summaries as a post-hoc return-map coordinate.",
            "The PCA direction is fitted after observing the trajectory, so it is exploratory rather than a fixed physical coordinate.",
        )
    )
    _save_nearby_plot(traj, nearby_path)
    figure_notes.append(
        _write_figure_note(
            nearby_path,
            "Nearby Trajectory Distance",
            traj_path.name,
            "step",
            "nearby distance",
            "Tracks whether a small perturbed trajectory separates from or contracts toward the reference trajectory over feedback iterations.",
            "Distances are measured under the configured mask and perturbation scale; they should be compared with those settings fixed.",
        )
    )
    _save_lag_plot(distance, lag_path)
    if lag_path.exists():
        figure_notes.append(
            _write_figure_note(
                lag_path,
                "Lagged State-Space Distance",
                distance_path.name,
                "lag window",
                "mean lag distance",
                "Measures how far states separated by a fixed number of feedback steps are in hidden-state space.",
                "Averaging hides per-sample variation; inspect the CSV when the curve is flat or noisy.",
            )
        )
    _save_product_plot(product, product_path_fig)
    if product_path_fig.exists():
        figure_notes.append(
            _write_figure_note(
                product_path_fig,
                "Multi-Step Jacobian Product Gain",
                product_path.name,
                "product window",
                "mean max log gain per step",
                "Estimates expansion or contraction under the product of Jacobians across multiple feedback steps; values below zero indicate average contraction.",
                "This is a stochastic probe estimate, so probe count and window count affect stability.",
            )
        )
    _write_figure_manifest(experiment, figures_dir, figure_notes)

    print(f"wrote {traj_path}")
    print(f"wrote {points_path}")
    print(f"wrote {return_path}")
    print(f"wrote {pca_return_path}")
    if not distance.empty:
        print(f"wrote {distance_path}")
    if not product.empty:
        print(f"wrote {product_path}")


if __name__ == "__main__":
    main()
