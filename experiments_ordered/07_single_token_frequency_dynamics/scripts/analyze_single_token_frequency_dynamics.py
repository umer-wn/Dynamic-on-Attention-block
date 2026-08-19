#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_io import read_jsonl

from src.checkpoint_utils import checkpoint_step

from scripts._bootstrap import require_packages

require_packages(["matplotlib", "numpy"])

import matplotlib.pyplot as plt
import numpy as np

from src.single_token_dynamics import projected_poincare_points


COLORS = {
    "isolated_token": "#7A5195",
    "frozen_context": "#2A9D8F",
    "dynamic_context": "#E76F51",
}
MARKERS = {"isolated_token": "o", "frozen_context": "s", "dynamic_context": "^"}


def grouped(rows: list[dict]) -> dict[tuple, list[dict]]:
    output: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["checkpoint"], row["group"], row["token_id"], row["context_id"])
        output[key].append(row)
    for values in output.values():
        values.sort(key=lambda row: row["step"])
    return output


def save_manifest(path: Path, title: str, sources: list[Path], question: str, caveat: str) -> None:
    payload = {
        "title": title,
        "sources": [str(source.resolve()) for source in sources],
        "question": question,
        "allowed_interpretation": "Describe trajectory geometry or the three registered convergence diagnostics.",
        "caveat": caveat,
        "current_evidence": True,
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(
        f"# {title}\n\n- 数据源：" + ", ".join(f"`{source.resolve()}`" for source in sources)
        + f"\n- 回答问题：{question}\n- 允许解释：{payload['allowed_interpretation']}"
        + f"\n- Caveat：{caveat}\n", encoding="utf-8"
    )


def convergence_figure(summary: list[dict], trajectories: list[dict], output: Path, sources: list[Path]) -> None:
    trajectory_groups = grouped(trajectories)
    checkpoints = sorted({row["checkpoint"] for row in summary}, key=checkpoint_number)
    figure, axes = plt.subplots(len(checkpoints), 3, figsize=(14, 4.2 * len(checkpoints)), squeeze=False)
    for row_index, checkpoint in enumerate(checkpoints):
        candidates = [row for row in summary if row["checkpoint"] == checkpoint and row["context_id"] in (-1, 0)]
        for item in candidates:
            key = (checkpoint, item["group"], item["token_id"], item["context_id"])
            trace = trajectory_groups[key]
            steps = [row["step"] for row in trace]
            label = f"{item['group']} / bin{item['frequency_bin']} / {item['decoded']!r}"
            color = COLORS[item["group"]]
            axes[row_index, 0].plot(steps, [max(row["relative_step_delta"], 1e-12) for row in trace], color=color, alpha=.35)
            axes[row_index, 1].plot(steps, [max(row["nearby_distance"], 1e-12) for row in trace], color=color, alpha=.35)
        for group_index, group in enumerate(COLORS):
            values = [row["lyapunov_mean"] for row in candidates if row["group"] == group and row["lyapunov_mean"] is not None]
            x = np.full(len(values), group_index, dtype=float) + np.linspace(-.08, .08, max(len(values), 1))
            axes[row_index, 2].scatter(x, values, color=COLORS[group], alpha=.75, label=group if row_index == 0 else None)
        axes[row_index, 0].set_yscale("log")
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 2].axhline(0, color="black", lw=1, ls="--")
        axes[row_index, 2].set_xticks(range(3), list(COLORS), rotation=15)
        axes[row_index, 0].set_ylabel(f"{checkpoint}\nrelative step delta")
        axes[row_index, 1].set_ylabel("nearby distance")
        axes[row_index, 2].set_ylabel("Benettin Lyapunov / step")
        for axis in axes[row_index]:
            axis.grid(alpha=.2)
            axis.set_xlabel("dynamics step" if axis is not axes[row_index, 2] else "group")
    figure.suptitle("Registered convergence diagnostics (Poincaré excluded from labeling)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    save_manifest(output, "Convergence diagnostics", sources,
                  "Do relative motion, finite nearby separation, and tangent growth agree?",
                  "A label is unresolved when signs conflict or nearby distance reaches the float32 floor.")


def checkpoint_transition_figure(
    summary: list[dict], jacobians: list[dict], output: Path, sources: list[Path]
) -> None:
    checkpoints = sorted({row["checkpoint"] for row in summary}, key=checkpoint_number)
    steps = [checkpoint_step(value) for value in checkpoints]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for group in COLORS:
        lyapunov_medians = []
        relative_medians = []
        frobenius_medians = []
        for checkpoint, training_step in zip(checkpoints, steps):
            selected = [row for row in summary if row["checkpoint"] == checkpoint and row["group"] == group]
            lyapunov_values = [row["lyapunov_mean"] for row in selected if row["lyapunov_mean"] is not None]
            relative_values = [row["tail_relative_step_delta_mean"] for row in selected]
            exact_values = [
                row["normalized_frobenius"] for row in jacobians
                if row["checkpoint"] == checkpoint and row["group"] == group and int(row["trajectory_step"]) == 767
            ]
            axes[0].scatter([training_step] * len(lyapunov_values), lyapunov_values,
                            color=COLORS[group], marker=MARKERS[group], alpha=.35, s=22)
            axes[1].scatter([training_step] * len(relative_values), relative_values,
                            color=COLORS[group], marker=MARKERS[group], alpha=.25, s=18)
            axes[2].scatter([training_step] * len(exact_values), exact_values,
                            color=COLORS[group], marker=MARKERS[group], alpha=.35, s=22)
            lyapunov_medians.append(float(np.median(lyapunov_values)))
            relative_medians.append(float(np.median(relative_values)))
            frobenius_medians.append(float(np.median(exact_values)))
        axes[0].plot(steps, lyapunov_medians, color=COLORS[group], marker=MARKERS[group], label=group)
        axes[1].plot(steps, relative_medians, color=COLORS[group], marker=MARKERS[group])
        axes[2].plot(steps, frobenius_medians, color=COLORS[group], marker=MARKERS[group])
    for axis in axes:
        axis.set_xscale("symlog", linthresh=100)
        axis.set_xticks(steps, [str(value) for value in steps], rotation=20)
        axis.grid(alpha=.2)
        axis.set_xlabel("Pythia training step")
    axes[0].axhline(0, color="black", ls="--", lw=1)
    axes[0].set_ylabel("Benettin Lyapunov / dynamics step")
    axes[0].set_title("Tangent expansion / contraction")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("tail relative step delta")
    axes[1].set_title("Residual target-state motion")
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set_ylabel("exact normalized Frobenius at t=767")
    axes[2].set_title("Local RMS Jacobian gain")
    figure.suptitle("Single-token dynamics across training checkpoints")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=.2)
    plt.close(figure)
    save_manifest(output, "Checkpoint transition summary", sources,
                  "How do tangent growth, residual motion, and local RMS Jacobian gain evolve during training?",
                  "Lines connect checkpoint medians, while translucent points are token/context samples. Nearby numerical-floor status is not shown and must be read from the convergence audit.")


def geometry_figures(summary: list[dict], trajectories: list[dict], output_dir: Path, sources: list[Path]) -> None:
    traces = grouped(trajectories)
    representatives = [row for row in summary if row["context_id"] in (-1, 0)]
    # One deterministic representative per checkpoint/group/frequency bin.
    selected = {}
    for row in representatives:
        selected.setdefault((row["checkpoint"], row["group"], row["frequency_bin"]), row)
    for (checkpoint, group, frequency_bin), item in selected.items():
        key = (checkpoint, group, item["token_id"], item["context_id"])
        trace = traces[key]
        eval_trace = [
            row for row in trace
            if int(item["eval_start"]) <= int(row["step"]) < int(item["steps"])
        ]
        steps = np.array([row["step"] for row in trace])
        z = [np.array([row[f"projection_{index}"] for row in trace]) for index in range(4)]
        crossings = projected_poincare_points(eval_trace, 4)
        stem = f"{checkpoint}__{group}__bin{frequency_bin}__token{item['token_id']}"

        figure = plt.figure(figsize=(12, 5))
        ax2 = figure.add_subplot(1, 2, 1)
        points = ax2.scatter(z[0], z[1], c=steps, s=8, cmap="viridis")
        ax2.scatter(z[0][0], z[1][0], marker="o", color="black", label="start")
        ax2.scatter(z[0][-1], z[1][-1], marker="X", color="red", label="end")
        ax2.set(xlabel="z0", ylabel="z1", title="2D fixed projection trajectory")
        ax2.legend()
        figure.colorbar(points, ax=ax2, label="dynamics step")
        ax3 = figure.add_subplot(1, 2, 2, projection="3d")
        ax3.plot(z[0], z[1], z[2], color=COLORS[group], lw=.7, alpha=.8)
        ax3.scatter(z[0][0], z[1][0], z[2][0], color="black", marker="o")
        ax3.scatter(z[0][-1], z[1][-1], z[2][-1], color="red", marker="X")
        ax3.set(xlabel="z0", ylabel="z1", zlabel="z2", title="3D fixed projection trajectory")
        figure.suptitle(f"{checkpoint} | {group} | frequency bin {frequency_bin} | token {item['decoded']!r}")
        figure.tight_layout()
        path = output_dir / f"trajectory__{stem}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        save_manifest(path, "2D/3D fixed-projection trajectory", sources,
                      "How does one target-token trajectory move in a shared random projection bank?",
                      "Projection overlap or apparent convergence is not a convergence proof.")

        figure = plt.figure(figsize=(12, 5))
        p2 = figure.add_subplot(1, 2, 1)
        p3 = figure.add_subplot(1, 2, 2, projection="3d")
        if crossings:
            crossing_steps = np.array([point["crossing_step"] for point in crossings])
            crossing_coordinates = np.array(
                [[point["z1"], point["z2"], point["z3"]] for point in crossings], dtype=float
            )
            centered = crossing_coordinates - crossing_coordinates.mean(axis=0, keepdims=True)
            scatter = p2.scatter(centered[:, 0], centered[:, 1],
                                 c=crossing_steps, cmap="plasma", s=28)
            figure.colorbar(scatter, ax=p2, label="interpolated crossing step")
            p3.scatter(centered[:, 0], centered[:, 1], centered[:, 2],
                       c=crossing_steps, cmap="plasma", s=28)
        else:
            p2.text(.5, .5, "No upward crossing", transform=p2.transAxes, ha="center")
            p3.text2D(.5, .5, "No upward crossing", transform=p3.transAxes, ha="center")
        p2.set(xlabel="Δz1 (centered)", ylabel="Δz2 (centered)", title="2D Projected Poincaré Section")
        p3.set(xlabel="Δz1", ylabel="Δz2", zlabel="Δz3", title="3D Projected Poincaré Section")
        figure.suptitle(f"median(z0), upward crossing | centered for display | crossings={len(crossings)}")
        figure.tight_layout()
        path = output_dir / f"poincare__{stem}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        save_manifest(path, "2D/3D Projected Poincaré Section", sources,
                      "At upward median-z0 crossings, do projected points form a point, cluster, or band?",
                      "Per-trajectory projected section with linear interpolation; crossing coordinates are mean-centered only for display. It does not determine convergence or prove periodicity.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary_paths = sorted(root.glob("**/raw/*__summary.jsonl"))
    trajectory_paths = sorted(root.glob("**/raw/*__trajectory.jsonl"))
    jacobian_paths = sorted(root.glob("**/raw/*__jacobians.jsonl"))
    if not summary_paths or not trajectory_paths:
        raise RuntimeError("no completed checkpoint outputs found")
    summary = [row for path in summary_paths for row in read_jsonl(path)]
    trajectories = [row for path in trajectory_paths for row in read_jsonl(path)]
    jacobians = [row for path in jacobian_paths for row in read_jsonl(path)]
    fieldnames = sorted({key for row in summary for key in row})
    with (output / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    convergence_figure(summary, trajectories, output / "convergence_diagnostics.png", summary_paths + trajectory_paths)
    checkpoint_transition_figure(
        summary, jacobians, output / "checkpoint_transition_summary.png", summary_paths + jacobian_paths
    )
    geometry_figures(summary, trajectories, output, summary_paths + trajectory_paths)
    print(json.dumps({"summary_rows": len(summary), "trajectory_rows": len(trajectories),
                      "figures": len(list(output.glob("*.png")))}, indent=2))


if __name__ == "__main__":
    main()
