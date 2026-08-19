#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_io import atomic_json
from src.experiment_plotting import save_figure


DEFAULT_ROOT = "/home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan"
DEFAULT_REPORT = "experiments_ordered/10_validation_corpus_loss_rescan/reports/validation_corpus_loss_rescan_report.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()
    root = Path(args.root)
    processed = root / "processed"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    loss = pd.read_csv(processed / "checkpoint_loss_by_source.csv")
    delta_path = processed / "adjacent_loss_deltas_by_source.csv"
    try:
        deltas = pd.read_csv(delta_path) if delta_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        deltas = pd.DataFrame()
    made = []
    fig, ax = plt.subplots(figsize=(12, 6))
    for source_id, group in loss.groupby("source_id"):
        group = group.sort_values("training_step")
        ax.plot(group.training_step, group.token_weighted_loss, marker="o", linewidth=1.5, label=source_id)
    ax.set(xlabel="training step", ylabel="token-weighted loss", title="Validation/test corpus loss by source")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    made.append(str(save_figure(fig, figures / "checkpoint_loss_by_source.png")))
    plt.close(fig)
    if not deltas.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        for source_id, group in deltas.groupby("source_id"):
            centers = (group.step_a + group.step_b) / 2
            ax.plot(centers, group.delta_loss, marker="o", linewidth=1, label=source_id)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="adjacent checkpoint midpoint", ylabel="delta loss", title="Adjacent loss changes by source")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        made.append(str(save_figure(fig, figures / "loss_delta_by_source.png")))
        plt.close(fig)
    merged_path = processed / "loss_vs_frobenius_merged.csv"
    if merged_path.exists():
        merged = pd.read_csv(merged_path)
        tail = merged[merged.condition == "tail_t767"] if "condition" in merged else pd.DataFrame()
        if not tail.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            for source_id, group in tail.groupby("source_id"):
                ax.scatter(group.token_weighted_loss, group["median"], s=35, alpha=0.75, label=source_id)
            ax.set(xlabel="token-weighted loss", ylabel="tail_t767 median normalized Frobenius", title="Loss versus tail Frobenius")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            made.append(str(save_figure(fig, figures / "loss_vs_tail_frobenius_by_source.png")))
            plt.close(fig)
        controls = merged[merged.condition.isin(["self_t0", "common_step1000_state"])] if "condition" in merged else pd.DataFrame()
        if not controls.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
            for ax, condition in zip(axes, ["self_t0", "common_step1000_state"]):
                subset = controls[controls.condition == condition]
                for source_id, group in subset.groupby("source_id"):
                    ax.scatter(group.token_weighted_loss, group["median"], s=28, alpha=0.7, label=source_id)
                ax.set(xlabel="token-weighted loss", ylabel="median normalized Frobenius", title=condition)
            axes[0].legend(frameon=False, fontsize=7)
            fig.tight_layout()
            made.append(str(save_figure(fig, figures / "loss_vs_self_common_controls.png")))
            plt.close(fig)
    rebound_path = processed / "step16000_vs_step100000_by_source.csv"
    try:
        rebound = pd.read_csv(rebound_path).to_dict("records") if rebound_path.exists() else []
    except pd.errors.EmptyDataError:
        rebound = []
    atomic_json(root / "manifests" / "figure_manifest.json", {"figures": made, "figure_count": len(made)})
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# The Pile / Paloma Loss Re-evaluation 报告\n\n"
        f"状态：`{'complete' if made else 'no_figures'}`\n\n"
        f"数据根：`{root}`\n\n"
        f"图表数：{len(made)}\n\n"
        "## step16000 vs step100000\n\n"
        + (pd.DataFrame(rebound).to_csv(index=False) if rebound else "暂无 rebound summary。\n"),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "figures": made, "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
