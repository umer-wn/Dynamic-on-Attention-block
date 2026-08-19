#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "processed/checkpoint_metric_summary.csv"
FIGURES = ROOT / "figures"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: int(row["training_step"]))


def main() -> None:
    rows = read_rows(SUMMARY)
    x = np.asarray([int(row["training_step"]) for row in rows], dtype=np.float64)
    definitions = [
        ("spectral_radius", "Spectral radius at dynamic step 1024", "rho(J)", 1.0),
        ("lyapunov_exponent_last_256", "Lyapunov exponent over dynamic steps 768-1024", "lambda per step", 0.0),
        ("normalized_frobenius_norm", "Normalized Frobenius norm at dynamic step 1024", "||J||F / sqrt(512)", 1.0),
    ]
    colors = ["#2F6F9F", "#C44E52", "#3B8C6E"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    for ax, (metric, title, ylabel, reference), color in zip(axes, definitions, colors):
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
        low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in rows])
        high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in rows])
        ax.fill_between(x, low, high, color=color, alpha=0.18, label="95% CI over 100 tokens")
        ax.plot(x, mean, color=color, marker="o", ms=4.5, lw=1.7, label="mean over 100 tokens")
        ax.axhline(reference, color="black", ls="--", lw=0.9, alpha=0.65)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
        ax.legend(loc="best", frameon=False)
    axes[-1].set_xlabel("Pythia-70M training checkpoint step")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(int(value)) for value in x], rotation=45, ha="right")
    fig.suptitle(
        "100-token endpoint dynamics across all visualization checkpoints\n"
        "paired WikiText-2 frequency-decile cohort; means with 95% confidence intervals",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "spectral_radius_lyapunov_normalized_frobenius_by_checkpoint.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for (metric, title, ylabel, reference), color in zip(definitions, colors):
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
        low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in rows])
        high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in rows])
        fig, ax = plt.subplots(figsize=(13, 5.5))
        ax.fill_between(x, low, high, color=color, alpha=0.18)
        ax.plot(x, mean, color=color, marker="o", lw=1.7)
        ax.axhline(reference, color="black", ls="--", lw=0.9, alpha=0.65)
        ax.set(title=title, xlabel="Pythia-70M training checkpoint step", ylabel=ylabel)
        ax.set_xticks(x, [str(int(value)) for value in x], rotation=45, ha="right")
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{metric}_by_checkpoint.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
