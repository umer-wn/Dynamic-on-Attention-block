#!/usr/bin/env python
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "processed/checkpoint_three_corpus_loss.csv"
OUTPUT = ROOT / "figures/three_corpus_loss_linear_x_linear_y.png"
BACKUP = ROOT / "figures/three_corpus_loss_linear_x_linear_y_full_range_backup.png"


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        {**row, "training_step": int(row["training_step"])}
        for row in rows
        if 1000 <= int(row["training_step"]) <= 61000
    ]
    rows.sort(key=lambda row: row["training_step"])
    if not rows or rows[0]["training_step"] != 1000 or rows[-1]["training_step"] != 61000:
        raise RuntimeError(
            f"unexpected selected range: {rows[0]['training_step'] if rows else None}.."
            f"{rows[-1]['training_step'] if rows else None}"
        )
    if OUTPUT.exists() and not BACKUP.exists():
        shutil.copy2(OUTPUT, BACKUP)

    specs = [
        ("train_loss", "The Pile train proxy", "#2878b5"),
        ("test_loss", "The Pile test", "#9c3f35"),
        ("hard_natural_language_loss", "local OpenWebMath hard set", "#6b4c9a"),
    ]
    x = [row["training_step"] for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 6.5), constrained_layout=True)
    for column, label, color in specs:
        ax.plot(
            x,
            [float(row[column]) for row in rows],
            color=color,
            lw=2,
            marker="o",
            ms=4,
            label=label,
        )
    ax.set_xlim(1000, 61000)
    ax.set_xticks([1000, 10000, 20000, 30000, 40000, 50000, 61000])
    ax.set_xlabel("Training checkpoint step")
    ax.set_ylabel("Token-weighted loss")
    ax.set_title("Three-corpus loss, checkpoints 1000-61000")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print({"selected_rows": len(rows), "first_step": x[0], "last_step": x[-1], "output": str(OUTPUT)})


if __name__ == "__main__":
    main()
