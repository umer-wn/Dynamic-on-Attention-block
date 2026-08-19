#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "processed/checkpoint_metric_summary.csv"
OUTPUT = ROOT / "REPORT_ALL19.md"


def main() -> None:
    with SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["training_step"]))
    lines = [
        "# 100-token endpoint metrics across 19 Pythia-70M checkpoints",
        "",
        "## Protocol",
        "",
        "- The same 100 unique WikiText-2 train token types are paired across every checkpoint; 10 tokens are sampled from each frequency decile with seed `190905176`.",
        "- Each token starts from the checkpoint-specific input embedding and is propagated through the frozen isolated-token map for 1024 dynamic steps.",
        "- At `x_1024`, an exact `512 x 512` Jacobian is constructed.",
        "- Spectral radius: `rho(J) = max_i |lambda_i(J)|`.",
        "- Normalized Frobenius: `||J||_F / sqrt(512)`.",
        "- Lyapunov exponent: exact JVP/Benettin tangent propagation with per-step renormalization; the reported value is the mean log growth over dynamic steps 768-1024. The 0-1024 value is retained in token-level CSV files as an auxiliary column.",
        "- Every checkpoint value below is the arithmetic mean over 100 per-token scalar metrics. CSV also stores standard deviation, median, min/max, SEM and normal-approximation 95% CI.",
        "",
        "## Means over 100 paired tokens",
        "",
        "| checkpoint | spectral radius | Lyapunov (768-1024) | normalized Frobenius |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {float(row['spectral_radius_mean']):.6f} | "
            f"{float(row['lyapunov_exponent_last_256_mean']):+.6f} | "
            f"{float(row['normalized_frobenius_norm_mean']):.6f} |"
        )
    rho_max = max(rows, key=lambda row: float(row["spectral_radius_mean"]))
    lyap_max = max(rows, key=lambda row: float(row["lyapunov_exponent_last_256_mean"]))
    lyap_min = min(rows, key=lambda row: float(row["lyapunov_exponent_last_256_mean"]))
    frob_max = max(rows, key=lambda row: float(row["normalized_frobenius_norm_mean"]))
    frob_min = min(rows, key=lambda row: float(row["normalized_frobenius_norm_mean"]))
    lines += [
        "",
        "## Descriptive checks",
        "",
        f"- Highest mean spectral radius: `{rho_max['checkpoint']}` = `{float(rho_max['spectral_radius_mean']):.6f}`.",
        f"- Highest mean last-window Lyapunov: `{lyap_max['checkpoint']}` = `{float(lyap_max['lyapunov_exponent_last_256_mean']):+.6f}`; lowest: `{lyap_min['checkpoint']}` = `{float(lyap_min['lyapunov_exponent_last_256_mean']):+.6f}`.",
        f"- Highest normalized Frobenius mean: `{frob_max['checkpoint']}` = `{float(frob_max['normalized_frobenius_norm_mean']):.6f}`; lowest: `{frob_min['checkpoint']}` = `{float(frob_min['normalized_frobenius_norm_mean']):.6f}`.",
        "- These local endpoint/window metrics do not by themselves prove a global attractor or chaos classification.",
        "",
        "## Outputs",
        "",
        "- `processed/checkpoint_metric_summary.csv`: 19-checkpoint summary with distribution statistics and 95% CI.",
        "- `processed/checkpoint_parts/step*.csv`: 1900 token-level rows.",
        "- `figures/spectral_radius_lyapunov_normalized_frobenius_by_checkpoint.png`: combined three-panel chart.",
        "- `figures/spectral_radius_by_checkpoint.png`, `lyapunov_exponent_last_256_by_checkpoint.png`, `normalized_frobenius_norm_by_checkpoint.png`: individual charts.",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
