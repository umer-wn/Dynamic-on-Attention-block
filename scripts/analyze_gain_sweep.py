#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def load(pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.glob(pattern)):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


main = load("pythia_gain_beta*__dynamical_edge.jsonl")
product = load("pythia_gain_beta*__product_jacobian_metrics.jsonl")
distance = load("pythia_gain_beta*__state_distance_metrics.jsonl")
main["final_step_delta"] = main["step_deltas"].map(lambda xs: xs[-1] if xs else float("nan"))

summary = (
    main.groupby("output_scale", as_index=False)
    .agg(
        samples=("sample_index", "count"),
        frobenius_mean=("normalized_frobenius_geomean", "mean"),
        frobenius_std=("normalized_frobenius_geomean", "std"),
        edge_distance_mean=("edge_distance_log", "mean"),
        nearby_log_growth_mean=("nearby_log_growth_per_step", "mean"),
        final_step_delta_mean=("final_step_delta", "mean"),
        diverged_fraction=("diverged", "mean"),
        collapsed_fraction=("collapsed", "mean"),
    )
)

phase = pd.crosstab(main["output_scale"], main["phase_label"]).reset_index()
product_summary = (
    product.groupby(["output_scale", "product_window"], as_index=False)
    .agg(product_log_gain_mean=("product_log_gain_mean", "mean"))
)
lag_summary = (
    distance.groupby(["output_scale", "lag_window"], as_index=False)
    .agg(lag_distance_mean=("lag_distance_mean", "mean"))
)

out = ROOT / "results" / "processed"
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / "pythia_operator_gain_sweep__summary.csv", index=False)
phase.to_csv(out / "pythia_operator_gain_sweep__phase_counts.csv", index=False)
product_summary.to_csv(out / "pythia_operator_gain_sweep__product_gain.csv", index=False)
lag_summary.to_csv(out / "pythia_operator_gain_sweep__lag_distance.csv", index=False)

print("SUMMARY")
print(summary.to_string(index=False))
print("PHASE")
print(phase.to_string(index=False))
print("PRODUCT")
print(product_summary.to_string(index=False))
