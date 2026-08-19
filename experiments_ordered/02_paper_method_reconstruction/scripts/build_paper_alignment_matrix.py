#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_ORDER = [
    "EleutherAI/pythia-70m",
    "gpt2",
    "Qwen/Qwen2.5-0.5B",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latest_dynamical_rows(root: Path) -> pd.DataFrame:
    paths = list(root.glob("experiments/*/results/processed/*__dynamical_edge_summary.csv"))
    paths.extend(root.glob("results/processed/*__dynamical_edge_summary.csv"))
    frames = []
    for path in paths:
        df = _read_csv(path)
        if df.empty:
            continue
        df["source_summary"] = str(path.relative_to(root))
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    rows = pd.concat(frames, ignore_index=True)
    preferred = rows[rows["experiment"].str.contains("normal|128", case=False, na=False)].copy()
    if preferred.empty:
        preferred = rows.copy()
    preferred["rank"] = preferred["experiment"].map(
        lambda x: -1 if "long" in str(x) else 0 if "normal" in str(x) else 1 if "128" in str(x) else 2
    )
    preferred = preferred.sort_values(["model", "sequence_length", "rank", "samples"], ascending=[True, False, True, False])
    return preferred.groupby("model", as_index=False, dropna=False).head(1)


def _dynamical_extras(root: Path) -> pd.DataFrame:
    rows = _latest_dynamical_rows(root)
    if rows.empty:
        return rows
    out = rows[
        [
            "model",
            "experiment",
            "sequence_length",
            "samples",
            "mean_normalized_frobenius",
            "median_normalized_frobenius",
            "mean_edge_distance_log",
            "diverged_fraction",
            "collapsed_fraction",
            "source_summary",
        ]
    ].copy()

    nearby_rows = []
    product_rows = []
    trajectory_rows = []
    for _, row in out.iterrows():
        exp = row["experiment"]
        source = root / row["source_summary"]
        processed = source.parent
        distance = _read_csv(processed / f"{exp}__state_distance_metrics.csv")
        product = _read_csv(processed / f"{exp}__product_jacobian_metrics.csv")
        trajectory = _read_csv(processed / f"{exp}__trajectory_summary.csv")
        if not distance.empty and "nearby_log_growth_per_step" in distance.columns:
            nearby_rows.append(
                {
                    "model": row["model"],
                    "nearby_growth_ratio_mean": distance.groupby("sample_index")["nearby_growth_ratio"].first().mean(),
                    "nearby_log_growth_per_step_mean": distance.groupby("sample_index")[
                        "nearby_log_growth_per_step"
                    ].first().mean(),
                    "lag_distance_window_max_mean": distance.loc[
                        distance["lag_window"] == distance["lag_window"].max(), "lag_distance_mean"
                    ].mean(),
                }
            )
        if not product.empty and "product_log_gain_max" in product.columns:
            product_rows.append(
                {
                    "model": row["model"],
                    "product_log_gain_max_mean": product["product_log_gain_max"].mean(),
                    "product_gain_max_mean": product["product_gain_max"].mean(),
                    "product_windows": ",".join(str(int(x)) for x in sorted(product["product_window"].dropna().unique())),
                }
            )
        if not trajectory.empty and "step_delta" in trajectory.columns:
            final = trajectory.sort_values("step_index").groupby("sample_index", dropna=False).tail(1)
            settled_mask = final["step_delta"] <= 1.0e-2
            trajectory_rows.append(
                {
                    "model": row["model"],
                    "final_step_delta_mean": final["step_delta"].mean(),
                    "final_step_delta_max": final["step_delta"].max(),
                    "settled_samples": int(settled_mask.sum()),
                    "settled_sample_fraction": float(settled_mask.mean()),
                    "asymptotic_status": "settled" if final["step_delta"].max() <= 1.0e-2 else "not settled",
                }
            )
    nearby = pd.DataFrame(nearby_rows)
    product = pd.DataFrame(product_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    if not nearby.empty:
        out = out.merge(nearby, on="model", how="left")
    if not product.empty:
        out = out.merge(product, on="model", how="left")
    if not trajectory.empty:
        out = out.merge(trajectory, on="model", how="left")
    return out


def _spectrum_rows(root: Path) -> pd.DataFrame:
    path = root / "results/processed/cross_model_spectrum_128__anisotropy_summary.csv"
    df = _read_csv(path)
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby("model", dropna=False)
        .agg(
            spectrum_layers=("layer", "count"),
            spectrum_seq_len=("sequence_length", "max"),
            spectrum_loss=("loss", "mean"),
            spectrum_sigma_max_lanczos_max=("sigma_max_lanczos", "max"),
            spectrum_sigma_max_lanczos_median=("sigma_max_lanczos", "median"),
            spectrum_geomean_topk_median=("geometric_mean_sigma_topk", "median"),
            spectrum_powerlaw_slope_mean=("powerlaw_slope_top", "mean"),
            spectrum_powerlaw_r2_mean=("powerlaw_r2_top", "mean"),
        )
        .reset_index()
    )
    grouped["spectrum_source"] = str(path.relative_to(root))
    return grouped


def _phase_label(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "missing"
    if x < 0.9:
        return "stable/contractive"
    if x <= 1.1:
        return "near edge"
    return "chaotic/expansive"


def build_matrix(root: Path) -> pd.DataFrame:
    dynamics = _dynamical_extras(root)
    spectrum = _spectrum_rows(root)
    matrix = pd.DataFrame({"model": MODEL_ORDER})
    if not dynamics.empty:
        matrix = matrix.merge(dynamics, on="model", how="left")
    if not spectrum.empty:
        matrix = matrix.merge(spectrum, on="model", how="left")
    if "mean_normalized_frobenius" in matrix.columns:
        matrix["paper_phase_by_normalized_frobenius"] = matrix["mean_normalized_frobenius"].map(_phase_label)
    else:
        matrix["paper_phase_by_normalized_frobenius"] = "missing"
    matrix["paper_comparable_status"] = matrix["mean_normalized_frobenius"].map(
        lambda x: "direct dynamics result" if pd.notna(x) else "missing direct dynamics result"
    )
    if "asymptotic_status" in matrix.columns:
        matrix.loc[matrix["asymptotic_status"].eq("not settled"), "paper_comparable_status"] = (
            "direct dynamics result, but not asymptotically settled"
        )
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/processed/paper_alignment_matrix.csv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    matrix = build_matrix(root)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
