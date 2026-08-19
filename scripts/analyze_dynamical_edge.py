#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import require_packages

require_packages(["pandas", "yaml"])

import pandas as pd

from src.io_utils import load_config, sanitize_name


def _raw_paths(config: dict[str, Any]) -> list[Path]:
    raw_dir = Path(config.get("output_dir", "results")) / "raw"
    paths: list[Path] = []
    for model_cfg in config["models"]:
        name = model_cfg["name"]
        revisions = model_cfg.get("revisions") or [model_cfg.get("revision", "main")]
        for revision in revisions:
            for seq_len in config["dataset"].get("sequence_lengths", [128]):
                paths.append(
                    raw_dir
                    / f"{config['experiment_name']}__{sanitize_name(name)}__{revision}__seq{seq_len}__dynamical_edge.jsonl"
                )
    return paths


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    rows: list[dict[str, Any]] = []
    for path in _raw_paths(config):
        if not path.exists():
            print(f"missing raw file: {path}")
            continue
        rows.extend(_load_jsonl(path))
    if not rows:
        raise SystemExit("no dynamical edge rows found")

    df = pd.DataFrame(rows)
    processed_dir = Path(config.get("output_dir", "results")) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    experiment = config["experiment_name"]
    row_path = processed_dir / f"{experiment}__dynamical_edge_rows.csv"
    summary_path = processed_dir / f"{experiment}__dynamical_edge_summary.csv"

    scalar_cols = [
        "experiment",
        "model",
        "checkpoint",
        "sequence_length",
        "sample_index",
        "active_dim",
        "normalized_frobenius_geomean",
        "normalized_frobenius_mean",
        "edge_distance_log",
        "phase_label",
        "diverged",
        "collapsed",
        "burn_in_steps",
        "eval_steps",
        "frobenius_eval_states",
        "frobenius_probes",
        "operator_update",
        "target",
        "token_mode",
    ]
    available = [col for col in scalar_cols if col in df.columns]
    df[available].to_csv(row_path, index=False)

    group_cols = ["experiment", "model", "checkpoint", "sequence_length", "operator_update", "target", "token_mode"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            samples=("sample_index", "count"),
            mean_normalized_frobenius=("normalized_frobenius_geomean", "mean"),
            median_normalized_frobenius=("normalized_frobenius_geomean", "median"),
            mean_edge_distance_log=("edge_distance_log", "mean"),
            diverged_fraction=("diverged", "mean"),
            collapsed_fraction=("collapsed", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(f"wrote {row_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
