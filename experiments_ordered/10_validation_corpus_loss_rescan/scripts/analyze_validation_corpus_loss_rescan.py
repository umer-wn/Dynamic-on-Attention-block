#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpoint_utils import checkpoint_step
from src.experiment_io import atomic_json, read_jsonl


DEFAULT_ROOT = "/home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan"
FROB_ROOT = "/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan"
THRESHOLD = math.log(1.01)


def paired_bootstrap_delta(a: pd.DataFrame, b: pd.DataFrame, draws: int, seed: int) -> dict:
    joined = a.merge(b, on=["source_id", "sample_index"], suffixes=("_a", "_b"), validate="one_to_one").sort_values(["source_id", "sample_index"])
    if joined.empty:
        raise RuntimeError("paired bootstrap has no matched samples")
    if not np.array_equal(joined.predicted_token_count_a.to_numpy(), joined.predicted_token_count_b.to_numpy()):
        raise RuntimeError("predicted-token counts differ between checkpoints")
    nll_a = joined.nll_sum_a.to_numpy(dtype=np.float64)
    nll_b = joined.nll_sum_b.to_numpy(dtype=np.float64)
    counts = joined.predicted_token_count_a.to_numpy(dtype=np.float64)
    observed = nll_b.sum() / counts.sum() - nll_a.sum() / counts.sum()
    rng = np.random.default_rng(seed)
    values = []
    remaining = int(draws)
    while remaining:
        size = min(1000, remaining)
        indices = rng.integers(0, len(joined), size=(size, len(joined)))
        denominator = counts[indices].sum(axis=1)
        values.append(nll_b[indices].sum(axis=1) / denominator - nll_a[indices].sum(axis=1) / denominator)
        remaining -= size
    boot = np.concatenate(values)
    lower, upper = np.quantile(boot, [0.025, 0.975])
    if upper < 0 and observed <= -THRESHOLD:
        label = "significant_decrease"
    elif lower > 0 and observed >= THRESHOLD:
        label = "significant_increase"
    elif observed < 0:
        label = "descriptive_decrease"
    elif observed > 0:
        label = "descriptive_increase"
    else:
        label = "no_change"
    return {"delta_loss": observed, "ci95_low": lower, "ci95_high": upper, "practical_threshold_abs_loss": THRESHOLD, "label": label}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--frobenius-root", default=FROB_ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    args = parser.parse_args()
    root = Path(args.root)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    loss_rows = []
    completion_rows = []
    for loss_path in sorted((root / "raw").glob("*/*/loss.jsonl")):
        loss_rows.extend(read_jsonl(loss_path))
        complete_path = loss_path.parent / "loss_complete.json"
        if complete_path.exists():
            completion_rows.append(json.loads(complete_path.read_text(encoding="utf-8")))
    if not loss_rows:
        raise RuntimeError("no validation corpus loss rows found")
    loss = pd.DataFrame(loss_rows)
    samples = loss[loss.sample_index.astype(str) != "aggregate"].copy()
    samples["sample_index"] = samples.sample_index.astype(int)
    aggregates = loss[loss.sample_index.astype(str) == "aggregate"].copy()
    aggregates = aggregates[
        ["source_id", "dataset_name", "config", "split", "checkpoint", "training_step", "token_weighted_loss",
         "token_weighted_perplexity", "predicted_token_count", "manifest_text_digest_sha256", "sample_count"]
    ].drop_duplicates(["source_id", "checkpoint"]).sort_values(["source_id", "training_step"])
    samples.to_csv(processed / "loss_by_sample.csv", index=False)
    aggregates.to_csv(processed / "checkpoint_loss_by_source.csv", index=False)
    deltas = []
    for source_id, group in aggregates.groupby("source_id"):
        checkpoints = group.checkpoint.tolist()
        for pair_index, (left, right) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
            a = samples[(samples.source_id == source_id) & (samples.checkpoint == left)]
            b = samples[(samples.source_id == source_id) & (samples.checkpoint == right)]
            result = paired_bootstrap_delta(a, b, args.bootstrap_draws, args.bootstrap_seed + pair_index)
            deltas.append({"source_id": source_id, "checkpoint_a": left, "checkpoint_b": right, "step_a": checkpoint_step(left), "step_b": checkpoint_step(right), **result})
    delta_columns = ["source_id", "checkpoint_a", "checkpoint_b", "step_a", "step_b", "delta_loss", "ci95_low", "ci95_high", "practical_threshold_abs_loss", "label"]
    delta_frame = pd.DataFrame(deltas, columns=delta_columns)
    delta_frame.to_csv(processed / "adjacent_loss_deltas_by_source.csv", index=False)
    rebound = []
    for source_id, group in aggregates.groupby("source_id"):
        by_checkpoint = {row.checkpoint: row for row in group.itertuples()}
        if "step16000" in by_checkpoint and "step100000" in by_checkpoint:
            loss_16 = float(by_checkpoint["step16000"].token_weighted_loss)
            loss_100 = float(by_checkpoint["step100000"].token_weighted_loss)
            label = "descriptive_rebound" if loss_100 > loss_16 else "no_rebound_detected_on_this_source"
            rebound.append({"source_id": source_id, "step16000_loss": loss_16, "step100000_loss": loss_100, "delta_100000_minus_16000": loss_100 - loss_16, "label": label})
    pd.DataFrame(rebound, columns=["source_id", "step16000_loss", "step100000_loss", "delta_100000_minus_16000", "label"]).to_csv(processed / "step16000_vs_step100000_by_source.csv", index=False)
    frob_path = Path(args.frobenius_root) / "processed" / "checkpoint_frobenius_conditions.csv"
    if frob_path.exists():
        frob = pd.read_csv(frob_path)
        merged = aggregates.merge(frob, on="training_step", how="left", suffixes=("_loss", "_frob"))
        merged.to_csv(processed / "loss_vs_frobenius_merged.csv", index=False)
    atomic_json(
        root / "status" / "analysis_complete.json",
        {
            "status": "complete",
            "source_count": int(aggregates.source_id.nunique()),
            "loss_rows": int(len(loss_rows)),
            "checkpoint_source_pairs": int(len(aggregates)),
            "rebound_summary": rebound,
        },
    )
    print(json.dumps({"status": "complete", "sources": sorted(aggregates.source_id.unique())}, indent=2))


if __name__ == "__main__":
    main()
