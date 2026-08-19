#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.checkpoint_utils import checkpoint_step
from src.experiment_io import read_jsonl


THRESHOLD = math.log(1.01)


def paired_bootstrap_delta(a: pd.DataFrame, b: pd.DataFrame, draws: int, seed: int) -> dict:
    joined = a.merge(b, on="sample_index", suffixes=("_a", "_b"), validate="one_to_one").sort_values("sample_index")
    if len(joined) != 128:
        raise RuntimeError(f"paired bootstrap requires 128 matched samples, got {len(joined)}")
    if not np.array_equal(joined.predicted_token_count_a.to_numpy(), joined.predicted_token_count_b.to_numpy()):
        raise RuntimeError("predicted-token counts differ between checkpoints")
    nll_a = joined.nll_sum_a.to_numpy(dtype=np.float64)
    nll_b = joined.nll_sum_b.to_numpy(dtype=np.float64)
    counts = joined.predicted_token_count_a.to_numpy(dtype=np.float64)
    observed = nll_b.sum() / counts.sum() - nll_a.sum() / counts.sum()
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    remaining = int(draws)
    while remaining:
        size = min(1000, remaining)
        indices = rng.integers(0, len(joined), size=(size, len(joined)))
        denominator = counts[indices].sum(axis=1)
        delta = nll_b[indices].sum(axis=1) / denominator - nll_a[indices].sum(axis=1) / denominator
        values.append(delta)
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
    return {
        "delta_loss": observed,
        "ci95_low": lower,
        "ci95_high": upper,
        "practical_threshold_abs_loss": THRESHOLD,
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    args = parser.parse_args()
    root = Path(args.root)
    processed = root / "processed"
    status_dir = root / "status"
    processed.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    loss_rows: list[dict] = []
    jacobian_rows: list[dict] = []
    control_rows: list[dict] = []
    manifests: list[dict] = []
    for checkpoint_dir in sorted((root / "raw").glob("step*"), key=lambda path: checkpoint_step(path.name)):
        loss_path = checkpoint_dir / "loss.jsonl"
        if loss_path.exists():
            loss_rows.extend(read_jsonl(loss_path))
        jacobian_path = checkpoint_dir / "jacobians.jsonl"
        if jacobian_path.exists():
            jacobian_rows.extend(read_jsonl(jacobian_path))
        controls_path = checkpoint_dir / "jacobians_controls.jsonl"
        if controls_path.exists():
            control_rows.extend(read_jsonl(controls_path))
        manifest_path = checkpoint_dir / "manifest.json"
        if manifest_path.exists():
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    if not loss_rows:
        raise RuntimeError("no checkpoint loss rows found")

    loss = pd.DataFrame(loss_rows)
    samples = loss[loss.sample_index.astype(str) != "aggregate"].copy()
    samples["sample_index"] = samples.sample_index.astype(int)
    aggregates = loss[loss.sample_index.astype(str) == "aggregate"].copy()
    aggregates = aggregates[
        ["checkpoint", "training_step", "sampling_stage", "token_weighted_loss", "token_weighted_perplexity",
         "predicted_token_count", "dataset_text_sha256"]
    ].drop_duplicates("checkpoint").sort_values("training_step")
    if aggregates.dataset_text_sha256.nunique() != 1:
        raise RuntimeError("dataset hash differs across checkpoints")
    if samples.groupby("checkpoint").sample_index.nunique().nunique() != 1 or samples.groupby("checkpoint").sample_index.nunique().iloc[0] != 128:
        raise RuntimeError("checkpoint sample ids are not the same fixed 128-sample cohort")
    samples.to_csv(processed / "loss_by_sample.csv", index=False)
    aggregates.to_csv(processed / "checkpoint_loss.csv", index=False)

    deltas: list[dict] = []
    checkpoints = aggregates.checkpoint.tolist()
    for pair_index, (left, right) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
        a = samples[samples.checkpoint == left]
        b = samples[samples.checkpoint == right]
        result = paired_bootstrap_delta(a, b, args.bootstrap_draws, args.bootstrap_seed + pair_index)
        deltas.append(
            {
                "checkpoint_a": left,
                "checkpoint_b": right,
                "step_a": checkpoint_step(left),
                "step_b": checkpoint_step(right),
                **result,
            }
        )
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(processed / "adjacent_loss_deltas.csv", index=False)

    if jacobian_rows:
        frobenius = pd.DataFrame(jacobian_rows)
        frobenius.to_csv(processed / "token_frobenius.csv", index=False)
        summary = frobenius.groupby(
            ["checkpoint", "training_step", "sampling_stage"], as_index=False
        ).agg(
            token_count=("normalized_frobenius", "count"),
            median=("normalized_frobenius", "median"),
            mean=("normalized_frobenius", "mean"),
            std=("normalized_frobenius", "std"),
            q25=("normalized_frobenius", lambda values: values.quantile(0.25)),
            q75=("normalized_frobenius", lambda values: values.quantile(0.75)),
        )
        summary.sort_values("training_step").to_csv(processed / "checkpoint_frobenius.csv", index=False)

        tail_conditions = frobenius.copy()
        tail_conditions["condition"] = "tail_t767"
        if control_rows:
            controls = pd.DataFrame(control_rows)
            controls.to_csv(processed / "token_frobenius_controls.csv", index=False)
            all_conditions = pd.concat([tail_conditions, controls], ignore_index=True, sort=False)
        else:
            all_conditions = tail_conditions
        all_conditions.to_csv(processed / "token_frobenius_all_conditions.csv", index=False)
        condition_summary = all_conditions.groupby(
            ["checkpoint", "training_step", "condition"], as_index=False
        ).agg(
            token_count=("normalized_frobenius", "count"),
            median=("normalized_frobenius", "median"),
            mean=("normalized_frobenius", "mean"),
            std=("normalized_frobenius", "std"),
            q25=("normalized_frobenius", lambda values: values.quantile(0.25)),
            q75=("normalized_frobenius", lambda values: values.quantile(0.75)),
        )
        condition_summary.sort_values(["condition", "training_step"]).to_csv(
            processed / "checkpoint_frobenius_conditions.csv", index=False
        )

    evaluated_steps = sorted(int(value) for value in aggregates.training_step)
    regular_evaluated = {step for step in evaluated_steps if 1000 <= step <= 100000 and step % 1000 == 0}
    all_regular = set(range(1000, 100001, 1000))
    loss_by_step = dict(zip(aggregates.training_step.astype(int), aggregates.token_weighted_loss.astype(float)))

    confirmed: list[dict] = []
    for index in range(1, len(delta_frame)):
        incoming = delta_frame.iloc[index - 1]
        outgoing = delta_frame.iloc[index]
        pivot = int(incoming.step_b)
        if int(incoming.step_b) != int(outgoing.step_a):
            continue
        if int(incoming.step_b) - int(incoming.step_a) != 1000 or int(outgoing.step_b) - int(outgoing.step_a) != 1000:
            continue
        if incoming.label == "significant_decrease" and outgoing.label == "significant_increase":
            confirmed.append(
                {
                    "pivot_step": pivot,
                    "pivot_loss": loss_by_step[pivot],
                    "incoming": incoming.to_dict(),
                    "outgoing": outgoing.to_dict(),
                }
            )
    confirmed.sort(key=lambda row: (row["pivot_loss"], row["pivot_step"]))

    descriptive: list[dict] = []
    ordered = aggregates.sort_values("training_step").reset_index(drop=True)
    for index in range(1, len(ordered) - 1):
        left, middle, right = ordered.iloc[index - 1], ordered.iloc[index], ordered.iloc[index + 1]
        if middle.token_weighted_loss < left.token_weighted_loss and middle.token_weighted_loss < right.token_weighted_loss:
            descriptive.append(
                {
                    "pivot_step": int(middle.training_step),
                    "pivot_loss": float(middle.token_weighted_loss),
                    "left_step": int(left.training_step),
                    "right_step": int(right.training_step),
                }
            )
    descriptive.sort(key=lambda row: (row["pivot_loss"], row["pivot_step"]))

    adaptive_loss: list[int] = []
    adaptive_full: list[int] = []
    if confirmed:
        primary = confirmed[0]
        pivot = int(primary["pivot_step"])
        band_start = max(1000, pivot - 2000)
        band_end = min(100000, pivot + 2000)
        adaptive_full = list(range(band_start, band_end + 1, 1000))
        search_status = "confirmed_reversal"
    elif regular_evaluated == all_regular:
        primary = descriptive[0] if descriptive else None
        if primary:
            pivot = int(primary["pivot_step"])
            adaptive_full = list(range(max(1000, pivot - 2000), min(100000, pivot + 2000) + 1, 1000))
        search_status = "no_confirmed_reversal_in_first_100"
    else:
        primary = descriptive[0] if descriptive else None
        if primary is None:
            minimum_row = ordered.loc[ordered.token_weighted_loss.idxmin()]
            pivot = int(minimum_row.training_step)
            primary = {"pivot_step": pivot, "pivot_loss": float(minimum_row.token_weighted_loss)}
        pivot = min(100000, max(1000, int(primary["pivot_step"])))
        missing = sorted(all_regular - regular_evaluated, key=lambda step: (abs(step - pivot), step))
        adaptive_loss = missing[:12]
        search_status = "needs_more_loss_points"

    full_steps = {
        checkpoint_step(row["checkpoint"])
        for row in manifests
        if row.get("mode") == "full"
    }
    adaptive_full = [step for step in adaptive_full if step not in full_steps]
    (status_dir / "adaptive_loss_revisions.txt").write_text(
        "".join(f"step{step}\n" for step in adaptive_loss), encoding="utf-8"
    )
    (status_dir / "adaptive_full_revisions.txt").write_text(
        "".join(f"step{step}\n" for step in adaptive_full), encoding="utf-8"
    )
    selection = {
        "search_status": search_status,
        "evaluated_loss_checkpoints": len(aggregates),
        "evaluated_regular_checkpoints": len(regular_evaluated),
        "full_dynamics_checkpoints": len(full_steps),
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "practical_threshold_abs_loss": THRESHOLD,
        "confirmed_reversals": confirmed,
        "descriptive_local_minima": descriptive,
        "primary_candidate": primary,
        "next_loss_revisions": [f"step{step}" for step in adaptive_loss],
        "next_full_revisions": [f"step{step}" for step in adaptive_full],
    }
    (status_dir / "adaptive_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
