#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
PERTURB = EXP / "processed/dense_periodic_checkpoint_perturbation_256.csv"
SUMMARY = EXP / "processed/dense_periodic_checkpoint_summary.csv"
SELECTION = EXP / "processed/dense_periodic_checkpoint_selection.csv"
DENSE_STATES = REPO / "experiments_ordered/25_dense_checkpoint_suite/raw/states8"
ROBUST_SCALES = (1e-4, 1e-2)
ALL_SCALES = (1e-6, 1e-4, 1e-2)


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def seed(checkpoint: str, token_id: int, epsilon: float, direction: int) -> int:
    text = f"19dense256|{checkpoint}|{token_id}|{epsilon:.1e}|{direction}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)


def label(mean: float) -> str:
    return "contracting" if mean < 1 else "amplifying" if mean > 1 else "neutral"


def robust_summary() -> tuple[list[dict], list[dict]]:
    perturbations = read(PERTURB)
    base = {row["checkpoint"]: row for row in read(SUMMARY)}
    selected = [row for row in perturbations if any(math.isclose(float(row["epsilon"]), value) for value in ROBUST_SCALES)]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        grouped[row["checkpoint"]].append(row)
    output = []
    for checkpoint in sorted(grouped, key=lambda value: int(value[4:])):
        rows = grouped[checkpoint]
        gains = [float(row["response_gain"]) for row in rows]
        logs = [math.log(max(value, 1e-30)) for value in gains]
        mean = statistics.mean(gains)
        old_mean = float(base[checkpoint]["response_gain_mean"])
        output.append({
            "checkpoint": checkpoint,
            "training_step": int(checkpoint[4:]),
            "representative_token_id": rows[0]["representative_token_id"],
            "representative_token": rows[0]["representative_token"],
            "candidate_period": rows[0]["candidate_period"],
            "included_scales": json.dumps(ROBUST_SCALES),
            "directions_per_scale": 8,
            "perturbation_count": len(rows),
            "horizon_steps": 256,
            "response_gain_mean": mean,
            "response_gain_geometric_mean": math.exp(statistics.mean(logs)),
            "response_gain_median": statistics.median(gains),
            "response_gain_min": min(gains),
            "response_gain_max": max(gains),
            "mean_log_response_gain": statistics.mean(logs),
            "contraction_fraction": sum(value < 1 for value in gains) / len(gains),
            "response_label": label(mean),
            "all_three_scales_gain_mean": old_mean,
            "all_three_scales_response_label": label(old_mean),
            "label_changed_after_excluding_1e-6": label(mean) != label(old_mean),
            "strict_shooting_pass": float(base[checkpoint]["shooting_normalized_residual_p95"]) <= 1e-5,
            "strict_floquet_label": base[checkpoint]["stability"] if float(base[checkpoint]["shooting_normalized_residual_p95"]) <= 1e-5 else "not_applicable",
            "leading_multiplier_modulus": base[checkpoint]["leading_multiplier_modulus"] if float(base[checkpoint]["shooting_normalized_residual_p95"]) <= 1e-5 else "",
        })
    write(EXP / "processed/dense_periodic_checkpoint_perturbation_robust_scales.csv", selected)
    write(EXP / "processed/dense_periodic_checkpoint_summary_robust_scales.csv", output)
    return selected, output


def injection_precision_audit() -> tuple[list[dict], list[dict]]:
    selection = read(SELECTION)
    records = []
    for row in selection:
        checkpoint = row["checkpoint"]
        if checkpoint == "step10000":
            continue  # Its 1024-step dense state file does not exist; norm is audited from recorded initial distances below.
        payload = torch.load(DENSE_STATES / f"{checkpoint}.pt", map_location="cpu", weights_only=True)
        token_id = int(row["representative_token_id"])
        token_index = payload["token_ids"].index(token_id)
        base = payload["states"][768, token_index].float().reshape(-1)
        base_norm = float(base.norm())
        ulp = (torch.nextafter(base, torch.full_like(base, float("inf"))) - base).abs()
        for epsilon in ALL_SCALES:
            for direction in range(8):
                generator = torch.Generator(device="cpu").manual_seed(seed(checkpoint, token_id, epsilon, direction))
                unit = torch.nn.functional.normalize(torch.randn(base.numel(), generator=generator), dim=-1)
                intended = float(epsilon) * base_norm * unit
                realized = (base + intended) - base
                target_norm = float(epsilon) * base_norm
                realized_norm = float(realized.norm())
                cosine = float(torch.dot(realized, intended) / (realized.norm() * intended.norm()).clamp_min(1e-30))
                component_ulp = intended.abs() / ulp.clamp_min(torch.finfo(torch.float32).tiny)
                records.append({
                    "checkpoint": checkpoint, "training_step": int(checkpoint[4:]),
                    "token_id": token_id, "epsilon": epsilon, "direction_id": direction,
                    "base_state_l2_norm": base_norm, "target_perturbation_l2": target_norm,
                    "realized_perturbation_l2": realized_norm,
                    "realized_to_target_l2_ratio": realized_norm / target_norm,
                    "direction_cosine": cosine,
                    "unchanged_coordinate_fraction": float((realized == 0).float().mean()),
                    "median_intended_component_in_ulp": float(component_ulp.median()),
                    "p10_intended_component_in_ulp": float(torch.quantile(component_ulp, 0.1)),
                })
    raw = read(PERTURB)
    summaries = []
    for epsilon in ALL_SCALES:
        rows = [row for row in records if math.isclose(float(row["epsilon"]), epsilon)]
        recorded = [row for row in raw if math.isclose(float(row["epsilon"]), epsilon)]
        inferred_norms = [float(row["initial_distance"]) / epsilon for row in recorded]
        item = {"epsilon": epsilon, "audited_checkpoints": 20, "audited_directions": len(rows)}
        for field in ("base_state_l2_norm", "target_perturbation_l2", "realized_to_target_l2_ratio", "direction_cosine", "unchanged_coordinate_fraction", "median_intended_component_in_ulp", "p10_intended_component_in_ulp"):
            values = [float(row[field]) for row in rows]
            item[f"{field}_min"] = min(values)
            item[f"{field}_median"] = statistics.median(values)
            item[f"{field}_max"] = max(values)
        item["all21_inferred_base_norm_min"] = min(inferred_norms)
        item["all21_inferred_base_norm_median"] = statistics.median(inferred_norms)
        item["all21_inferred_base_norm_max"] = max(inferred_norms)
        initial = [float(row["initial_distance"]) for row in recorded]
        final = [float(row["final_distance"]) for row in recorded]
        gains = [float(row["response_gain"]) for row in recorded]
        item["recorded_initial_distance_median"] = statistics.median(initial)
        item["recorded_final_distance_median"] = statistics.median(final)
        item["recorded_response_gain_median"] = statistics.median(gains)
        summaries.append(item)
    write(EXP / "processed/perturbation_injection_precision_audit.csv", records)
    write(EXP / "processed/perturbation_injection_precision_summary.csv", summaries)
    return records, summaries


def main() -> None:
    selected, robust = robust_summary()
    audit, precision = injection_precision_audit()
    print(json.dumps({
        "status": "complete", "robust_perturbation_rows": len(selected),
        "robust_checkpoint_rows": len(robust), "precision_audit_rows": len(audit),
        "precision_summary_rows": len(precision),
        "changed_labels": [row["checkpoint"] for row in robust if row["label_changed_after_excluding_1e-6"]],
    }))


if __name__ == "__main__":
    main()
