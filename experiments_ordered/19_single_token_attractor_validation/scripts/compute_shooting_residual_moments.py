#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml


EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
CONFIG_PATH = EXP / "configs/experiment19.yaml"
DENSE_ORBITS = EXP / "raw/dense_extension_orbits"


def module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(specification)
    assert specification and specification.loader
    specification.loader.exec_module(value)
    return value


exp19 = module("experiment19_for_residual_moments", EXP / "scripts/run_experiment19.py")


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


def moments(residual: torch.Tensor) -> dict:
    value = residual.double().cpu()
    return {
        "shooting_ri_count": len(value),
        "shooting_ri_mean": float(value.mean()),
        "shooting_ri_variance_population": float(value.var(unbiased=False)),
        "shooting_ri_std_population": float(value.std(unbiased=False)),
        "shooting_ri_rms": float(value.square().mean().sqrt()),
        "shooting_ri_min": float(value.min()),
        "shooting_ri_median": float(value.median()),
        "shooting_ri_p95_recomputed": float(torch.quantile(value, 0.95)),
        "shooting_ri_max": float(value.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    original_rows = read(EXP / "processed/orbit_candidates.csv")
    screen_rows = read(EXP / "processed/screen_summary.csv")
    screen_scale = {
        (row["checkpoint"], row["condition"], row["initial_state_bank"], int(row["token_id"])): float(row["normalization_scale"])
        for row in screen_rows
    }
    solution_path = Path(config["data_root"]) / "validate/orbit_solutions.pt"
    original_solutions = torch.load(solution_path, map_location="cpu", weights_only=True)
    stage4_keys = {row["solution_key"] for row in read(EXP / "processed/stage4_system_summary.csv")}

    tasks: dict[str, list[dict]] = defaultdict(list)
    for row in original_rows:
        key = row["solution_key"]
        scale_key = (row["checkpoint"], row["condition"], row["initial_state_bank"], int(row["token_id"]))
        tasks[row["checkpoint"]].append({
            "protocol": "original_full", "row": row, "orbit": original_solutions[key],
            "orbit_scale": screen_scale[scale_key], "solution_key": key,
        })
    for path in sorted(DENSE_ORBITS.glob("step*.pt"), key=lambda value: int(value.stem[4:])):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tasks[path.stem].append({
            "protocol": "dense_extension", "row": payload["summary"], "orbit": payload["orbit"],
            "orbit_scale": float(payload["summary"]["orbit_scale"]), "solution_key": path.stem,
        })

    summary_rows: list[dict] = []
    phase_rows: list[dict] = []
    max_p95_difference = 0.0
    for checkpoint in sorted(tasks, key=lambda value: int(value[4:])):
        model = exp19.load_model(config, checkpoint, device)
        operator = exp19.PythiaAttractorOperator(model, "full")
        with torch.inference_mode():
            for task in tasks[checkpoint]:
                orbit = task["orbit"].to(device=device, dtype=torch.float32)
                predicted = operator(orbit)
                absolute = torch.linalg.vector_norm(predicted - orbit.roll(-1, 0), dim=-1)
                normalized = absolute / max(float(task["orbit_scale"]), 1e-12)
                stats = moments(normalized)
                source_p95 = float(task["row"]["shooting_normalized_residual_p95"])
                p95_difference = abs(stats["shooting_ri_p95_recomputed"] - source_p95)
                max_p95_difference = max(max_p95_difference, p95_difference)
                common = {
                    "residual_protocol": task["protocol"], "solution_key": task["solution_key"],
                    "checkpoint": checkpoint, "orbit_scale": task["orbit_scale"],
                    "source_shooting_normalized_residual_p95": source_p95,
                    "p95_recompute_absolute_difference": p95_difference,
                    **stats,
                }
                summary_rows.append({**task["row"], **common})
                for phase_index, (raw_value, normalized_value) in enumerate(zip(absolute.cpu(), normalized.cpu())):
                    phase_rows.append({
                        "residual_protocol": task["protocol"], "solution_key": task["solution_key"],
                        "checkpoint": checkpoint, "phase_index": phase_index,
                        "period": len(orbit), "orbit_scale": task["orbit_scale"],
                        "absolute_residual_l2": float(raw_value), "normalized_ri": float(normalized_value),
                    })
        del operator, model
        torch.cuda.empty_cache()
        print(json.dumps({"checkpoint": checkpoint, "tasks": len(tasks[checkpoint])}), flush=True)

    original_summary = [row for row in summary_rows if row["residual_protocol"] == "original_full"]
    dense_summary = [row for row in summary_rows if row["residual_protocol"] == "dense_extension"]
    write(EXP / "processed/shooting_residual_phase_values.csv", phase_rows)
    write(EXP / "processed/orbit_candidates_with_residual_moments.csv", original_summary)
    write(EXP / "processed/stage4_system_summary_with_residual_moments.csv", [row for row in original_summary if row["solution_key"] in stage4_keys])
    write(EXP / "processed/dense_periodic_checkpoint_summary_with_residual_moments.csv", dense_summary)
    validation = {
        "status": "complete", "original_orbits": len(original_summary), "dense_orbits": len(dense_summary),
        "phase_residual_rows": len(phase_rows), "stage4_system_rows": sum(row["solution_key"] in stage4_keys for row in original_summary),
        "maximum_p95_recompute_absolute_difference": max_p95_difference,
        "variance_definition": "population variance: mean((ri - mean(ri))**2)",
    }
    (EXP / "processed/shooting_residual_moments_metadata.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    main()
