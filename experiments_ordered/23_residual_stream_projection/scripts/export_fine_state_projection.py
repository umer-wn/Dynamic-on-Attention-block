#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import torch

REPO = Path("/data1/luohaoming/model_feature")
EXP16 = REPO / "experiments_ordered/16_frequency_stratified_window_jacobian"
EXP23 = REPO / "experiments_ordered/23_residual_stream_projection"
STATE_ROOT = Path("/data1/luohaoming/model_feature_experiments/experiment23_residual_stream_projection/states")
CHECKPOINTS = ["step27000", "step28000", "step39000", "step40000", "step58000", "step59000"]
FIELDS = ["checkpoint", "dynamic_step", "selection_index", "token_id", "token", "wikitext_train_count", "frequency_bin", "projection_1", "projection_2", "projection_3", "projection_4"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    tokens = read_csv(EXP16 / "manifests/frequency_stratified_tokens.csv")
    saved = torch.load(EXP16 / "processed/projection_basis.pt", map_location="cpu", weights_only=True)
    basis = saved["basis"].float()
    fine_rows: list[dict] = []
    for checkpoint in CHECKPOINTS:
        state_file = STATE_ROOT / f"{checkpoint}_states.pt"
        payload = torch.load(state_file, map_location="cpu", weights_only=True)
        states = payload["states"].float()
        if tuple(states.shape) != (1025, 4, 512):
            raise RuntimeError(f"{checkpoint}: unexpected state shape {tuple(states.shape)}")
        projected = states @ basis.T
        for dynamic_step in range(1025):
            for token_index, token in enumerate(tokens):
                row = {"checkpoint": checkpoint, "dynamic_step": dynamic_step, **token}
                for direction in range(4):
                    row[f"projection_{direction + 1}"] = float(projected[dynamic_step, token_index, direction])
                fine_rows.append(row)
    write_csv(EXP23 / "processed/state_projection_trajectory_fine.csv", fine_rows)
    base_rows = read_csv(EXP16 / "processed/projection_trajectory.csv")
    combined = sorted(base_rows + fine_rows, key=lambda row: (int(row["checkpoint"][4:]), int(row["dynamic_step"]), int(row["selection_index"])))
    write_csv(EXP23 / "processed/state_projection_trajectory_combined.csv", combined)
    print({"fine_rows": len(fine_rows), "combined_rows": len(combined), "checkpoints": CHECKPOINTS})


if __name__ == "__main__":
    main()
