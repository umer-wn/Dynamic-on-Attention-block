#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path("/data1/luohaoming/model_feature")
EXP18 = ROOT / "experiments_ordered/18_fine_grained_window_jacobian"
EXP20 = ROOT / "experiments_ordered/20_100token_endpoint_jacobian"
EXP23 = ROOT / "experiments_ordered/23_residual_stream_projection"
FINE = ["step27000", "step28000", "step39000", "step40000", "step58000", "step59000"]
STEPS = list(range(0, 1025, 64))


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    manifest={int(row["token_id"]):row for row in read(EXP18/"manifests/frequency_stratified_tokens_8.csv")}
    raw=[]
    for checkpoint in FINE:
        part=read(EXP23/"processed/fine_metric_parts_8token"/f"{checkpoint}.csv")
        unique={(int(float(row["dynamic_step"])),int(row["token_id"])):row for row in part}
        if len(unique)!=136:
            raise RuntimeError(f"{checkpoint}: expected 136 8-token metric rows, got {len(unique)}")
        for row in unique.values():
            token=manifest[int(row["token_id"])]
            for key in ("selection_index","token","wikitext_train_count","frequency_bin"):
                row[key]=token[key]
            raw.append(row)
    raw.sort(key=lambda row:(int(row["training_step"]),int(float(row["dynamic_step"])),int(row["selection_index"])))
    write(EXP23/"processed/jacobian_fine_grained_8tokens_raw.csv",raw)

    losses={row["checkpoint"]:float(row["proof_pile2_test_loss"]) for row in read(EXP23/"processed/fine_checkpoint_loss.csv")}
    fine=[]
    for checkpoint in FINE:
        for dynamic_step in STEPS:
            group=[row for row in raw if row["checkpoint"]==checkpoint and int(float(row["dynamic_step"]))==dynamic_step]
            if len(group)!=8: raise RuntimeError(f"{checkpoint}/{dynamic_step}: expected 8 tokens, got {len(group)}")
            rho=[float(row["spectral_radius"]) for row in group]
            fro=[float(row["normalized_frobenius_norm"]) for row in group]
            fine.append({"checkpoint":checkpoint,"training_step":int(checkpoint[4:]),"dynamic_step":dynamic_step,"token_count":8,
                         "spectral_radius_median":statistics.median(rho),"spectral_radius_min":min(rho),"spectral_radius_max":max(rho),
                         "normalized_frobenius_norm_median":statistics.median(fro),"normalized_frobenius_norm_min":min(fro),"normalized_frobenius_norm_max":max(fro),
                         "proof_pile2_test_loss":losses[checkpoint]})
    write(EXP23/"processed/jacobian_fine_grained_8tokens.csv",fine)

    base=read(EXP18/"processed/jacobian_fine_grained_8tokens.csv")
    base_losses={row["checkpoint"]:float(row["proof_pile2_test_loss"]) for row in read(EXP18/"processed/proof_pile2_test_loss_by_checkpoint.csv")}
    for row in base:
        if not row.get("normalized_frobenius_norm_median"):
            for suffix in ("median","min","max"):
                row[f"normalized_frobenius_norm_{suffix}"]=float(row[f"jacobian_frobenius_norm_{suffix}"])/math.sqrt(512)
        row["proof_pile2_test_loss"]=base_losses[row["checkpoint"]]
    combined=sorted(base+fine,key=lambda row:(int(row["training_step"]),int(float(row["dynamic_step"]))))
    if len(combined)!=425 or {int(float(row["token_count"])) for row in combined}!={8}:
        raise RuntimeError("combined 8-token metric coverage mismatch")
    write(EXP23/"processed/checkpoint_metrics_combined.csv",combined)

    endpoint=read(EXP20/"processed/checkpoint_metric_summary.csv")
    needed=("spectral_radius_mean","normalized_frobenius_norm_mean","lyapunov_exponent_last_256_mean","lyapunov_exponent_0_1024_mean")
    if len(endpoint)!=25 or any(int(float(row["token_count"]))!=100 for row in endpoint):
        raise RuntimeError(f"100-token summary coverage mismatch: {len(endpoint)} rows")
    if any(not math.isfinite(float(row[key])) for row in endpoint for key in needed):
        raise RuntimeError("100-token summary contains non-finite metrics")

    states=read(EXP23/"processed/state_projection_trajectory_combined.csv")
    if len(states)!=102500 or len({row["checkpoint"] for row in states})!=25:
        raise RuntimeError("combined state trajectory coverage mismatch")

    builder=ROOT/"experiments_ordered/17_visualize/build_dynamic_pair_dashboard.py"
    subprocess.run(["python3",str(builder)],check=True)
    html=ROOT/"experiments_ordered/17_visualize/dynamic_step_projection_visualization.html"
    text=html.read_text(encoding="utf-8")
    for needle in ("100-token终点（均值）","Lyapunov指数（0–1024）","state_projection_trajectory_combined.csv","step59000"):
        if needle not in text: raise RuntimeError(f"HTML missing {needle}")
    print(json.dumps({"status":"complete","state_rows":len(states),"metric_8token_rows":len(combined),"metric_100token_rows":len(endpoint),"html_bytes":html.stat().st_size},ensure_ascii=False))


if __name__=="__main__":
    main()
