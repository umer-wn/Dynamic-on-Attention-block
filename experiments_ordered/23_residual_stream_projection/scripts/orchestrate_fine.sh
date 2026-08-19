#!/usr/bin/env bash
set -euo pipefail
cd /data1/luohaoming/model_feature
EXP=experiments_ordered/23_residual_stream_projection
DOWNLOAD_PID_FILE=${DOWNLOAD_PID_FILE:-$EXP/logs/download_fine_checkpoints_robust.pid}
DOWNLOAD_LOG=${DOWNLOAD_LOG:-$EXP/logs/download_fine_checkpoints_robust.log}
if [[ ${SKIP_DOWNLOAD_WAIT:-0} != 1 ]]; then
  DOWNLOAD_PID=$(cat "$DOWNLOAD_PID_FILE")
  while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do sleep 10; done
fi
grep -q "DOWNLOAD_COMPLETE" "$DOWNLOAD_LOG"
for CP in step27000 step28000 step39000 step40000 step58000 step59000; do
  grep -q "READY $CP" "$DOWNLOAD_LOG"
done

/public/luohaoming/model_feature/.venv/bin/python "$EXP/scripts/run_experiment23.py" --stage formal --checkpoint-set fine --device cuda:3 > "$EXP/logs/fine_residual.log" 2>&1 & R=$!
/public/luohaoming/model_feature/.venv/bin/python "$EXP/scripts/run_fine_checkpoint_loss.py" --device cuda:1 > "$EXP/logs/fine_loss.log" 2>&1 & L=$!
/public/luohaoming/model_feature/.venv/bin/python "$EXP/scripts/run_fine_checkpoint_metrics.py" --device cuda:2 > "$EXP/logs/fine_metrics.log" 2>&1 & M=$!
wait "$R"; wait "$L"; wait "$M"

/public/luohaoming/model_feature/.venv/bin/python - <<'PY'
import csv, math
from pathlib import Path
root=Path('/data1/luohaoming/model_feature')
e23=root/'experiments_ordered/23_residual_stream_projection/processed'
e18=root/'experiments_ordered/18_fine_grained_window_jacobian/processed/jacobian_fine_grained_8tokens.csv'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def write(p,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(p)
losses={r['checkpoint']:r['proof_pile2_test_loss'] for r in read(e23/'fine_checkpoint_loss.csv')}
fine=read(e23/'fine_checkpoint_metrics.csv')
for row in fine: row['proof_pile2_test_loss']=losses[row['checkpoint']]
write(e23/'fine_checkpoint_metrics.csv',fine)
base=read(e23/'residual_projection_trajectory.csv'); extra=read(e23/'residual_projection_trajectory_fine.csv')
write(e23/'residual_projection_trajectory_combined.csv',sorted(base+extra,key=lambda r:(int(r['checkpoint'][4:]),int(r['dynamic_step']),int(r['selection_index']))))
rows=[]
for row in read(e18):
    row['normalized_frobenius_norm_median']=float(row['jacobian_frobenius_norm_median'])/math.sqrt(512)
    row['normalized_frobenius_norm_min']=float(row['jacobian_frobenius_norm_min'])/math.sqrt(512)
    row['normalized_frobenius_norm_max']=float(row['jacobian_frobenius_norm_max'])/math.sqrt(512)
    rows.append(row)
rows+=fine
write(e23/'checkpoint_metrics_combined.csv',sorted(rows,key=lambda r:(int(r['training_step']),int(r['dynamic_step']))))
PY

/public/luohaoming/model_feature/.venv/bin/python experiments_ordered/17_visualize/build_dynamic_pair_dashboard.py
echo COMPLETE
