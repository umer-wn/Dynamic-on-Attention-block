#!/usr/bin/env bash
set -u
cd /data1/luohaoming/model_feature

PY=/public/luohaoming/model_feature/.venv/bin/python
E24=experiments_ordered/24_hle_subset_loss
E23=experiments_ordered/23_residual_stream_projection
HLE=$E24/data/official/data/test-00000-of-00001.parquet
HLE_PID=${1:?HLE download PID required}

echo "$(date -Is) WAIT_HLE pid=$HLE_PID"
while kill -0 "$HLE_PID" 2>/dev/null; do
  partial=$(find "$E24/data/official/.cache/huggingface/download" -name '*.incomplete' -printf '%s\n' 2>/dev/null | sort -nr | head -1)
  echo "$(date -Is) HLE_DOWNLOADING partial_bytes=${partial:-0}"
  sleep 30
done

if [[ ! -f "$HLE" ]]; then
  echo "$(date -Is) HLE_FAILED final_file_missing"
  exit 1
fi
"$PY" - "$HLE" <<'PY'
import sys
import pyarrow.parquet as pq
p=sys.argv[1]
t=pq.read_table(p, columns=['question','answer'])
assert t.num_rows >= 2000, t.num_rows
print({'hle_file':p,'rows':t.num_rows,'columns':t.column_names},flush=True)
PY
echo "$(date -Is) HLE_READY bytes=$(stat -c %s "$HLE")"

nohup bash "$E23/scripts/download_fine_checkpoints_robust.sh" > "$E23/logs/download_fine_checkpoints_robust_resume.log" 2>&1 < /dev/null &
download_pid=$!
echo "$download_pid" > "$E23/logs/download_fine_checkpoints_robust_resume.pid"
echo "$(date -Is) FINE_DOWNLOAD_STARTED pid=$download_pid"
while kill -0 "$download_pid" 2>/dev/null; do sleep 30; done
grep -q 'DOWNLOAD_COMPLETE' "$E23/logs/download_fine_checkpoints_robust_resume.log" || { echo "$(date -Is) FINE_DOWNLOAD_FAILED"; exit 1; }

echo "$(date -Is) FINE_EXPERIMENTS_START"
SKIP_DOWNLOAD_WAIT=1 DOWNLOAD_LOG="$E23/logs/download_fine_checkpoints_robust_resume.log" bash "$E23/scripts/orchestrate_fine.sh"
grep -q '^COMPLETE$' "$E23/logs/fine_supervised_orchestrate.log" 2>/dev/null || true

"$PY" - <<'PY'
import csv
from pathlib import Path
root=Path('experiments_ordered/23_residual_stream_projection/processed')
def rows(name):
    with (root/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
fine_cps={'step27000','step28000','step39000','step40000','step58000','step59000'}
res=rows('residual_projection_trajectory_fine.csv')
loss=rows('fine_checkpoint_loss.csv')
metrics=rows('fine_checkpoint_metrics.csv')
assert {r['checkpoint'] for r in res} == fine_cps
assert {r['checkpoint'] for r in loss} == fine_cps
assert {r['checkpoint'] for r in metrics} == fine_cps
assert len(res) == 6*4*1024, len(res)
assert len(metrics) == 6*17, len(metrics)
print({'residual_rows':len(res),'loss_rows':len(loss),'metric_rows':len(metrics),'status':'COMPLETE'},flush=True)
PY
echo "$(date -Is) ALL_COMPLETE"
