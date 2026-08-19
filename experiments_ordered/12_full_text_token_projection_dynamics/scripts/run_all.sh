#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/luohaoming/model_feature
PHASE="$REPO/experiments_ordered/12_full_text_token_projection_dynamics"
PY=/public/luohaoming/model_feature/.venv/bin/python
ROOT=/home/luohaoming/model_feature_experiments/full_text_token_projection_dynamics
LOG_ROOT="$ROOT/logs"
mkdir -p "$LOG_ROOT"

checkpoints=(step1000 step41000 step81000 step121000)
pids=()
for index in "${!checkpoints[@]}"; do
  checkpoint="${checkpoints[$index]}"
  CUDA_VISIBLE_DEVICES="$index" "$PY" "$PHASE/scripts/run_full_text_token_projection.py" \
    --config "$PHASE/configs/${checkpoint}.yaml" \
    >"$LOG_ROOT/${checkpoint}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more checkpoint jobs failed; inspect $LOG_ROOT" >&2
  exit 1
fi

"$PY" "$PHASE/scripts/build_full_text_token_projection_report.py" \
  --root "$ROOT" \
  --repo-report-dir "$PHASE"
