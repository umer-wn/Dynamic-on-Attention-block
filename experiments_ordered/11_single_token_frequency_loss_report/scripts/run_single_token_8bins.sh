#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/luohaoming/model_feature
PHASE="$REPO/experiments_ordered/11_single_token_frequency_loss_report"
PY=/public/luohaoming/model_feature/.venv/bin/python
COMPUTE="$REPO/scripts/compute_single_token_dynamics.py"
LOG_ROOT=/home/luohaoming/model_feature_experiments/single_token_frequency_8bins/logs
mkdir -p "$LOG_ROOT"

"$PY" "$REPO/scripts/compute_token_frequency.py" \
  --config "$PHASE/configs/frequency_audit_8bins.yaml"

jobs=(
  step0:0:0 step0:1:1 step0:2:2 step0:3:3
  step1000:0:4 step1000:1:5 step1000:2:6 step1000:3:7
  step16000:0:0 step16000:1:1 step16000:2:2 step16000:3:3
  step143000:0:4 step143000:1:5 step143000:2:6 step143000:3:7
)

run_job() {
  local checkpoint="$1"
  local shard="$2"
  local gpu="$3"
  local config="$PHASE/configs/single_token_8bins_${checkpoint}.yaml"
  local log="$LOG_ROOT/${checkpoint}__shard${shard}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$COMPUTE" \
    --config "$config" \
    --shard-index "$shard" \
    --shard-count 4 \
    --output-dir "/home/luohaoming/model_feature_experiments/single_token_frequency_8bins/pilot/${checkpoint}/shard${shard}" \
    >"$log" 2>&1
}

for round in 0 1; do
  pids=()
  start=$((round * 8))
  stop=$((start + 8))
  for ((index=start; index<stop; index++)); do
    IFS=: read -r checkpoint shard gpu <<<"${jobs[$index]}"
    run_job "$checkpoint" "$shard" "$gpu" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "round $round failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
done
