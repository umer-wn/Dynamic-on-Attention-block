#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/luohaoming/model_feature
PY=/public/luohaoming/model_feature/.venv/bin/python
EXP=$ROOT/experiments_ordered/23_residual_stream_projection
SCRIPT=$ROOT/experiments_ordered/18_fine_grained_window_jacobian/scripts/run_experiment18.py
mkdir -p "$EXP/logs"
"$PY" "$SCRIPT" \
  --stage all \
  --data-root /data1/luohaoming/model_feature_experiments/experiment23_fine_8token_states \
  --report-root "$EXP" \
  --cache-dir /home/luohaoming/model_feature_cache/hf_cache \
  --token-manifest "$ROOT/experiments_ordered/18_fine_grained_window_jacobian/manifests/frequency_stratified_tokens_8.csv" \
  --device cuda:1 \
  --jacobian-chunk-size 128 \
  --checkpoints step27000 step28000 step39000 step40000 step58000 step59000
echo FINE_8TOKEN_COMPLETE
