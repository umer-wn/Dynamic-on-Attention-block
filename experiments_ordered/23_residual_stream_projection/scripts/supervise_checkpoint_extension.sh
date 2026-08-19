#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/luohaoming/model_feature
E23=$ROOT/experiments_ordered/23_residual_stream_projection
E20=$ROOT/experiments_ordered/20_100token_endpoint_jacobian
fast=$(cat "$E23/logs/fine_8token_fast.pid")
hundred=$(cat "$E20/logs/fine_100token_extension.pid")
while kill -0 "$fast" 2>/dev/null || kill -0 "$hundred" 2>/dev/null; do
  sleep 30
done
grep -q 'FINE_100TOKEN_COMPLETE' "$E20/logs/fine_100token_extension.log"
grep -q '"checkpoint": "step59000"' "$E23/logs/fine_8token_fast.log"
/public/luohaoming/model_feature/.venv/bin/python "$E23/scripts/finalize_checkpoint_extension.py" > "$E23/logs/finalize_checkpoint_extension.log" 2>&1
echo CHECKPOINT_EXTENSION_COMPLETE
