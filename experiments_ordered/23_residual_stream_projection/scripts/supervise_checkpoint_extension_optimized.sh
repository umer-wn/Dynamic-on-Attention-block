#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/luohaoming/model_feature
E23=$ROOT/experiments_ordered/23_residual_stream_projection
E20=$ROOT/experiments_ordered/20_100token_endpoint_jacobian
hundred=$(cat "$E20/logs/fine_100token_extension.pid")
while kill -0 "$hundred" 2>/dev/null; do sleep 30; done
grep -q 'FINE_100TOKEN_COMPLETE' "$E20/logs/fine_100token_extension.log"

old_fast=$(cat "$E23/logs/fine_8token_fast.pid")
if kill -0 "$old_fast" 2>/dev/null; then
  kill -TERM "$old_fast"
  for _ in 1 2 3 4 5; do kill -0 "$old_fast" 2>/dev/null || break; sleep 1; done
  kill -0 "$old_fast" 2>/dev/null && kill -KILL "$old_fast"
fi

cps=(step27000 step28000 step39000 step40000 step58000 step59000)
gpus=(1 2 3 4 5 6)
pids=()
for i in "${!cps[@]}"; do
  cp=${cps[$i]}; gpu=${gpus[$i]}
  /public/luohaoming/model_feature/.venv/bin/python "$E23/scripts/run_fine_checkpoint_metrics_8token_parallel.py" --device "cuda:$gpu" --chunk 128 --checkpoints "$cp" > "$E23/logs/fine_8token_parallel_$cp.log" 2>&1 &
  pids+=("$!")
  echo "STARTED_8TOKEN $cp gpu=$gpu pid=${pids[-1]}"
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "COMPLETE_8TOKEN ${cps[$i]}"; else echo "FAILED_8TOKEN ${cps[$i]}" >&2; failed=1; fi
done
[[ $failed -eq 0 ]]
/public/luohaoming/model_feature/.venv/bin/python "$E23/scripts/finalize_checkpoint_extension.py" > "$E23/logs/finalize_checkpoint_extension.log" 2>&1
echo CHECKPOINT_EXTENSION_COMPLETE
