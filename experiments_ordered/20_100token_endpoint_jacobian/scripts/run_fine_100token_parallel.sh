#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/luohaoming/model_feature
PY=/public/luohaoming/model_feature/.venv/bin/python
SCRIPT=$ROOT/experiments_ordered/20_100token_endpoint_jacobian/scripts/run_100token_endpoint_jacobian.py
LOGDIR=$ROOT/experiments_ordered/20_100token_endpoint_jacobian/logs/fine_checkpoints
mkdir -p "$LOGDIR"
cps=(step27000 step28000 step39000 step40000 step58000 step59000)
gpus=(2 3 4 5 6 7)
pids=()
for i in "${!cps[@]}"; do
  cp=${cps[$i]}; gpu=${gpus[$i]}
  "$PY" "$SCRIPT" --stage compute --checkpoint "$cp" --device "cuda:$gpu" --jacobian-chunk-size 128 > "$LOGDIR/$cp.log" 2>&1 &
  pids+=("$!")
  echo "STARTED $cp gpu=$gpu pid=${pids[-1]}"
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "COMPLETE ${cps[$i]}"
  else
    echo "FAILED ${cps[$i]}" >&2
    failed=1
  fi
done
if [[ $failed -ne 0 ]]; then exit 1; fi
"$PY" "$SCRIPT" --stage summarize > "$LOGDIR/summary.log" 2>&1
echo FINE_100TOKEN_COMPLETE
