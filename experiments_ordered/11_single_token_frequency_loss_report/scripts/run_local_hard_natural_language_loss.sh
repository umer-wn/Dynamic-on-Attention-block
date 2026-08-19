#!/usr/bin/env bash
set -euo pipefail

REPO=/data1/luohaoming/model_feature
PHASE="$REPO/experiments_ordered/11_single_token_frequency_loss_report"
PY=/public/luohaoming/model_feature/.venv/bin/python
ROOT=/home/luohaoming/model_feature_experiments/single_token_frequency_loss_report
COMPUTE="$REPO/scripts/compute_validation_corpus_loss.py"
LOG_ROOT="$ROOT/logs/open_web_math_local_hard"
mkdir -p "$LOG_ROOT"

"$PY" "$PHASE/scripts/prepare_local_hard_natural_language_manifest.py" \
  --output-root "$ROOT"

revisions=(
  step0 step1000 step5000 step9000 step10000 step13000 step16000
  step17000 step21000 step25000 step29000 step33000 step37000 step41000
  step45000 step49000 step53000 step57000 step61000 step65000 step69000
  step73000 step77000 step81000 step85000 step89000 step93000 step97000
  step100000 step101000 step105000 step133000 step143000
)

run_one() {
  local revision="$1"
  local gpu="$2"
  local tokenizer_args=()
  case "$revision" in
    step101000|step105000|step133000|step143000)
      tokenizer_args=(--tokenizer-revision step100000)
      ;;
  esac
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$COMPUTE" \
    --root "$ROOT" \
    --source-id open_web_math_local_hard \
    --revision "$revision" \
    --sequence-length 64 \
    --loss-batch-size 16 \
    "${tokenizer_args[@]}" \
    >"$LOG_ROOT/${revision}.log" 2>&1
}

for ((start=0; start<${#revisions[@]}; start+=8)); do
  pids=()
  for ((offset=0; offset<8 && start+offset<${#revisions[@]}; offset++)); do
    revision="${revisions[$((start + offset))]}"
    run_one "$revision" "$offset" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more loss jobs failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
done
