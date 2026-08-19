#!/usr/bin/env bash
set -euo pipefail

cd /data1/luohaoming/model_feature
PY=/data1/luohaoming/langurage_feature/venv/bin/python
ROOT=/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan
mkdir -p "$ROOT"/{raw,jacobians,processed,figures,manifests,logs,status}
test -f "$ROOT/status/pilot_gate_passed.json"
grep -q '"status": "passed"' "$ROOT/status/pilot_gate_passed.json"
export HF_HOME=/home/luohaoming/model_feature_cache/hf_cache
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
PREFETCH_MAX_WORKERS=${PREFETCH_MAX_WORKERS:-2}
PREFETCH_REVISION_WORKERS=${PREFETCH_REVISION_WORKERS:-1}
PREFETCH_ATTEMPTS=${PREFETCH_ATTEMPTS:-4}
PREFETCH_RETRY_SLEEP=${PREFETCH_RETRY_SLEEP:-20}

prefetch_revisions() {
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  echo "[$(date -Is)] PREFETCH count=$# max_workers=$PREFETCH_MAX_WORKERS revision_workers=$PREFETCH_REVISION_WORKERS endpoint=$HF_ENDPOINT"
  "$PY" scripts/prefetch_pythia_checkpoints.py "$@" \
    --max-workers "$PREFETCH_MAX_WORKERS" \
    --revision-workers "$PREFETCH_REVISION_WORKERS" \
    --attempts "$PREFETCH_ATTEMPTS" \
    --retry-sleep "$PREFETCH_RETRY_SLEEP"
}

run_queue() {
  local gpu="$1"
  local mode="$2"
  local stage="$3"
  shift 3
  local revision
  for revision in "$@"; do
    echo "[$(date -Is)] gpu=$gpu mode=$mode stage=$stage revision=$revision START"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/compute_pythia_early_single_token_scan.py \
      --revision "$revision" \
      --mode "$mode" \
      --sampling-stage "$stage" \
      --output-root "$ROOT" \
      >"$ROOT/logs/${stage}__${mode}__${revision}.log" 2>&1
    echo "[$(date -Is)] gpu=$gpu mode=$mode stage=$stage revision=$revision DONE"
  done
}

run_distributed() {
  local mode="$1"
  local stage="$2"
  shift 2
  local revisions=("$@")
  local q5=() q6=() q7=()
  local index revision
  for index in "${!revisions[@]}"; do
    revision="${revisions[$index]}"
    case $((index % 3)) in
      0) q5+=("$revision") ;;
      1) q6+=("$revision") ;;
      2) q7+=("$revision") ;;
    esac
  done
  run_queue 5 "$mode" "$stage" "${q5[@]}" & local p5=$!
  run_queue 6 "$mode" "$stage" "${q6[@]}" & local p6=$!
  run_queue 7 "$mode" "$stage" "${q7[@]}" & local p7=$!
  local failed=0
  wait "$p5" || failed=1
  wait "$p6" || failed=1
  wait "$p7" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more GPU queues failed" >&2
    return 1
  fi
}

run_control_queue() {
  local gpu="$1"
  shift
  local revision
  for revision in "$@"; do
    echo "[$(date -Is)] gpu=$gpu controls revision=$revision START"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/compute_pythia_single_token_jacobian_controls.py \
      --revision "$revision" \
      --root "$ROOT" \
      >"$ROOT/logs/controls__${revision}.log" 2>&1
    echo "[$(date -Is)] gpu=$gpu controls revision=$revision DONE"
  done
}

run_controls_distributed() {
  local revisions=("$@")
  local q5=() q6=() q7=()
  local index revision
  for index in "${!revisions[@]}"; do
    revision="${revisions[$index]}"
    case $((index % 3)) in
      0) q5+=("$revision") ;;
      1) q6+=("$revision") ;;
      2) q7+=("$revision") ;;
    esac
  done
  run_control_queue 5 "${q5[@]}" & local p5=$!
  run_control_queue 6 "${q6[@]}" & local p6=$!
  run_control_queue 7 "${q7[@]}" & local p7=$!
  local failed=0
  wait "$p5" || failed=1
  wait "$p6" || failed=1
  wait "$p7" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more control queues failed" >&2
    return 1
  fi
}

coarse=()
for step in $(seq 1000 4000 97000); do
  coarse+=("step${step}")
done
if [[ "${#coarse[@]}" -ne 25 ]]; then
  echo "coarse grid must contain 25 checkpoints" >&2
  exit 1
fi

echo "[$(date -Is)] MILESTONE coarse full scan"
prefetch_revisions "${coarse[@]}"
run_distributed full coarse "${coarse[@]}"
run_controls_distributed "${coarse[@]}"

echo "[$(date -Is)] MILESTONE sentinel loss"
prefetch_revisions step0 step100000
run_distributed loss-only sentinel step0 step100000

"$PY" scripts/analyze_pythia_early_single_token_scan.py --root "$ROOT" \
  >"$ROOT/logs/adaptive_analysis_0.log" 2>&1

for iteration in $(seq 1 10); do
  mapfile -t adaptive <"$ROOT/status/adaptive_loss_revisions.txt"
  if [[ "${#adaptive[@]}" -eq 0 ]]; then
    break
  fi
  echo "[$(date -Is)] MILESTONE adaptive loss iteration=$iteration count=${#adaptive[@]}"
  prefetch_revisions "${adaptive[@]}"
  run_distributed loss-only adaptive "${adaptive[@]}"
  "$PY" scripts/analyze_pythia_early_single_token_scan.py --root "$ROOT" \
    >"$ROOT/logs/adaptive_analysis_${iteration}.log" 2>&1
done

mapfile -t adaptive_full <"$ROOT/status/adaptive_full_revisions.txt"
if [[ "${#adaptive_full[@]}" -gt 0 ]]; then
  echo "[$(date -Is)] MILESTONE adaptive full band count=${#adaptive_full[@]}"
  prefetch_revisions "${adaptive_full[@]}"
  run_distributed full adaptive "${adaptive_full[@]}"
  run_controls_distributed "${adaptive_full[@]}"
  "$PY" scripts/analyze_pythia_early_single_token_scan.py --root "$ROOT" \
    >"$ROOT/logs/adaptive_analysis_final.log" 2>&1
fi

"$PY" scripts/plot_pythia_early_single_token_scan.py --root "$ROOT" \
  --report reports/pythia_early_training_frobenius_scan_report.md \
  >"$ROOT/logs/plot_and_report.log" 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan')
selection = json.loads((root / 'status/adaptive_selection.json').read_text())
complete = {
    'status': 'complete',
    'search_status': selection['search_status'],
    'loss_checkpoints': selection['evaluated_loss_checkpoints'],
    'regular_loss_checkpoints': selection['evaluated_regular_checkpoints'],
    'full_dynamics_checkpoints': selection['full_dynamics_checkpoints'],
}
(root / 'status/experiment_complete.json').write_text(json.dumps(complete, indent=2))
print(json.dumps(complete))
PY

echo "[$(date -Is)] EXPERIMENT COMPLETE"
