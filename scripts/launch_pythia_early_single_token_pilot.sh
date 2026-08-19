#!/usr/bin/env bash
set -euo pipefail

cd /data1/luohaoming/model_feature
PY=/data1/luohaoming/langurage_feature/venv/bin/python
ROOT=/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan
export HF_HOME=/home/luohaoming/model_feature_cache/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT"/{raw,jacobians,processed,figures,manifests,logs,status}

for revision in step0 step1000 step16000; do
  test -f "$ROOT/status/prefetch/${revision}.json"
  grep -q '"status": "complete"' "$ROOT/status/prefetch/${revision}.json"
done
test -f "$ROOT/manifests/wikitext_test_first128.jsonl"

run_one() {
  local gpu="$1" revision="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/compute_pythia_early_single_token_scan.py \
    --revision "$revision" --sampling-stage coarse --mode full --output-root "$ROOT" --force \
    >"$ROOT/logs/pilot__${revision}.log" 2>&1
}

run_one 5 step0 & p5=$!
run_one 6 step1000 & p6=$!
run_one 7 step16000 & p7=$!
failed=0
wait "$p5" || failed=1
wait "$p6" || failed=1
wait "$p7" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo "pilot GPU worker failed" >&2
  exit 1
fi

"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan')
expected_tokens=None; expected_hash=None
summary=[]
for revision in ['step0','step1000','step16000']:
    directory=root/'raw'/revision
    manifest=json.loads((directory/'manifest.json').read_text())
    jac=[json.loads(line) for line in (directory/'jacobians.jsonl').read_text().splitlines()]
    traj=[json.loads(line) for line in (directory/'trajectories.jsonl').read_text().splitlines()]
    loss=[json.loads(line) for line in (directory/'loss.jsonl').read_text().splitlines()]
    assert len(jac)==16 and {tuple(row['shape']) for row in jac}=={(512,512)}
    assert len(traj)==16*769
    assert all(len([row for row in traj if row['token_id']==token])==769 for token in {row['token_id'] for row in traj})
    assert len(loss)==129 and loss[-1]['sample_index']=='aggregate'
    tokens=manifest['token_ids']; dataset_hash=manifest['dataset_text_sha256']
    expected_tokens=tokens if expected_tokens is None else expected_tokens
    expected_hash=dataset_hash if expected_hash is None else expected_hash
    assert tokens==expected_tokens and dataset_hash==expected_hash
    assert manifest['projection_seed']==1234 and manifest['local_files_only']
    summary.append({
        'checkpoint':revision,
        'loss':manifest['token_weighted_loss'],
        'seconds':manifest['elapsed_seconds'],
        'model_load_seconds':manifest['model_load_seconds'],
        'loss_seconds':manifest['loss_seconds'],
        'trajectory_seconds':manifest['trajectory_seconds'],
        'jacobian_seconds':manifest['jacobian_seconds'],
        'rho_median':sorted(row['normalized_frobenius'] for row in jac)[7:9],
    })
(root/'status/pilot_audit.json').write_text(json.dumps({'status':'passed','rows':summary},indent=2))
(root/'status/pilot_gate_passed.json').write_text(json.dumps({'status':'passed','checkpoints':['step0','step1000','step16000'],'gate_kind':'cached_fallback'},indent=2))
print(json.dumps({'status':'passed','rows':summary}))
PY

"$PY" scripts/analyze_pythia_early_single_token_scan.py --root "$ROOT" \
  >"$ROOT/logs/pilot_analysis.log" 2>&1
"$PY" scripts/plot_pythia_early_single_token_scan.py --root "$ROOT" \
  --report reports/pythia_early_training_frobenius_scan_report.md \
  >"$ROOT/logs/pilot_plot.log" 2>&1
echo "pilot complete"
