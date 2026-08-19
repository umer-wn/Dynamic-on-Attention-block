#!/usr/bin/env bash
set -euo pipefail

cd /data1/luohaoming/model_feature
PY=${PY:-/data1/luohaoming/langurage_feature/venv/bin/python}
ROOT=${ROOT:-/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan}
CACHE=${HF_HOME:-/home/luohaoming/model_feature_cache/hf_cache}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME="$CACHE"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}

mkdir -p "$ROOT/status" "$ROOT/logs"

mapfile -t revisions < <("$PY" - <<'PY'
from pathlib import Path

root = Path("/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan")
cache = Path("/home/luohaoming/model_feature_cache/hf_cache/models--EleutherAI--pythia-70m")
refs = cache / "refs"
targets = [f"step{s}" for s in range(1000, 97001, 4000)] + ["step100000"]
for rev in targets:
    ref = refs / rev
    if not ref.exists():
        print(rev)
        continue
    sha = ref.read_text().strip()
    snap = cache / "snapshots" / sha
    files = {p.name for p in snap.glob("*")} if snap.exists() else set()
    if "config.json" not in files or "model.safetensors" not in files:
        print(rev)
PY
)

printf '%s\n' "${revisions[@]}" >"$ROOT/status/minimal_weight_revisions.txt"
echo "[$(date -Is)] minimal prefetch missing_count=${#revisions[@]}"
if [[ "${#revisions[@]}" -eq 0 ]]; then
  exit 0
fi

"$PY" scripts/prefetch_pythia_checkpoints.py "${revisions[@]}" \
  --max-workers "${PREFETCH_MAX_WORKERS:-1}" \
  --revision-workers "${PREFETCH_REVISION_WORKERS:-1}" \
  --attempts "${PREFETCH_ATTEMPTS:-12}" \
  --retry-sleep "${PREFETCH_RETRY_SLEEP:-60}" \
  --allow-patterns \
    .gitattributes \
    config.json \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    model.safetensors
