#!/usr/bin/env bash
set -euo pipefail

cd /data1/luohaoming/model_feature
PY=${PY:-/data1/luohaoming/langurage_feature/venv/bin/python}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-/home/luohaoming/model_feature_cache/hf_cache}
export HF_HUB_DISABLE_XET=1

revisions=()
for step in $(seq 1000 4000 97000); do
  revisions+=("step${step}")
done
revisions+=("step100000")

"$PY" scripts/prefetch_pythia_checkpoints.py "${revisions[@]}" \
  --max-workers "${PREFETCH_MAX_WORKERS:-2}" \
  --revision-workers "${PREFETCH_REVISION_WORKERS:-1}" \
  --attempts "${PREFETCH_ATTEMPTS:-4}" \
  --retry-sleep "${PREFETCH_RETRY_SLEEP:-20}"
