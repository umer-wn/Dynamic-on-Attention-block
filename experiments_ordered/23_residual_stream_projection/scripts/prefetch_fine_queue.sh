#!/usr/bin/env bash
set -euo pipefail
cd /data1/luohaoming/model_feature
E23=experiments_ordered/23_residual_stream_projection
PY=/public/luohaoming/model_feature/.venv/bin/python
CACHE=/home/luohaoming/model_feature_cache/hf_cache
endpoint=https://hf-p-cfw.fyan.top
for checkpoint in step58000 step59000; do
  url="$endpoint/EleutherAI/pythia-70m/resolve/$checkpoint/model.safetensors"
  headers=$(curl -sSI --connect-timeout 30 --max-time 60 "$url")
  hash=$(printf '%s\n' "$headers" | awk -F': ' 'tolower($1)=="x-linked-etag"{gsub(/["\r]/,"",$2);print $2;exit}')
  size=$(printf '%s\n' "$headers" | awk -F': ' 'tolower($1)=="x-linked-size"{gsub(/\r/,"",$2);print $2;exit}')
  [[ -n "$hash" && "$size" =~ ^[0-9]+$ ]]
  echo "$(date -Is) PREFETCH $checkpoint hash=$hash size=$size"
  "$PY" "$E23/scripts/range_download_blob.py" "$url" "$CACHE/models--EleutherAI--pythia-70m/blobs/$hash" "$hash" "$size"
  echo "$(date -Is) PREFETCH_READY $checkpoint"
done
echo "$(date -Is) PREFETCH_COMPLETE"
