#!/usr/bin/env bash
set -u
cd /data1/luohaoming/model_feature

PY=/public/luohaoming/model_feature/.venv/bin/python
CACHE=/home/luohaoming/model_feature_cache/hf_cache
HELPER=experiments_ordered/23_residual_stream_projection/scripts/download_one_checkpoint.py
RANGE_HELPER=experiments_ordered/23_residual_stream_projection/scripts/range_download_blob.py
CHECKPOINTS=(step27000 step28000 step39000 step40000 step58000 step59000)

ENDPOINTS=(https://hf-mirror.com https://hf-p-cfw.fyan.top)
export HF_ENDPOINT=${ENDPOINTS[0]}
export HF_HOME="$CACHE"
export HUGGINGFACE_HUB_CACHE="$CACHE"
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_ENABLE_HF_TRANSFER=0

for checkpoint in "${CHECKPOINTS[@]}"; do
  ready=0
  for attempt in $(seq 1 20); do
    endpoint_index=$(( (attempt - 1) % ${#ENDPOINTS[@]} ))
    export HF_ENDPOINT=${ENDPOINTS[$endpoint_index]}
    echo "$(date -Is) DOWNLOAD $checkpoint attempt=$attempt endpoint=$HF_ENDPOINT"
    model_url="$HF_ENDPOINT/EleutherAI/pythia-70m/resolve/$checkpoint/model.safetensors"
    headers=$(curl -sSI --connect-timeout 30 --max-time 60 "$model_url" || true)
    blob_hash=$(printf '%s\n' "$headers" | awk -F': ' 'tolower($1)=="x-linked-etag"{gsub(/["\r]/,"",$2);print $2;exit}')
    blob_size=$(printf '%s\n' "$headers" | awk -F': ' 'tolower($1)=="x-linked-size"{gsub(/\r/,"",$2);print $2;exit}')
    if [[ -n "$blob_hash" && "$blob_size" =~ ^[0-9]+$ ]]; then
      blob_dir="$CACHE/models--EleutherAI--pythia-70m/blobs"
      final_blob="$blob_dir/$blob_hash"
      mkdir -p "$blob_dir"
      if [[ ! -f "$final_blob" || $(stat -c %s "$final_blob") != "$blob_size" ]]; then
        echo "$(date -Is) RANGE_DOWNLOAD $checkpoint expected=$blob_size sha=$blob_hash"
        "$PY" "$RANGE_HELPER" "$model_url" "$final_blob" "$blob_hash" "$blob_size"
      fi
    fi
    "$PY" "$HELPER" "$checkpoint" "$blob_hash" & child=$!
    last_signature=''; idle=0
    while kill -0 "$child" 2>/dev/null; do
      signature=$(find "$CACHE/models--EleutherAI--pythia-70m/blobs" -maxdepth 1 -type f -name '*.incomplete' -printf '%s:%T@\n' 2>/dev/null | sort | sha256sum | cut -d' ' -f1)
      ref="$CACHE/models--EleutherAI--pythia-70m/refs/$checkpoint"
      links=0
      if [[ -f "$ref" ]]; then sha=$(cat "$ref"); links=$(find "$CACHE/models--EleutherAI--pythia-70m/snapshots/$sha" -maxdepth 1 -type l 2>/dev/null | wc -l); fi
      signature="$signature:$links"
      if [[ "$signature" == "$last_signature" ]]; then idle=$((idle+15)); else idle=0; last_signature="$signature"; fi
      if (( idle >= 120 )); then echo "$(date -Is) STALLED $checkpoint child=$child idle=${idle}s"; kill "$child" 2>/dev/null || true; sleep 2; kill -9 "$child" 2>/dev/null || true; fi
      sleep 15
    done
    if wait "$child"; then
      echo "$(date -Is) READY $checkpoint"
      ready=1
      break
    fi
    echo "$(date -Is) RETRY $checkpoint after failure"
    sleep 20
  done
  if [[ "$ready" != 1 ]]; then
    echo "$(date -Is) FAILED $checkpoint after 20 attempts"
    exit 1
  fi
done

echo "$(date -Is) DOWNLOAD_COMPLETE"
