#!/usr/bin/env python3
import os
import sys
from huggingface_hub import snapshot_download

checkpoint, blob_hash = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id="EleutherAI/pythia-70m",
    revision=checkpoint,
    cache_dir=os.environ["HF_HOME"],
    allow_patterns=[
        "config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "generation_config.json",
    ],
    max_workers=1,
)
snapshot = __import__('pathlib').Path(path)
model_link = snapshot / "model.safetensors"
if model_link.exists() or model_link.is_symlink(): model_link.unlink()
model_link.symlink_to(__import__('pathlib').Path("../../blobs") / blob_hash)
print(path, flush=True)
