#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import require_packages

require_packages(["torch", "transformers", "yaml"])

from src.io_utils import load_config, setup_storage_env
from src.model_utils import iter_model_revisions, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_storage_env(config)
    for model_cfg in config["models"]:
        name = model_cfg["name"]
        tokenizer = model_cfg.get("tokenizer", name)
        for revision in iter_model_revisions(model_cfg):
            print(f"loading {name}@{revision}")
            load_model_and_tokenizer(
                name,
                revision,
                tokenizer,
                config.get("dtype", "float32"),
                config.get("device", "auto"),
                config.get("cache_dir"),
                config.get("attn_implementation"),
                bool(config.get("offline", False)),
            )
            print(f"ok {name}@{revision}")


if __name__ == "__main__":
    main()
