#!/usr/bin/env python3
"""Inspect tokenizer and model vocabulary sizes without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODEL = "EleutherAI/pythia-70m"
DEFAULT_REVISION = "step0"
DEFAULT_CACHE = Path("/home/luohaoming/model_feature_cache/hf_cache")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print the tokenizer vocabulary size and the model-config vocabulary "
            "size. Model weights are not loaded."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow missing tokenizer/config files to be downloaded from the Hub.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()

    common = {
        "pretrained_model_name_or_path": args.model,
        "revision": args.revision,
        "cache_dir": str(args.cache_dir),
        "local_files_only": not args.allow_download,
    }
    tokenizer = AutoTokenizer.from_pretrained(**common)
    config = AutoConfig.from_pretrained(**common)

    vocabulary = tokenizer.get_vocab()
    token_ids = list(vocabulary.values())
    tokenizer_size = len(tokenizer)
    base_vocab_size = int(tokenizer.vocab_size)
    model_vocab_size = int(config.vocab_size)

    result = {
        "model": args.model,
        "revision": args.revision,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": base_vocab_size,
        "len_tokenizer_including_added_tokens": tokenizer_size,
        "get_vocab_entries": len(vocabulary),
        "minimum_token_id": min(token_ids),
        "maximum_token_id": max(token_ids),
        "maximum_token_id_plus_one": max(token_ids) + 1,
        "added_tokens": tokenizer_size - base_vocab_size,
        "model_config_vocab_size": model_vocab_size,
        "model_padding_slots_beyond_tokenizer": model_vocab_size - tokenizer_size,
        "special_tokens": {
            "bos": {"token": tokenizer.bos_token, "id": tokenizer.bos_token_id},
            "eos": {"token": tokenizer.eos_token, "id": tokenizer.eos_token_id},
            "pad": {"token": tokenizer.pad_token, "id": tokenizer.pad_token_id},
            "unk": {"token": tokenizer.unk_token, "id": tokenizer.unk_token_id},
        },
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"Model / revision              : {args.model} / {args.revision}")
    print(f"Tokenizer class               : {result['tokenizer_class']}")
    print(f"Tokenizer base vocab_size     : {base_vocab_size:,}")
    print(f"len(tokenizer)                : {tokenizer_size:,}")
    print(f"get_vocab() entries           : {len(vocabulary):,}")
    print(f"Token ID range                : {min(token_ids):,} .. {max(token_ids):,}")
    print(f"Model config vocab_size       : {model_vocab_size:,}")
    print(
        "Padding slots in model vocab  : "
        f"{model_vocab_size - tokenizer_size:,}"
    )
    print()
    print(
        "Interpretation: tokenizer size is the number of usable token IDs; "
        "model config vocab_size can be larger because the embedding/output "
        "matrices are padded for hardware-friendly dimensions."
    )


if __name__ == "__main__":
    main()
