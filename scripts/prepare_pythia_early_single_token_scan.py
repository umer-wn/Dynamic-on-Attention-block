#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import Dataset


DEFAULT_ARROW = (
    "/home/luohaoming/model_feature_cache/hf_cache/wikitext/wikitext-2-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow"
)
DEFAULT_ROOT = "/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow", default=DEFAULT_ARROW)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--sample-count", type=int, default=128)
    args = parser.parse_args()
    arrow = Path(args.arrow)
    if not arrow.is_file():
        raise FileNotFoundError(arrow)
    dataset = Dataset.from_file(str(arrow))
    rows = []
    texts = []
    for source_row_index, item in enumerate(dataset):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        rows.append({"sample_index": len(rows), "source_row_index": source_row_index, "text": text})
        texts.append(text)
        if len(rows) == args.sample_count:
            break
    if len(rows) != args.sample_count:
        raise RuntimeError(f"expected {args.sample_count} non-empty rows, got {len(rows)}")
    text_hash = hashlib.sha256("\n\0\n".join(texts).encode("utf-8")).hexdigest()
    root = Path(args.root)
    manifest = root / "manifests" / "wikitext_test_first128.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(manifest)
    metadata = {
        "source_arrow": str(arrow),
        "source_arrow_sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
        "sample_count": len(rows),
        "selection": "first 128 non-empty text rows in source order",
        "dataset_text_sha256": text_hash,
        "expected_profile_hash": "3365c25e5e38801ce65b3fa1dcc9b1d65a47d88a75b9ef139b9fcff364bbfbbf",
        "matches_profile": text_hash == "3365c25e5e38801ce65b3fa1dcc9b1d65a47d88a75b9ef139b9fcff364bbfbbf",
        "manifest": str(manifest),
    }
    if not metadata["matches_profile"]:
        raise RuntimeError(f"materialized text hash differs from profiling: {text_hash}")
    atomic_json(root / "manifests" / "wikitext_test_first128.metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
