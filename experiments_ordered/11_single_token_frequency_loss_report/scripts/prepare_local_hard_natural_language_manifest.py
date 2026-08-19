#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import time
from pathlib import Path

import pyarrow.parquet as pq


SOURCE_ID = "open_web_math_local_hard"
DEFAULT_SOURCE_ROOT = Path(
    "/home/luohaoming/.cache/huggingface/hub/"
    "datasets--open-web-math--open-web-math/snapshots/"
    "fde8ef8de2300f5e778f56261843dab89f230815/data"
)
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, local-only OpenWebMath manifest containing "
            "natural-language-heavy mathematical documents."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--min-chars", type=int, default=1024)
    parser.add_argument("--min-words", type=int, default=120)
    parser.add_argument("--min-letter-fraction", type=float, default=0.55)
    parser.add_argument("--min-math-score", type=float, default=0.80)
    parser.add_argument("--max-stored-chars", type=int, default=16384)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest_path = output_root / "manifests" / f"{SOURCE_ID}.jsonl"
    metadata_path = output_root / "manifests" / f"{SOURCE_ID}.metadata.json"
    if manifest_path.exists() and metadata_path.exists():
        print(
            json.dumps(
                {"status": "already_complete", "manifest_path": str(manifest_path)}
            )
        )
        return

    parquet_paths = sorted(args.source_root.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no local parquet files under {args.source_root}")

    started = time.perf_counter()
    heap: list[tuple[int, str, dict]] = []
    seen_hashes: set[str] = set()
    raw_rows = 0
    eligible_rows = 0
    rejection_counts = {
        "invalid_text_or_metadata": 0,
        "too_short": 0,
        "too_few_words": 0,
        "low_letter_fraction": 0,
        "no_detected_math": 0,
        "low_math_score": 0,
        "duplicate_text": 0,
    }

    for shard_index, parquet_path in enumerate(parquet_paths):
        parquet = pq.ParquetFile(parquet_path)
        shard_row_index = 0
        for batch in parquet.iter_batches(
            batch_size=1024,
            columns=["url", "text", "date", "metadata"],
        ):
            for source in batch.to_pylist():
                original_index = shard_row_index
                shard_row_index += 1
                raw_rows += 1
                text = source.get("text")
                metadata_raw = source.get("metadata")
                if not isinstance(text, str) or not isinstance(metadata_raw, str):
                    rejection_counts["invalid_text_or_metadata"] += 1
                    continue
                if len(text) < int(args.min_chars):
                    rejection_counts["too_short"] += 1
                    continue
                words = WORD_PATTERN.findall(text)
                if len(words) < int(args.min_words):
                    rejection_counts["too_few_words"] += 1
                    continue
                nonspace = sum(not character.isspace() for character in text)
                letters = sum(character.isalpha() and character.isascii() for character in text)
                letter_fraction = letters / max(nonspace, 1)
                if letter_fraction < float(args.min_letter_fraction):
                    rejection_counts["low_letter_fraction"] += 1
                    continue
                try:
                    metadata = json.loads(metadata_raw)
                    extraction = metadata["extraction_info"]
                    found_math = bool(extraction.get("found_math"))
                    math_score = float(extraction.get("math_score", 0.0))
                    source_perplexity = float(extraction.get("perplexity", float("nan")))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    rejection_counts["invalid_text_or_metadata"] += 1
                    continue
                if not found_math:
                    rejection_counts["no_detected_math"] += 1
                    continue
                if math_score < float(args.min_math_score):
                    rejection_counts["low_math_score"] += 1
                    continue

                text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if text_sha256 in seen_hashes:
                    rejection_counts["duplicate_text"] += 1
                    continue
                seen_hashes.add(text_sha256)
                eligible_rows += 1
                priority_digest = hashlib.sha256(
                    f"{args.seed}\0{parquet_path.name}\0{original_index}\0{text_sha256}".encode(
                        "utf-8"
                    )
                ).digest()
                priority = int.from_bytes(priority_digest[:8], "big")
                stored_text = text[: int(args.max_stored_chars)]
                candidate = {
                    "source_id": SOURCE_ID,
                    "dataset_name": "open-web-math/open-web-math",
                    "config": "local_math_natural_language",
                    "split": "train_local_cache",
                    "source_shard": parquet_path.name,
                    "source_shard_index": shard_index,
                    "original_index": original_index,
                    "url_sha256": hashlib.sha256(
                        str(source.get("url", "")).encode("utf-8")
                    ).hexdigest(),
                    "date": source.get("date"),
                    "text": stored_text,
                    "text_sha256": hashlib.sha256(
                        stored_text.encode("utf-8")
                    ).hexdigest(),
                    "original_text_sha256": text_sha256,
                    "original_char_count": len(text),
                    "stored_char_count": len(stored_text),
                    "word_count": len(words),
                    "ascii_letter_fraction": letter_fraction,
                    "math_score": math_score,
                    "source_filter_perplexity": source_perplexity,
                    "selection_priority_u64": priority,
                }
                item = (-priority, text_sha256, candidate)
                if len(heap) < int(args.sample_count):
                    heapq.heappush(heap, item)
                elif priority < -heap[0][0]:
                    heapq.heapreplace(heap, item)

    if len(heap) != int(args.sample_count):
        raise RuntimeError(
            f"only found {len(heap)}/{args.sample_count} eligible unique documents"
        )

    selected = [item[2] for item in heap]
    selected.sort(
        key=lambda row: (
            int(row["selection_priority_u64"]),
            str(row["original_text_sha256"]),
        )
    )
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index

    digest = hashlib.sha256(
        "\n".join(str(row["text_sha256"]) for row in selected).encode("ascii")
    ).hexdigest()
    atomic_jsonl(manifest_path, selected)

    source_files = [
        {
            "path": str(path),
            "linked_size_bytes": path.stat().st_size,
            "resolved_blob": path.resolve().name,
        }
        for path in parquet_paths
    ]
    metadata = {
        "source_id": SOURCE_ID,
        "dataset_name": "open-web-math/open-web-math",
        "dataset_revision": "fde8ef8de2300f5e778f56261843dab89f230815",
        "config": "local_math_natural_language",
        "split": "train_local_cache",
        "local_only": True,
        "source_files": source_files,
        "sampling_method": (
            "select the smallest deterministic SHA-256 priorities over all eligible "
            "documents in the two locally cached shards"
        ),
        "difficulty_definition": (
            "natural-language-heavy OpenWebMath documents with detected mathematics; "
            "selection does not use Pythia loss"
        ),
        "eligibility": {
            "min_chars": int(args.min_chars),
            "min_words": int(args.min_words),
            "min_ascii_letter_fraction_nonspace": float(args.min_letter_fraction),
            "require_found_math": True,
            "min_math_score": float(args.min_math_score),
        },
        "sample_count": len(selected),
        "seed": int(args.seed),
        "raw_records_scanned": raw_rows,
        "eligible_unique_records": eligible_rows,
        "rejection_counts": rejection_counts,
        "max_stored_chars": int(args.max_stored_chars),
        "text_digest_sha256": digest,
        "manifest_path": str(manifest_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(metadata_path, metadata)

    sources_path = output_root / "manifests" / "sources.json"
    sources_payload = (
        json.loads(sources_path.read_text(encoding="utf-8"))
        if sources_path.exists()
        else {"sources": []}
    )
    sources = [
        source
        for source in sources_payload.get("sources", [])
        if source.get("source_id") != SOURCE_ID
    ]
    sources.append({**metadata, "target_sample_count": len(selected)})
    atomic_json(
        sources_path,
        {"sources": sources, "source_count": len(sources)},
    )
    print(json.dumps({"status": "complete", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
