#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path


DEFAULT_URL = (
    "https://hf-mirror.com/datasets/monology/pile-uncopyrighted/"
    "resolve/main/train/00.jsonl.zst"
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            # ASCII escaping keeps U+0085/U+2028/U+2029 inside one physical
            # JSONL record when downstream readers use str.splitlines().
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small deterministic The Pile train-split manifest without downloading a full 11 GB shard."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--scan-limit", type=int, default=20_000)
    parser.add_argument("--min-chars", type=int, default=128)
    parser.add_argument("--range-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.sample_count <= 0 or args.scan_limit < args.sample_count:
        raise ValueError("scan-limit must be at least sample-count")

    root = Path(args.output_root)
    manifest_path = root / "manifests" / "the_pile_train.jsonl"
    metadata_path = root / "manifests" / "the_pile_train.metadata.json"
    if manifest_path.exists() and metadata_path.exists():
        print(json.dumps({"status": "already_complete", "manifest_path": str(manifest_path)}))
        return

    curl = subprocess.Popen(
        [
            "curl",
            "-fL",
            "--retry",
            "3",
            "--range",
            f"0-{int(args.range_bytes) - 1}",
            args.source_url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if curl.stdout is None:
        raise RuntimeError("failed to open curl stdout")
    zstd = subprocess.Popen(
        ["zstd", "-dc"],
        stdin=curl.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    curl.stdout.close()
    if zstd.stdout is None:
        raise RuntimeError("failed to open zstd stdout")

    rng = random.Random(int(args.seed))
    reservoir: list[dict] = []
    valid_seen = 0
    raw_seen = 0
    started = time.perf_counter()
    try:
        for raw_line in zstd.stdout:
            raw_seen += 1
            try:
                source = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            text = source.get("text")
            if not isinstance(text, str) or len(text.strip()) < int(args.min_chars):
                continue
            candidate = {
                "source_id": "the_pile_train",
                "dataset_name": "monology/pile-uncopyrighted",
                "config": None,
                "split": "train",
                "original_index": raw_seen - 1,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            valid_seen += 1
            if len(reservoir) < int(args.sample_count):
                reservoir.append(candidate)
            else:
                replacement = rng.randrange(valid_seen)
                if replacement < int(args.sample_count):
                    reservoir[replacement] = candidate
            if valid_seen >= int(args.scan_limit):
                break
    finally:
        zstd.stdout.close()
        zstd.terminate()
        curl.terminate()
        try:
            zstd.wait(timeout=5)
        except subprocess.TimeoutExpired:
            zstd.kill()
        try:
            curl.wait(timeout=5)
        except subprocess.TimeoutExpired:
            curl.kill()

    if len(reservoir) != int(args.sample_count):
        curl_error = (curl.stderr.read() if curl.stderr else b"").decode("utf-8", "replace")
        zstd_error = (zstd.stderr.read() if zstd.stderr else b"").decode("utf-8", "replace")
        raise RuntimeError(
            f"only collected {len(reservoir)}/{args.sample_count} rows; "
            f"raw={raw_seen}, valid={valid_seen}; curl={curl_error[-500:]}; zstd={zstd_error[-500:]}"
        )

    reservoir.sort(key=lambda row: int(row["original_index"]))
    for sample_index, row in enumerate(reservoir):
        row["sample_index"] = sample_index
    digest = hashlib.sha256(
        "\n".join(row["text_sha256"] for row in reservoir).encode("ascii")
    ).hexdigest()
    atomic_jsonl(manifest_path, reservoir)
    metadata = {
        "source_id": "the_pile_train",
        "dataset_name": "monology/pile-uncopyrighted",
        "split": "train",
        "source_url": args.source_url,
        "source_shard": "train/00.jsonl.zst",
        "source_shard_linked_size_bytes": 11_152_428_427,
        "sampling_method": (
            "deterministic reservoir sample from the first scan_limit eligible records "
            "of train shard 00, streamed through an HTTP byte range"
        ),
        "sampling_scope_caveat": (
            "This is a train-split proxy, not a uniform sample over all 30 train shards, "
            "and the uncopyrighted mirror is not guaranteed identical to the exact Pythia training mixture."
        ),
        "sample_count": len(reservoir),
        "scan_limit": int(args.scan_limit),
        "raw_records_scanned": raw_seen,
        "eligible_records_scanned": valid_seen,
        "min_chars": int(args.min_chars),
        "range_bytes_cap": int(args.range_bytes),
        "seed": int(args.seed),
        "text_digest_sha256": digest,
        "manifest_path": str(manifest_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(metadata_path, metadata)
    atomic_json(
        root / "manifests" / "sources.json",
        {"sources": [{**metadata, "target_sample_count": len(reservoir)}], "source_count": 1},
    )
    print(json.dumps({"status": "complete", **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
