#!/usr/bin/env python
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "/home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan"
PILE_URLS = {
    "validation": "https://hf-mirror.com/datasets/monology/pile-uncopyrighted/resolve/main/val.jsonl.zst",
    "test": "https://hf-mirror.com/datasets/monology/pile-uncopyrighted/resolve/main/test.jsonl.zst",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_text(row: dict[str, Any]) -> str | None:
    for key in ("text", "content", "document", "raw_content"):
        if key in row and row[key] is not None:
            return str(row[key])
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) > 20:
            return value
    return None


def valid_text(row: dict[str, Any], min_chars: int) -> str | None:
    text = extract_text(row)
    if text is None:
        return None
    text = text.strip()
    if len(text) < min_chars:
        return None
    return text


def stream_tail_window(
    url: str,
    tail_window: int,
    min_chars: int,
    status_path: Path,
    progress_every: int,
) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    command = f"curl -k -L --fail --retry 10 --connect-timeout 30 --max-time 0 {shlex.quote(url)} | zstd -dc"
    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    tail: collections.deque[tuple[int, str]] = collections.deque(maxlen=int(tail_window))
    valid_seen = 0
    raw_seen = 0
    started = time.time()
    last_status = 0
    try:
        for raw_index, line in enumerate(process.stdout):
            raw_seen = raw_index + 1
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = valid_text(payload, min_chars)
            if text is None:
                continue
            valid_seen += 1
            tail.append((raw_index, text))
            if valid_seen - last_status >= progress_every:
                last_status = valid_seen
                atomic_json(
                    status_path,
                    {
                        "status": "running",
                        "url": url,
                        "raw_seen": raw_seen,
                        "valid_seen": valid_seen,
                        "tail_window_current": len(tail),
                        "tail_window_target": int(tail_window),
                        "elapsed_seconds": time.time() - started,
                    },
                )
    finally:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    stderr = process.stderr.read() if process.stderr is not None else ""
    if process.returncode not in (0, None):
        raise RuntimeError(f"stream failed returncode={process.returncode} stderr_tail={stderr[-2000:]}")
    meta = {
        "raw_seen": raw_seen,
        "valid_seen": valid_seen,
        "tail_window_current": len(tail),
        "tail_window_target": int(tail_window),
        "elapsed_seconds": time.time() - started,
    }
    return list(tail), meta


def build_manifest(args: argparse.Namespace, split: str) -> dict[str, Any]:
    root = Path(args.root)
    url = PILE_URLS[split]
    source_id = f"the_pile_{split}_tail_random"
    status_path = root / "status" / f"{source_id}_prepare_status.json"
    tail, meta = stream_tail_window(
        url=url,
        tail_window=args.tail_window,
        min_chars=args.min_chars,
        status_path=status_path,
        progress_every=args.progress_every,
    )
    if len(tail) < args.sample_count:
        raise RuntimeError(f"tail window has only {len(tail)} valid rows, need {args.sample_count}")
    rng = random.Random(int(args.seed) + (0 if split == "validation" else 10_000))
    sampled = sorted(rng.sample(tail, int(args.sample_count)), key=lambda item: item[0])
    rows = []
    for sample_index, (original_index, text) in enumerate(sampled):
        rows.append(
            {
                "source_id": source_id,
                "dataset_name": "EleutherAI/pile",
                "dataset_mirror": "monology/pile-uncopyrighted",
                "config": None,
                "split": split,
                "sample_index": sample_index,
                "original_index": int(original_index),
                "tail_window_rank": None,
                "tail_window_size": int(args.tail_window),
                "sampling_mode": "tail_window_random",
                "text": text,
                "text_sha256": text_hash(text),
            }
        )
    manifest_path = root / "manifests" / f"{source_id}.jsonl"
    atomic_jsonl(manifest_path, rows)
    digest = hashlib.sha256("\n\0\n".join(row["text"] for row in rows).encode("utf-8")).hexdigest()
    metadata = {
        "source_id": source_id,
        "dataset_name": "EleutherAI/pile",
        "dataset_mirror": "monology/pile-uncopyrighted",
        "split": split,
        "source_url": url,
        "sample_count": len(rows),
        "target_sample_count": int(args.sample_count),
        "seed": int(args.seed),
        "tail_window": int(args.tail_window),
        "min_chars": int(args.min_chars),
        "sampling_note": "tail-window random sample: stream full compressed split, keep only final valid documents, then sample with fixed seed; no full corpus is stored",
        "manifest_path": str(manifest_path),
        "text_digest_sha256": digest,
        "created_unix": time.time(),
        **meta,
    }
    atomic_json(root / "manifests" / f"{source_id}.metadata.json", metadata)
    atomic_json(status_path, {"status": "complete", **metadata})
    return metadata


def update_sources(root: Path, new_sources: list[dict[str, Any]]) -> None:
    sources_path = root / "manifests" / "sources.json"
    if sources_path.exists():
        payload = json.loads(sources_path.read_text(encoding="utf-8"))
        sources = payload.get("sources", [])
    else:
        sources = []
    by_id = {source["source_id"]: source for source in sources}
    for source in new_sources:
        by_id[source["source_id"]] = source
    updated = list(by_id.values())
    atomic_json(sources_path, {"sources": updated, "source_count": len(updated)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["validation", "test"])
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--tail-window", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=50000)
    args = parser.parse_args()

    root = Path(args.root)
    outputs = [build_manifest(args, split) for split in args.splits]
    update_sources(root, outputs)
    print(json.dumps({"status": "complete", "sources": [o["source_id"] for o in outputs]}, indent=2))


if __name__ == "__main__":
    main()
