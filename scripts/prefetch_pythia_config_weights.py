#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download


REQUIRED_FILES = ["config.json"]
WEIGHT_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def download_one(args: argparse.Namespace, revision: str, filename: str) -> dict:
    result = {"filename": filename, "status": "running", "attempts": []}
    for attempt_index in range(1, args.attempts + 1):
        started = time.perf_counter()
        attempt = {"attempt": attempt_index, "status": "running", "started_unix": time.time()}
        result["attempts"].append(attempt)
        try:
            path = hf_hub_download(
                repo_id=args.repo,
                revision=revision,
                filename=filename,
                cache_dir=args.cache_dir,
                local_files_only=False,
            )
            attempt.update({"status": "complete", "seconds": time.perf_counter() - started})
            result.update({"status": "complete", "path": path})
            return result
        except Exception as exc:
            attempt.update(
                {
                    "status": "failed",
                    "seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if attempt_index < args.attempts:
                time.sleep(args.retry_sleep)
    result.update(
        {
            "status": "failed",
            "error_type": result["attempts"][-1].get("error_type"),
            "error": result["attempts"][-1].get("error"),
        }
    )
    return result


def download_revision(args: argparse.Namespace, revision: str) -> bool:
    status_path = Path(args.status_dir) / f"{revision}.json"
    record = {
        "repo": args.repo,
        "revision": revision,
        "status": "running",
        "mode": "config_weight_only_prefetch",
        "required_files": REQUIRED_FILES,
        "weight_candidates": WEIGHT_CANDIDATES,
        "files": [],
        "started_unix": time.time(),
    }
    atomic_json(status_path, record)
    started = time.perf_counter()
    try:
        for filename in REQUIRED_FILES:
            item = download_one(args, revision, filename)
            record["files"].append(item)
            atomic_json(status_path, record)
            if item["status"] != "complete":
                raise RuntimeError(f"required file failed: {filename}: {item.get('error')}")

        weight = None
        for filename in WEIGHT_CANDIDATES:
            item = download_one(args, revision, filename)
            record["files"].append(item)
            atomic_json(status_path, record)
            if item["status"] == "complete":
                weight = item
                break
        if weight is None:
            raise RuntimeError("no weight candidate could be downloaded")

        record.update(
            {
                "status": "complete",
                "weight_file": weight["filename"],
                "seconds": time.perf_counter() - started,
            }
        )
        failed = False
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "seconds": time.perf_counter() - started,
            }
        )
        failed = True
    atomic_json(status_path, record)
    print(json.dumps(record), flush=True)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("revisions", nargs="+")
    parser.add_argument("--repo", default="EleutherAI/pythia-70m")
    parser.add_argument("--cache-dir", default="/home/luohaoming/model_feature_cache/hf_cache")
    parser.add_argument("--status-dir", required=True)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    args = parser.parse_args()

    failures = 0
    for revision in args.revisions:
        failures += int(download_revision(args, revision))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
