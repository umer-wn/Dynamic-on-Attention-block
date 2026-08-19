#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import time
import traceback
from pathlib import Path

from huggingface_hub import snapshot_download


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def worker_schedule(max_workers: int, attempts: int) -> list[int]:
    first = max(1, max_workers)
    values = [first]
    if first > 2:
        values.append(max(2, int(math.ceil(first / 2))))
    values.append(1)
    schedule: list[int] = []
    for value in values:
        if value not in schedule:
            schedule.append(value)
    while len(schedule) < attempts:
        schedule.append(1)
    return schedule[:attempts]


def prefetch_one(args: argparse.Namespace, revision: str) -> tuple[dict, bool]:
    record = {
        "repo": args.repo,
        "revision": revision,
        "status": "running",
        "started_unix": time.time(),
        "max_workers": args.max_workers,
        "revision_workers": args.revision_workers,
        "attempts_requested": args.attempts,
        "allow_patterns": args.allow_patterns,
        "attempts": [],
    }
    status_path = Path(args.status_dir) / f"{revision}.json"
    atomic_json(status_path, record)
    started = time.perf_counter()
    failed = False
    try:
        cache_hit = True
        try:
            snapshot = snapshot_download(
                repo_id=args.repo,
                revision=revision,
                cache_dir=args.cache_dir,
                local_files_only=True,
                allow_patterns=args.allow_patterns,
            )
        except Exception:
            cache_hit = False
            snapshot = None
            last_error: BaseException | None = None
            for attempt_index, workers in enumerate(worker_schedule(args.max_workers, args.attempts), start=1):
                attempt = {
                    "attempt": attempt_index,
                    "max_workers": workers,
                    "started_unix": time.time(),
                    "status": "running",
                }
                record["attempts"].append(attempt)
                atomic_json(status_path, record)
                attempt_started = time.perf_counter()
                try:
                    snapshot = snapshot_download(
                        repo_id=args.repo,
                        revision=revision,
                        cache_dir=args.cache_dir,
                        max_workers=workers,
                        resume_download=True,
                        allow_patterns=args.allow_patterns,
                    )
                    attempt.update(
                        {
                            "status": "complete",
                            "seconds": time.perf_counter() - attempt_started,
                        }
                    )
                    atomic_json(status_path, record)
                    break
                except Exception as error:
                    last_error = error
                    attempt.update(
                        {
                            "status": "failed",
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "seconds": time.perf_counter() - attempt_started,
                        }
                    )
                    atomic_json(status_path, record)
                    if attempt_index < args.attempts:
                        time.sleep(args.retry_sleep)
            if snapshot is None:
                if last_error is None:
                    raise RuntimeError("snapshot_download failed without an exception")
                raise last_error
        record.update(
            {
                "status": "complete",
                "snapshot_path": snapshot,
                "snapshot_hash": Path(snapshot).name,
                "cache_hit": cache_hit,
                "seconds": time.perf_counter() - started,
            }
        )
    except Exception as error:
        failed = True
        record.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "seconds": time.perf_counter() - started,
            }
        )
    atomic_json(status_path, record)
    return record, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("revisions", nargs="+")
    parser.add_argument("--repo", default="EleutherAI/pythia-70m")
    parser.add_argument("--cache-dir", default="/home/luohaoming/model_feature_cache/hf_cache")
    parser.add_argument("--status-dir", default=(
        "/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/status/prefetch"
    ))
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--revision-workers",
        type=int,
        default=1,
        help="Number of checkpoint revisions to download concurrently.",
    )
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=20.0)
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional file allow-list for snapshot_download, e.g. config/tokenizer/model.safetensors only.",
    )
    args = parser.parse_args()
    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    revision_workers = max(1, min(args.revision_workers, len(args.revisions)))
    if revision_workers == 1:
        for revision in args.revisions:
            record, failed = prefetch_one(args, revision)
            failures += int(failed)
            print(json.dumps(record), flush=True)
    else:
        with ThreadPoolExecutor(max_workers=revision_workers) as executor:
            futures = {executor.submit(prefetch_one, args, revision): revision for revision in args.revisions}
            for future in as_completed(futures):
                record, failed = future.result()
                failures += int(failed)
                print(json.dumps(record), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
