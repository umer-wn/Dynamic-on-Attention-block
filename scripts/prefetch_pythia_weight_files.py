#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from huggingface_hub import hf_hub_download


REQUIRED_METADATA = [
    ".gitattributes",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]
WEIGHT_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def have_file(repo: str, revision: str, cache_dir: str, filename: str) -> bool:
    try:
        hf_hub_download(
            repo_id=repo,
            revision=revision,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def download_file(
    repo: str,
    revision: str,
    cache_dir: str,
    filename: str,
    attempts: int,
    retry_sleep: float,
) -> dict:
    result = {"filename": filename, "status": "running", "attempts": []}
    if have_file(repo, revision, cache_dir, filename):
        result.update({"status": "complete", "cache_hit": True})
        return result
    last_error: BaseException | None = None
    for attempt_index in range(1, attempts + 1):
        attempt = {"attempt": attempt_index, "started_unix": time.time(), "status": "running"}
        result["attempts"].append(attempt)
        started = time.perf_counter()
        try:
            path = hf_hub_download(
                repo_id=repo,
                revision=revision,
                filename=filename,
                cache_dir=cache_dir,
                resume_download=True,
            )
            attempt.update({"status": "complete", "seconds": time.perf_counter() - started})
            result.update({"status": "complete", "cache_hit": False, "path": path})
            return result
        except Exception as error:
            last_error = error
            attempt.update(
                {
                    "status": "failed",
                    "seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if attempt_index < attempts:
                time.sleep(retry_sleep)
    result.update(
        {
            "status": "failed",
            "error_type": type(last_error).__name__ if last_error else "UnknownError",
            "error": str(last_error) if last_error else "unknown error",
        }
    )
    return result


def download_revision(args: argparse.Namespace, revision: str) -> tuple[dict, bool]:
    status_path = Path(args.status_dir) / f"{revision}.json"
    record: dict = {
        "repo": args.repo,
        "revision": revision,
        "status": "running",
        "started_unix": time.time(),
        "mode": "file_level_weight_prefetch",
        "attempts_requested": args.attempts,
        "required_metadata": REQUIRED_METADATA,
        "weight_candidates": WEIGHT_CANDIDATES,
        "files": [],
    }
    atomic_json(status_path, record)
    started = time.perf_counter()
    try:
        for filename in REQUIRED_METADATA:
            file_result = download_file(
                args.repo, revision, args.cache_dir, filename, args.attempts, args.retry_sleep
            )
            record["files"].append(file_result)
            atomic_json(status_path, record)
            if file_result["status"] != "complete" and filename == "config.json":
                raise RuntimeError(f"required metadata failed: {filename}: {file_result.get('error')}")

        weight_result = None
        for filename in WEIGHT_CANDIDATES:
            file_result = download_file(
                args.repo, revision, args.cache_dir, filename, args.attempts, args.retry_sleep
            )
            record["files"].append(file_result)
            atomic_json(status_path, record)
            if file_result["status"] == "complete":
                weight_result = file_result
                break
            if file_result.get("error_type") in {"EntryNotFoundError", "RemoteEntryNotFoundError"}:
                continue
        if weight_result is None:
            raise RuntimeError("no weight candidate could be downloaded")

        record.update(
            {
                "status": "complete",
                "weight_file": weight_result["filename"],
                "seconds": time.perf_counter() - started,
            }
        )
        failed = False
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "seconds": time.perf_counter() - started,
            }
        )
        failed = True
    atomic_json(status_path, record)
    return record, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("revisions", nargs="+")
    parser.add_argument("--repo", default="EleutherAI/pythia-70m")
    parser.add_argument("--cache-dir", default="/home/luohaoming/model_feature_cache/hf_cache")
    parser.add_argument(
        "--status-dir",
        default="/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/status/prefetch",
    )
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    args = parser.parse_args()

    failures = 0
    for revision in args.revisions:
        record, failed = download_revision(args, revision)
        failures += int(failed)
        print(json.dumps(record), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
