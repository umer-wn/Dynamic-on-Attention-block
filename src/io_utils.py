from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import yaml
from src.experiment_io import atomic_json, atomic_jsonl, read_jsonl, save_manifest, sha256_file


PathLike = Union[str, os.PathLike]


def load_config(path: PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_storage_env(config: dict[str, Any]) -> None:
    cache_dir = config.get("cache_dir")
    if cache_dir:
        cache_path = ensure_dir(cache_dir)
        os.environ.setdefault("HF_HOME", str(cache_path))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_path / "hub"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache_path / "datasets"))
    if config.get("output_dir"):
        ensure_dir(config["output_dir"])
    if config.get("offline"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def base_metadata(config: dict[str, Any], model: str, revision: str, tokenizer: str, sequence_length: Optional[int]) -> dict[str, Any]:
    return {
        "experiment": config.get("experiment_name"),
        "model": model,
        "checkpoint": revision,
        "tokenizer": tokenizer,
        "dataset": config.get("dataset", {}).get("name"),
        "dataset_config": config.get("dataset", {}).get("config"),
        "dataset_split": config.get("dataset", {}).get("split"),
        "sequence_length": sequence_length,
        "dtype": config.get("dtype", "float32"),
        "device": config.get("device", "auto"),
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_jsonl(path: PathLike, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(Path(path).parent)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def append_jsonl(path: PathLike, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(Path(path).parent)
    count = 0
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def should_skip(path: PathLike, skip_existing: bool) -> bool:
    return skip_existing and Path(path).exists() and Path(path).stat().st_size > 0
