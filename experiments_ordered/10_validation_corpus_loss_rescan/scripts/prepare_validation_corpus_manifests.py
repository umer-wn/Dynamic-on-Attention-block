#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_io import atomic_json, atomic_jsonl


DEFAULT_ROOT = "/home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan"
DEFAULT_CACHE = "/home/luohaoming/model_feature_cache/hf_cache"
PALOMA_PATTERNS = [
    ("pile", re.compile("pile", re.I)),
    ("dolma", re.compile("dolma", re.I)),
    ("redpajama", re.compile("redpajama|red_pajama", re.I)),
    ("c4_common_crawl", re.compile("c4|common.?crawl", re.I)),
    ("wiki", re.compile("wiki|wikitext", re.I)),
    ("ptb", re.compile("ptb|penn", re.I)),
    ("code", re.compile("github|code|python", re.I)),
    ("reddit", re.compile("reddit", re.I)),
]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def valid_text(value: Any, min_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) < min_chars:
        return None
    return text


def extract_text(row: dict[str, Any]) -> str | None:
    for key in ("text", "content", "document", "raw_content"):
        if key in row and row[key] is not None:
            return str(row[key])
    for value in row.values():
        if isinstance(value, str) and len(value.strip()) > 20:
            return value
    return None


def iter_stream(dataset: Iterable[dict[str, Any]], sample_count: int, min_chars: int) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for original_index, row in enumerate(dataset):
        text = valid_text(extract_text(dict(row)), min_chars)
        if text is None:
            continue
        rows.append((original_index, text))
        if len(rows) >= sample_count:
            break
    return rows


PILE_URLS = {
    "validation": "https://hf-mirror.com/datasets/monology/pile-uncopyrighted/resolve/main/val.jsonl.zst",
    "test": "https://hf-mirror.com/datasets/monology/pile-uncopyrighted/resolve/main/test.jsonl.zst",
}


def iter_jsonl_zst_url(url: str, sample_count: int, min_chars: int, seed: int, scan_limit: int) -> list[tuple[int, str]]:
    """Stream .jsonl.zst and reservoir-sample from the first scan_limit valid texts."""
    command = f"curl -k -L --fail --retry 5 --connect-timeout 30 --max-time 3600 {shlex.quote(url)} | zstd -dc"
    process = subprocess.Popen(["bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert process.stdout is not None
    rng = random.Random(int(seed))
    reservoir: list[tuple[int, str]] = []
    valid_seen = 0
    try:
        for original_index, line in enumerate(process.stdout):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = valid_text(extract_text(payload), min_chars)
            if text is None:
                continue
            valid_seen += 1
            item = (original_index, text)
            if len(reservoir) < sample_count:
                reservoir.append(item)
            else:
                j = rng.randrange(valid_seen)
                if j < sample_count:
                    reservoir[j] = item
            if valid_seen >= scan_limit:
                process.terminate()
                break
    finally:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    if len(reservoir) < sample_count:
        stderr = process.stderr.read() if process.stderr is not None else ""
        if process.returncode not in (0, -15, 141, None):
            raise RuntimeError(f"failed streaming {url}: returncode={process.returncode} stderr={stderr[-1000:]}")
    return sorted(reservoir, key=lambda item: item[0])


def write_source_manifest(root: Path, source: dict[str, Any], sampled: list[tuple[int, str]]) -> dict[str, Any]:
    source_id = source["source_id"]
    rows = []
    for sample_index, (original_index, text) in enumerate(sampled):
        rows.append(
            {
                "source_id": source_id,
                "dataset_name": source["dataset_name"],
                "dataset_mirror": source.get("dataset_mirror"),
                "config": source.get("config"),
                "split": source["split"],
                "sample_index": sample_index,
                "original_index": int(original_index),
                "text": text,
                "text_sha256": text_hash(text),
            }
        )
    manifest_path = root / "manifests" / f"{source_id}.jsonl"
    atomic_jsonl(manifest_path, rows)
    digest = hashlib.sha256("\n\0\n".join(row["text"] for row in rows).encode("utf-8")).hexdigest()
    meta = {
        **source,
        "manifest_path": str(manifest_path),
        "sample_count": len(rows),
        "text_digest_sha256": digest,
        "created_unix": time.time(),
    }
    atomic_json(root / "manifests" / f"{source_id}.metadata.json", meta)
    return meta


def prepare_the_pile(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for split, url in PILE_URLS.items():
        source = {
            "source_id": f"the_pile_{split}",
            "dataset_name": "EleutherAI/pile",
            "dataset_mirror": "monology/pile-uncopyrighted",
            "config": None,
            "split": split,
            "source_url": url,
            "target_sample_count": args.sample_count,
            "seed": args.seed,
            "sampling_note": "deterministic reservoir sample from monology/pile-uncopyrighted jsonl.zst mirror via hf-mirror.com because EleutherAI/pile uses an unsupported dataset script and the-eye official URL is unavailable",
        }
        sampled = iter_jsonl_zst_url(url, args.sample_count, args.min_chars, args.seed, max(args.shuffle_buffer, args.sample_count))
        outputs.append(write_source_manifest(root, source, sampled))
    return outputs


def select_paloma_configs(configs: list[str]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used: set[str] = set()
    for label, pattern in PALOMA_PATTERNS:
        for config in sorted(configs):
            if config in used:
                continue
            if pattern.search(config):
                selected.append({"label": label, "config": config})
                used.add(config)
                break
    if len(selected) < 4:
        for config in sorted(configs):
            if config in used:
                continue
            selected.append({"label": f"fallback_{len(selected)}", "config": config})
            used.add(config)
            if len(selected) >= 4:
                break
    return selected[:8]


def prepare_paloma(root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    status_dir = root / "status"
    outputs: list[dict[str, Any]] = []
    try:
        configs = get_dataset_config_names("allenai/paloma")
    except Exception as exc:
        atomic_json(
            status_dir / "paloma_access_blocked.json",
            {"status": "blocked", "stage": "get_dataset_config_names", "error_type": type(exc).__name__, "error": str(exc)},
        )
        return outputs
    selected = select_paloma_configs(configs)
    atomic_json(root / "manifests" / "paloma_selected_configs.json", {"available_count": len(configs), "selected": selected})
    for item in selected:
        config = item["config"]
        try:
            splits = get_dataset_split_names("allenai/paloma", config)
        except Exception as exc:
            atomic_json(
                status_dir / f"paloma_{sanitize(config)}_blocked.json",
                {"status": "blocked", "stage": "get_dataset_split_names", "config": config, "error_type": type(exc).__name__, "error": str(exc)},
            )
            continue
        wanted_splits = [split for split in ("validation", "test", "val") if split in splits]
        if not wanted_splits and splits:
            wanted_splits = [splits[0]]
        for split in wanted_splits[:2]:
            source_id = f"paloma_{sanitize(config)}_{sanitize(split)}"
            source = {
                "source_id": source_id,
                "dataset_name": "allenai/paloma",
                "config": config,
                "paloma_label": item["label"],
                "split": split,
                "target_sample_count": args.sample_count,
                "seed": args.seed,
            }
            try:
                dataset = load_dataset(
                    "allenai/paloma",
                    config,
                    split=split,
                    streaming=True,
                    cache_dir=args.cache_dir,
                )
                dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
                sampled = iter_stream(dataset, args.sample_count, args.min_chars)
                outputs.append(write_source_manifest(root, source, sampled))
            except Exception as exc:
                atomic_json(
                    status_dir / f"{source_id}_blocked.json",
                    {"status": "blocked", "stage": "load_dataset", **source, "error_type": type(exc).__name__, "error": str(exc)},
                )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--skip-paloma", action="store_true")
    parser.add_argument("--only-pile", action="store_true")
    parser.add_argument("--skip-pile", action="store_true")
    parser.add_argument("--only-paloma", action="store_true")
    args = parser.parse_args()
    if args.sample_limit is not None:
        args.sample_count = int(args.sample_limit)
    root = Path(args.root)
    for sub in ["manifests", "raw", "processed", "figures", "logs", "status"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", args.cache_dir)
    all_meta: list[dict[str, Any]] = []
    if not args.skip_pile and not args.only_paloma:
        try:
            all_meta.extend(prepare_the_pile(root, args))
        except Exception as exc:
            atomic_json(root / "status" / "the_pile_blocked.json", {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)})
            raise
    if not args.skip_paloma and not args.only_pile:
        all_meta.extend(prepare_paloma(root, args))
    sources_path = root / "manifests" / "sources.json"
    if sources_path.exists():
        existing = json.loads(sources_path.read_text(encoding="utf-8")).get("sources", [])
    else:
        existing = []
    merged = {source["source_id"]: source for source in existing}
    for source in all_meta:
        merged[source["source_id"]] = source
    merged_sources = list(merged.values())
    atomic_json(sources_path, {"sources": merged_sources, "source_count": len(merged_sources)})
    print(json.dumps({"status": "complete", "new_source_count": len(all_meta), "source_count": len(merged_sources), "sources": [m["source_id"] for m in merged_sources]}, indent=2))


if __name__ == "__main__":
    main()
