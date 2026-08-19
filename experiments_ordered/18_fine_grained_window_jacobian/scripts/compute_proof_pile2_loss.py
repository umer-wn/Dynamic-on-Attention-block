#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts._bootstrap import require_packages

require_packages(["huggingface_hub", "torch", "transformers"])

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "EleutherAI/pythia-70m"
DATASET_NAME = "EleutherAI/proof-pile-2"
CHECKPOINTS = [
    "step0",
    "step2000",
    "step3000",
    "step4000",
    "step5000",
    "step7000",
    "step8000",
    "step9000",
    "step10000",
    "step13000",
    "step21000",
    "step25000",
    "step29000",
    "step33000",
    "step37000",
    "step41000",
    "step53000",
    "step57000",
    "step61000",
]
SUBSET_COUNTS = {
    "arxiv": 270,
    "open-web-math": 140,
    "algebraic-stack": 102,
}
APPENDED_LOSS_COLUMNS = ("proof_pile2_test_loss",)
CHECKPOINT_ONLY_COLUMNS = (
    "proof_pile2_test_perplexity",
    "proof_pile2_test_predicted_tokens",
    "proof_pile2_test_sample_count",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download_test_splits(dataset_root: Path, revision: str | None) -> tuple[Path, str]:
    api = HfApi()
    resolved_revision = revision or api.dataset_info(DATASET_NAME).sha
    local_dir = dataset_root / "dataset"
    snapshot_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        revision=resolved_revision,
        local_dir=local_dir,
        allow_patterns=[
            "README.md",
            "arxiv/test/*.jsonl.zst",
            "open-web-math/test/*.jsonl.zst",
            "algebraic-stack/test/*.jsonl.zst",
        ],
    )
    return local_dir, resolved_revision


def iter_zstd_json(path: Path):
    process = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            if line.strip():
                yield json.loads(line)
    finally:
        _, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f"zstd failed for {path}: {stderr}")


def reservoir_sample(
    files: list[Path],
    count: int,
    seed: int,
    dataset_dir: Path,
) -> list[dict]:
    generator = random.Random(seed)
    reservoir: list[dict] = []
    eligible_index = 0
    original_index = 0
    for path in files:
        for row in iter_zstd_json(path):
            text = str(row.get("text", ""))
            if len(text) < 256:
                original_index += 1
                continue
            item = {
                "source_file": str(path.relative_to(dataset_dir)),
                "original_index": original_index,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            eligible_index += 1
            if len(reservoir) < count:
                reservoir.append(item)
            else:
                replacement = generator.randrange(eligible_index)
                if replacement < count:
                    reservoir[replacement] = item
            original_index += 1
    if len(reservoir) != count:
        raise RuntimeError(
            f"requested {count} samples but only found {len(reservoir)} in {files}"
        )
    return reservoir


def build_manifest(
    dataset_dir: Path,
    manifest_path: Path,
    tokenizer,
    sequence_length: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for subset_index, (subset, count) in enumerate(SUBSET_COUNTS.items()):
        files = sorted((dataset_dir / subset / "test").glob("*.jsonl.zst"))
        if not files:
            raise FileNotFoundError(f"no test files found for {subset}")
        selected = reservoir_sample(
            files, count, seed + subset_index * 100_003, dataset_dir
        )
        for item in selected:
            input_ids = tokenizer(
                item.pop("text"),
                add_special_tokens=False,
                truncation=True,
                max_length=sequence_length,
            )["input_ids"]
            if len(input_ids) < 2:
                raise RuntimeError(f"sample tokenized to fewer than 2 tokens: {item}")
            rows.append(
                {
                    "sample_index": len(rows),
                    "subset": subset,
                    **item,
                    "input_ids": input_ids,
                }
            )
    atomic_jsonl(manifest_path, rows)
    return rows


def load_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_model(checkpoint: str, cache_dir: Path, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=checkpoint,
        cache_dir=str(cache_dir),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def evaluate_loss(
    model,
    rows: list[dict],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, int]:
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            width = max(len(row["input_ids"]) for row in batch)
            input_ids = torch.full(
                (len(batch), width),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for index, row in enumerate(batch):
                values = torch.tensor(
                    row["input_ids"], dtype=torch.long, device=device
                )
                input_ids[index, : len(values)] = values
                attention_mask[index, : len(values)] = 1
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits.float()
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = input_ids[:, 1:].contiguous()
            valid = attention_mask[:, 1:].bool()
            losses = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                reduction="none",
            ).view_as(shifted_labels)
            total_nll += float(losses[valid].sum().cpu())
            total_tokens += int(valid.sum().item())
    return total_nll / total_tokens, total_tokens


def append_loss_columns(processed_dir: Path, results: list[dict]) -> None:
    by_checkpoint = {row["checkpoint"]: row for row in results}
    targets = [
        *sorted((processed_dir / "raw_parts").glob("step*.csv")),
        processed_dir / "jacobian_fine_grained_raw.csv",
        processed_dir / "jacobian_fine_grained_8tokens.csv",
    ]
    for path in targets:
        rows = read_csv(path)
        for row in rows:
            loss = by_checkpoint[row["checkpoint"]]
            for column in CHECKPOINT_ONLY_COLUMNS:
                row.pop(column, None)
            row.update(
                {column: loss[column] for column in APPENDED_LOSS_COLUMNS}
            )
        atomic_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/luohaoming/proof_pile2"),
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("/home/luohaoming/model_feature_cache/hf_cache"),
    )
    parser.add_argument("--report-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--sample-seed", type=int, default=1818)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    args = parser.parse_args()

    os.environ["HF_HOME"] = str(args.dataset_root / "hf_cache")
    os.environ["HF_DATASETS_CACHE"] = str(args.dataset_root / "datasets_cache")
    dataset_dir, dataset_revision = download_test_splits(
        args.dataset_root, args.dataset_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision="step0",
        cache_dir=str(args.model_cache),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    manifest_path = (
        args.dataset_root
        / "manifests"
        / f"proof_pile2_test_{sum(SUBSET_COUNTS.values())}_seed{args.sample_seed}.jsonl"
    )
    manifest_rows = (
        load_manifest(manifest_path)
        if manifest_path.exists()
        else build_manifest(
            dataset_dir,
            manifest_path,
            tokenizer,
            args.sequence_length,
            args.sample_seed,
        )
    )
    if len(manifest_rows) != sum(SUBSET_COUNTS.values()):
        raise RuntimeError(f"unexpected manifest size: {len(manifest_rows)}")

    output = args.report_root / "processed/proof_pile2_test_loss_by_checkpoint.csv"
    results = read_csv(output) if output.exists() else []
    done = {row["checkpoint"] for row in results}
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        if checkpoint in done:
            continue
        started = time.perf_counter()
        model = load_model(checkpoint, args.model_cache, args.device)
        loss, predicted_tokens = evaluate_loss(
            model,
            manifest_rows,
            int(tokenizer.pad_token_id),
            args.device,
            args.batch_size,
        )
        row = {
            "checkpoint": checkpoint,
            "training_step": int(checkpoint.removeprefix("step")),
            "dataset": DATASET_NAME,
            "dataset_revision": dataset_revision,
            "split": "test",
            "subset_sample_counts": json.dumps(SUBSET_COUNTS, sort_keys=True),
            "sample_seed": args.sample_seed,
            "sample_count": len(manifest_rows),
            "sequence_length": args.sequence_length,
            "manifest_path": str(manifest_path),
            "proof_pile2_test_loss": loss,
            "proof_pile2_test_perplexity": math.exp(min(loss, 20.0)),
            "proof_pile2_test_predicted_tokens": predicted_tokens,
            "proof_pile2_test_sample_count": len(manifest_rows),
            "runtime_seconds": time.perf_counter() - started,
        }
        results.append(row)
        atomic_csv(output, results)
        print(
            json.dumps(
                {
                    "checkpoint": checkpoint,
                    "completed": checkpoint_index + 1,
                    "total": len(CHECKPOINTS),
                    "loss": loss,
                    "perplexity": row["proof_pile2_test_perplexity"],
                    "predicted_tokens": predicted_tokens,
                    "seconds": row["runtime_seconds"],
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    results = sorted(results, key=lambda row: int(row["training_step"]))
    atomic_csv(output, results)
    append_loss_columns(args.report_root / "processed", results)
    atomic_json(
        args.dataset_root / "manifests/proof_pile2_test_loss_metadata.json",
        {
            "dataset": DATASET_NAME,
            "dataset_revision": dataset_revision,
            "downloaded_splits": {
                subset: "test" for subset in SUBSET_COUNTS
            },
            "subset_sample_counts": SUBSET_COUNTS,
            "sample_seed": args.sample_seed,
            "sequence_length": args.sequence_length,
            "manifest_path": str(manifest_path),
            "checkpoints": CHECKPOINTS,
            "result_path": str(output),
        },
    )


if __name__ == "__main__":
    main()
