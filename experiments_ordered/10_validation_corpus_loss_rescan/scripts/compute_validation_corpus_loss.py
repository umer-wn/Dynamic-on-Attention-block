#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpoint_utils import checkpoint_step
from src.experiment_io import atomic_json, atomic_jsonl, read_jsonl, sha256_file
from src.model_utils import load_model_and_tokenizer


MODEL_NAME = "EleutherAI/pythia-70m"
DEFAULT_ROOT = "/home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan"
DEFAULT_CACHE = "/home/luohaoming/model_feature_cache/hf_cache"


def compute_loss_rows(model, tokenizer, rows: list[dict], device: torch.device, sequence_length: int, batch_size: int) -> list[dict]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs: list[dict] = []
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_start in range(0, len(rows), int(batch_size)):
            batch_rows = rows[batch_start: batch_start + int(batch_size)]
            texts = [str(row["text"]) for row in batch_rows]
            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=int(sequence_length),
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = input_ids[:, 1:].contiguous()
            valid = attention_mask[:, 1:].bool()
            token_nll = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                reduction="none",
            ).view(shifted_labels.shape)
            for local_index, source_row in enumerate(batch_rows):
                predicted_tokens = int(valid[local_index].sum().item())
                nll_sum = float(token_nll[local_index][valid[local_index]].sum().cpu()) if predicted_tokens else 0.0
                mean_loss = nll_sum / predicted_tokens if predicted_tokens else None
                out = {
                    **{k: source_row.get(k) for k in ("source_id", "dataset_name", "config", "split", "sample_index", "original_index", "text_sha256")},
                    "nll_sum": nll_sum,
                    "predicted_token_count": predicted_tokens,
                    "mean_loss": mean_loss,
                }
                outputs.append(out)
                total_nll += nll_sum
                total_tokens += predicted_tokens
    loss = total_nll / max(total_tokens, 1)
    outputs.append(
        {
            "sample_index": "aggregate",
            "nll_sum": total_nll,
            "predicted_token_count": total_tokens,
            "mean_loss": loss,
            "token_weighted_loss": loss,
            "token_weighted_perplexity": math.exp(min(loss, 20.0)),
        }
    )
    return outputs


def coarse_revisions() -> list[str]:
    return [f"step{s}" for s in range(1000, 98000, 4000)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--source-id")
    parser.add_argument("--revision")
    parser.add_argument("--checkpoint-limit", type=int)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--loss-batch-size", type=int, default=16)
    parser.add_argument("--allow-network-model", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    sources = json.loads((root / "manifests" / "sources.json").read_text(encoding="utf-8"))["sources"]
    if args.source_id:
        sources = [source for source in sources if source["source_id"] == args.source_id]
    revisions = [args.revision] if args.revision else ["step0", *coarse_revisions(), "step16000", "step100000"]
    # de-duplicate preserving order
    seen = set()
    revisions = [rev for rev in revisions if not (rev in seen or seen.add(rev))]
    if args.checkpoint_limit is not None:
        revisions = revisions[: int(args.checkpoint_limit)]
    os.environ.setdefault("HF_HOME", args.cache_dir)
    for source in sources:
        source_id = source["source_id"]
        manifest_path = Path(source["manifest_path"])
        manifest_rows = read_jsonl(manifest_path)
        if args.sample_limit is not None:
            manifest_rows = manifest_rows[: int(args.sample_limit)]
        manifest_digest = source.get("text_digest_sha256") or sha256_file(manifest_path)
        for revision in revisions:
            out_dir = root / "raw" / source_id / revision
            complete = out_dir / "loss_complete.json"
            if complete.exists():
                print(json.dumps({"source_id": source_id, "revision": revision, "status": "already_complete"}))
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            model, tokenizer, device = load_model_and_tokenizer(
                MODEL_NAME,
                revision,
                MODEL_NAME,
                "float32",
                "cuda",
                cache_dir=args.cache_dir,
                attn_implementation="eager",
                local_files_only=not args.allow_network_model,
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            loss_rows = compute_loss_rows(model, tokenizer, manifest_rows, device, args.sequence_length, args.loss_batch_size)
            common = {
                "experiment": "validation_corpus_loss_rescan",
                "model": MODEL_NAME,
                "checkpoint": revision,
                "training_step": checkpoint_step(revision),
                "source_id": source_id,
                "dataset_name": source.get("dataset_name"),
                "config": source.get("config"),
                "split": source.get("split"),
                "manifest_path": str(manifest_path),
                "manifest_text_digest_sha256": manifest_digest,
                "sample_count": len(manifest_rows),
                "sequence_length": args.sequence_length,
                "loss_batch_size": args.loss_batch_size,
                "local_files_only": not args.allow_network_model,
            }
            atomic_jsonl(out_dir / "loss.jsonl", [{**common, **row} for row in loss_rows])
            aggregate = next(row for row in loss_rows if str(row.get("sample_index")) == "aggregate")
            completion = {
                "status": "complete",
                **common,
                "token_weighted_loss": aggregate["token_weighted_loss"],
                "token_weighted_perplexity": aggregate["token_weighted_perplexity"],
                "predicted_token_count": aggregate["predicted_token_count"],
                "elapsed_seconds": time.perf_counter() - started,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            atomic_json(out_dir / "loss_complete.json", completion)
            print(json.dumps(completion), flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
