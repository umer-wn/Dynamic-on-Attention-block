#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import require_packages

require_packages(["torch", "transformers", "datasets", "tqdm", "yaml"])

import torch
from tqdm import tqdm

from src.data_utils import load_text_samples, tokenize_samples
from src.io_utils import base_metadata, load_config, sanitize_name, setup_storage_env, should_skip, write_jsonl
from src.model_utils import cuda_memory, iter_model_revisions, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_storage_env(config)
    texts = load_text_samples(config)
    raw_dir = Path(config.get("output_dir", "results")) / "raw"

    for model_cfg in config["models"]:
        name = model_cfg["name"]
        tokenizer_name = model_cfg.get("tokenizer", name)
        for revision in iter_model_revisions(model_cfg):
            model, tokenizer, device = load_model_and_tokenizer(
                name,
                revision,
                tokenizer_name,
                config.get("dtype", "float32"),
                config.get("device", "auto"),
                config.get("cache_dir"),
                config.get("attn_implementation"),
                bool(config.get("offline", False)),
            )
            for seq_len in config["dataset"].get("sequence_lengths", [128]):
                out = raw_dir / f"{config['experiment_name']}__{sanitize_name(name)}__{revision}__seq{seq_len}__perplexity.jsonl"
                if should_skip(out, bool(config.get("skip_existing", True))):
                    print(f"skip existing {out}")
                    continue
                batches = tokenize_samples(tokenizer, texts, int(seq_len), int(config["dataset"].get("num_samples", len(texts))))
                metadata = base_metadata(config, name, revision, tokenizer.name_or_path, int(seq_len))
                rows = []
                losses = []
                nll_sum = 0.0
                predicted_token_count = 0
                for batch_idx, batch in enumerate(tqdm(batches, desc=f"ppl {name}@{revision} seq{seq_len}")):
                    sample_index = batch.pop("sample_index")
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch["input_ids"].clone()
                    labels[batch["attention_mask"] == 0] = -100
                    with torch.no_grad():
                        out_obj = model(**batch, labels=labels, use_cache=False)
                    loss = float(out_obj.loss.detach().cpu())
                    token_count = max(0, int(batch["attention_mask"].sum().item()) - 1)
                    sample_nll = loss * token_count
                    losses.append(loss)
                    nll_sum += sample_nll
                    predicted_token_count += token_count
                    rows.append({
                        **metadata,
                        "batch_index": batch_idx,
                        "sample_index": sample_index,
                        "loss": loss,
                        "perplexity": math.exp(min(loss, 20.0)),
                        "predicted_token_count": token_count,
                        "nll_sum": sample_nll,
                        "cuda_memory": cuda_memory(),
                    })
                mean_loss = sum(losses) / max(len(losses), 1)
                token_weighted_loss = nll_sum / max(predicted_token_count, 1)
                rows.append({
                    **metadata,
                    "batch_index": "mean",
                    "sample_index": "mean",
                    "loss": mean_loss,
                    "perplexity": math.exp(min(mean_loss, 20.0)),
                    "token_weighted_loss": token_weighted_loss,
                    "token_weighted_perplexity": math.exp(min(token_weighted_loss, 20.0)),
                    "predicted_token_count": predicted_token_count,
                    "nll_sum": nll_sum,
                    "cuda_memory": cuda_memory(),
                })
                write_jsonl(out, rows)
                print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
