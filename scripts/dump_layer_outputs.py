#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import require_packages

require_packages(["torch", "transformers", "datasets", "yaml"])

import torch

from src.data_utils import load_text_samples, tokenize_samples
from src.io_utils import ensure_dir, load_config, sanitize_name, setup_storage_env
from src.model_utils import iter_model_revisions, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="/tmp/model_feature_layer_dump")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--logit-top-k", type=int, default=20)
    parser.add_argument("--save-full-logits", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_storage_env(config)
    texts = load_text_samples(config)
    seq_len = args.sequence_length or int(config["dataset"].get("sequence_lengths", [128])[0])
    out_root = ensure_dir(args.output_dir)
    manifest = {
        "config": args.config,
        "output_dir": str(out_root),
        "sequence_length": seq_len,
        "num_samples": args.num_samples,
        "files": [],
    }

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
            batches = tokenize_samples(tokenizer, texts, seq_len, args.num_samples)
            model_dir = ensure_dir(out_root / sanitize_name(name) / revision)
            for batch_idx, batch in enumerate(batches):
                sample_index = batch.pop("sample_index")
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["input_ids"].clone()
                labels[batch["attention_mask"] == 0] = -100
                with torch.no_grad():
                    outputs = model(
                        **batch,
                        labels=labels,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                hidden_states = [h.detach().cpu() for h in outputs.hidden_states]
                final_hidden_state = hidden_states[-1]
                logits = outputs.logits.detach().cpu()
                last_token_logits = logits[:, -1, :]
                top_values, top_indices = torch.topk(last_token_logits, k=min(args.logit_top_k, last_token_logits.shape[-1]), dim=-1)
                loss = float(outputs.loss.detach().cpu())
                path = model_dir / f"sample{sample_index}_seq{seq_len}.pt"
                payload = {
                    "model": name,
                    "checkpoint": revision,
                    "tokenizer": tokenizer.name_or_path,
                    "sample_index": sample_index,
                    "sequence_length": seq_len,
                    "input_ids": batch["input_ids"].detach().cpu(),
                    "attention_mask": batch["attention_mask"].detach().cpu(),
                    "hidden_states": hidden_states,
                    "final_hidden_state": final_hidden_state,
                    "last_token_topk_logit_values": top_values,
                    "last_token_topk_token_ids": top_indices,
                    "logits_shape": list(logits.shape),
                    "loss": loss,
                }
                if args.save_full_logits:
                    payload["logits"] = logits
                torch.save(payload, path)
                manifest["files"].append(
                    {
                        "model": name,
                        "checkpoint": revision,
                        "sample_index": sample_index,
                        "path": str(path),
                        "num_hidden_states": len(hidden_states),
                        "hidden_shape": list(hidden_states[0].shape) if hidden_states else None,
                        "logits_shape": list(logits.shape),
                        "loss": loss,
                    }
                )
                print(f"wrote {path} loss={loss:.6f}")

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
