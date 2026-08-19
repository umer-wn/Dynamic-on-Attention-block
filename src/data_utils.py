from __future__ import annotations

from typing import Any, Optional

import torch


def load_text_samples(config: dict[str, Any]) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install datasets from requirements.txt") from exc

    ds_cfg = config["dataset"]
    dataset = load_dataset(
        ds_cfg["name"],
        ds_cfg.get("config"),
        split=ds_cfg.get("split", "validation"),
        cache_dir=config.get("cache_dir"),
        download_mode="reuse_dataset_if_exists" if config.get("offline") else None,
    )
    text_col = ds_cfg.get("text_column", "text")
    samples: list[str] = []
    for item in dataset:
        text = str(item.get(text_col, "")).strip()
        if text:
            samples.append(text)
        if len(samples) >= int(ds_cfg.get("num_samples", 128)):
            break
    if not samples:
        raise ValueError("Dataset produced no non-empty text samples")
    return samples


def tokenize_samples(tokenizer: Any, texts: list[str], sequence_length: int, max_samples: Optional[int] = None) -> list[dict[str, Any]]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    selected = texts[:max_samples] if max_samples is not None else texts
    batches: list[dict[str, Any]] = []
    for sample_index, text in enumerate(selected):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
            return_tensors="pt",
        )
        batch = {k: v for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
        batch["sample_index"] = sample_index
        batches.append(batch)
    return batches
