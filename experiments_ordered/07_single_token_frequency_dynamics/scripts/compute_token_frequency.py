#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import require_packages

require_packages(["transformers", "datasets", "yaml"])

from src.io_utils import load_config, setup_storage_env, write_jsonl


def lexical(decoded: str) -> bool:
    return any(character.isalnum() for character in decoded)


def quantile_bin(rank: int, total: int, bins: int) -> int:
    return min(int(rank * bins / max(total, 1)), bins - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    setup_storage_env(config)
    from datasets import load_dataset

    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    model_cfg = config["models"][0]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.get("tokenizer", model_cfg["name"]),
        revision=model_cfg.get("revision", "main"),
        cache_dir=config.get("cache_dir"),
        local_files_only=bool(config.get("offline", False)),
    )
    data_cfg = config["dataset"]
    dataset = load_dataset(
        data_cfg["name"], data_cfg.get("config"), split=data_cfg["split"], cache_dir=config.get("cache_dir")
    )
    text = "\n".join(str(row[data_cfg.get("text_column", "text")]) for row in dataset)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    counts = Counter(int(token) for token in token_ids)
    context_length = int(config["sampling"]["context_length"])
    min_contexts = int(config["sampling"].get("min_contexts", 2))
    occurrences: dict[int, list[int]] = defaultdict(list)
    for position, token in enumerate(token_ids):
        if position >= context_length - 1:
            occurrences[int(token)].append(position)

    eligible = []
    special = set(tokenizer.all_special_ids)
    for token, positions in occurrences.items():
        decoded = tokenizer.decode([token])
        if token in special or len(positions) < min_contexts or not lexical(decoded):
            continue
        eligible.append((token, counts[token], decoded, positions))
    eligible.sort(key=lambda row: (row[1], row[0]))
    bins = int(config["sampling"].get("frequency_bins", 4))
    per_bin = int(config["sampling"].get("tokens_per_bin", 4))
    seed = int(config.get("seed", 1234))
    import random

    randomizer = random.Random(seed)
    rows: list[dict] = []
    grouped: dict[int, list] = defaultdict(list)
    for rank, item in enumerate(eligible):
        grouped[quantile_bin(rank, len(eligible), bins)].append((rank, item))
    for bin_index in range(bins):
        candidates = grouped[bin_index]
        if len(candidates) < per_bin:
            raise RuntimeError(f"frequency bin {bin_index} has only {len(candidates)} eligible tokens")
        # Stratified-within-bin positions avoid selecting only a narrow count tie.
        selected = []
        for index in range(per_bin):
            left = math.floor(index * len(candidates) / per_bin)
            right = max(left + 1, math.floor((index + 1) * len(candidates) / per_bin))
            selected.append(randomizer.choice(candidates[left:right]))
        for rank, (token, count, decoded, positions) in selected:
            chosen_positions = positions[:]
            randomizer.shuffle(chosen_positions)
            context_positions = sorted(chosen_positions[:min_contexts])
            contexts = [token_ids[position - context_length + 1 : position + 1] for position in context_positions]
            rows.append(
                {
                    "token_id": int(token),
                    "decoded": decoded,
                    "count": int(count),
                    "log10_count_plus1": math.log10(count + 1),
                    "eligible_rank": int(rank),
                    "eligible_total": len(eligible),
                    "frequency_bin": int(bin_index),
                    "context_length": context_length,
                    "context_positions": [int(value) for value in context_positions],
                    "contexts": [[int(value) for value in context] for context in contexts],
                }
            )
    write_jsonl(output / "selected_tokens.jsonl", rows)
    audit = {
        "token_count": len(token_ids),
        "vocab_observed": len(counts),
        "eligible_token_types": len(eligible),
        "context_length": context_length,
        "min_contexts": min_contexts,
        "frequency_bins": bins,
        "tokens_per_bin": per_bin,
        "selected_count": len(rows),
        "selected_ids": [row["token_id"] for row in rows],
        "selection_seed": seed,
    }
    (output / "frequency_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
