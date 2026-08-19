#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts._bootstrap import require_packages

require_packages(["torch", "transformers", "yaml"])

import torch

from src.io_utils import base_metadata, load_config, setup_storage_env, write_jsonl
from src.model_utils import load_model_and_tokenizer


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fixed_projection_vectors(
    shape: torch.Size,
    device: torch.device,
    count: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    vectors: list[torch.Tensor] = []
    for _ in range(int(count)):
        vector = torch.randn(
            shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        vectors.append(vector / vector.norm().clamp_min(1e-12))
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--text-steps", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    setup_storage_env(config)
    output_dir = Path(config["output_dir"])
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = config["models"][0]
    model_name = str(model_cfg["name"])
    revision = str(model_cfg["revision"])
    tokenizer_name = str(model_cfg.get("tokenizer", model_name))
    model, tokenizer, device = load_model_and_tokenizer(
        model_name,
        revision,
        tokenizer_name,
        config.get("dtype", "float32"),
        config.get("device", "cuda"),
        cache_dir=config.get("cache_dir"),
        attn_implementation=config.get("attn_implementation"),
        local_files_only=bool(config.get("offline", False)),
        tokenizer_revision=model_cfg.get("tokenizer_revision"),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    source_cfg = config["source"]
    source_rows = read_jsonl(Path(source_cfg["manifest_path"]))
    sample_index = int(source_cfg.get("sample_index", 0))
    source_row = source_rows[sample_index]
    text = str(source_row[source_cfg.get("text_field", "text")])
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_offset = int(source_cfg.get("token_offset", 0))

    dynamics = config["dynamics"]
    window_length = int(dynamics["window_length"])
    tokens_per_text_step = int(dynamics["tokens_per_text_step"])
    text_steps = int(args.text_steps or dynamics["text_steps"])
    if tokens_per_text_step != window_length:
        raise ValueError(
            "tokens_per_text_step must equal window_length so one text-level "
            "step completely replaces the preceding window"
        )
    initial_ids = token_ids[token_offset : token_offset + window_length]
    if len(initial_ids) != window_length:
        raise ValueError(
            f"source has insufficient tokens at offset {token_offset}: "
            f"{len(initial_ids)}/{window_length}"
        )
    window = torch.tensor([initial_ids], device=device, dtype=torch.long)
    embedding_layer = model.get_input_embeddings()
    with torch.no_grad():
        initial_embedding_state = embedding_layer(window).float()

    projection_cfg = config["projection"]
    projection_vectors = fixed_projection_vectors(
        initial_embedding_state.shape,
        device,
        int(projection_cfg.get("count", 4)),
        int(projection_cfg.get("seed", 1234)),
    )
    projection_digest = hashlib.sha256(
        b"".join(vector.detach().cpu().numpy().tobytes() for vector in projection_vectors)
    ).hexdigest()

    position_ids = torch.arange(window_length, device=device, dtype=torch.long).unsqueeze(0)
    seen_windows: dict[tuple[int, ...], int] = {
        tuple(int(value) for value in window[0].tolist()): 0
    }
    cycle_start: int | None = None
    cycle_length: int | None = None
    trajectory_rows: list[dict] = []
    started = time.perf_counter()

    with torch.inference_mode():
        for text_step in range(1, text_steps + 1):
            previous_window = window.detach().clone()
            generated_ids: list[int] = []
            for _ in range(tokens_per_text_step):
                attention_mask = torch.ones_like(window)
                outputs = model(
                    input_ids=window,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                )
                next_token = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
                generated_ids.append(next_token)
                next_tensor = torch.tensor(
                    [[next_token]],
                    device=device,
                    dtype=window.dtype,
                )
                window = torch.cat([window[:, 1:], next_tensor], dim=1)

            current_ids = [int(value) for value in window[0].tolist()]
            if current_ids != generated_ids:
                raise RuntimeError(
                    "a text-level step must end with exactly the eight newly generated tokens"
                )
            state = embedding_layer(window).float()
            key = tuple(current_ids)
            if key in seen_windows and cycle_length is None:
                cycle_start = seen_windows[key]
                cycle_length = text_step - seen_windows[key]
            seen_windows.setdefault(key, text_step)
            row = {
                "checkpoint": revision,
                "training_step": int(revision.removeprefix("step")),
                "text_step": text_step,
                "micro_steps_completed": text_step * tokens_per_text_step,
                "token_ids": current_ids,
                "decoded_text": tokenizer.decode(current_ids),
                "token_change_fraction": float(
                    (window != previous_window).float().mean().cpu()
                ),
                "unique_token_ratio": len(set(current_ids)) / window_length,
                "window_sha256": hashlib.sha256(
                    ",".join(str(value) for value in current_ids).encode("ascii")
                ).hexdigest(),
            }
            for projection_index, vector in enumerate(projection_vectors):
                row[f"projection_{projection_index}"] = float(
                    (state * vector).sum().cpu()
                )
            trajectory_rows.append(row)

    common = {
        **base_metadata(
            config,
            model_name,
            revision,
            tokenizer_name,
            window_length,
        ),
        "training_step": int(revision.removeprefix("step")),
        "source_id": source_row.get("source_id"),
        "source_dataset_name": source_row.get("dataset_name"),
        "source_manifest": str(source_cfg["manifest_path"]),
        "source_sample_index": sample_index,
        "source_text_sha256": source_row.get("text_sha256"),
        "token_offset": token_offset,
        "initial_token_ids": initial_ids,
        "initial_text": tokenizer.decode(initial_ids),
        "window_length": window_length,
        "tokens_per_text_step": tokens_per_text_step,
        "text_steps": text_steps,
        "generation": "greedy_argmax",
        "position_mode": "reset_0_to_7_each_micro_step",
        "padding": "none",
        "use_cache": False,
        "projection_target": "flattened_full_window_input_embeddings_8xH",
        "projection_count": len(projection_vectors),
        "projection_seed": int(projection_cfg.get("seed", 1234)),
        "projection_vectors_sha256": projection_digest,
    }
    trajectory_rows = [{**common, **row} for row in trajectory_rows]
    summary = {
        **common,
        "status": "complete",
        "trajectory_rows": len(trajectory_rows),
        "total_generated_tokens": text_steps * tokens_per_text_step,
        "unique_text_windows": len(seen_windows),
        "text_cycle_start": cycle_start,
        "text_cycle_length": cycle_length,
        "final_token_ids": trajectory_rows[-1]["token_ids"],
        "final_text": trajectory_rows[-1]["decoded_text"],
        "mean_token_change_fraction": sum(
            float(row["token_change_fraction"]) for row in trajectory_rows
        )
        / len(trajectory_rows),
        "runtime_seconds": time.perf_counter() - started,
    }
    prefix = f"{config['experiment_name']}__{revision}"
    write_jsonl(raw_dir / f"{prefix}__trajectory.jsonl", trajectory_rows)
    write_jsonl(raw_dir / f"{prefix}__summary.jsonl", [summary])
    (output_dir / "run_complete.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
