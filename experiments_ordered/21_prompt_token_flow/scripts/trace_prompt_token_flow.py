#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CACHE = Path("/home/luohaoming/model_feature_cache/hf_cache")
ROOT = Path(__file__).resolve().parents[1]


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tensor_from_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(type(output))


def add_box(ax, x, y, w, h, text, color, fontsize=9):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.13,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.1))


def draw_structure(path: Path, tokens: list[str], layer_count: int, vocab: int, hidden: int):
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    boxes = [
        (0.02, 0.38, 0.13, 0.25, f"input_ids\n[1, {len(tokens)}]"),
        (0.19, 0.38, 0.14, 0.25, f"Embedding\n{vocab} → {hidden}\n[1, {len(tokens)}, {hidden}]"),
        (0.38, 0.30, 0.20, 0.41, f"GPT-NeoX blocks × {layer_count}\n\nLayerNorm → Attention\n+ residual\nLayerNorm → MLP\n+ residual\n\n[1, {len(tokens)}, {hidden}]"),
        (0.63, 0.38, 0.13, 0.25, f"Final LayerNorm\n[1, {len(tokens)}, {hidden}]"),
        (0.81, 0.34, 0.17, 0.33, f"LM head\n{hidden} → {vocab}\nlogits\n[1, {len(tokens)}, {vocab}]"),
    ]
    for idx, (x, y, w, h, text) in enumerate(boxes):
        add_box(ax, x, y, w, h, text, colors[idx])
        if idx:
            px, py, pw, ph, _ = boxes[idx - 1]
            arrow(ax, px + pw, py + ph / 2, x, y + h / 2)
    ax.text(0.5, 0.91, "Pythia-70M causal LM forward structure", ha="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.09, "Attention is causal: logits at position i only use input positions 0…i", ha="center", fontsize=11)

    ax2 = fig.add_subplot(gs[1]); ax2.set_axis_off(); ax2.set_xlim(-0.7, len(tokens) + 0.7); ax2.set_ylim(0, 1)
    for i, token in enumerate(tokens):
        add_box(ax2, i - 0.38, 0.68, 0.76, 0.18, f"input[{i}]\n{token!r}", "#4c78a8", 9)
        add_box(ax2, i - 0.38, 0.24, 0.76, 0.18, f"logits[{i}]\npredicts token {i+1}", "#e45756", 9)
        arrow(ax2, i, 0.68, i, 0.42)
        if i < len(tokens) - 1:
            arrow(ax2, i + 0.10, 0.24, i + 0.90, 0.68)
    ax2.text((len(tokens) - 1) / 2, 0.94, "Token-position alignment (zero-based Python indexing)", ha="center", fontsize=14, weight="bold")
    ax2.text(len(tokens) - 1, 0.08, f"logits[{len(tokens)-1}] predicts the NEW token after the prompt", ha="center", fontsize=10)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="step57000")
    parser.add_argument("--prompt", default="the cat sit on the")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for directory in (ROOT / "processed", ROOT / "figures"):
        directory.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m", revision="step100000",
        cache_dir=str(args.cache_dir), local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", revision=args.checkpoint,
        cache_dir=str(args.cache_dir), local_files_only=True,
        dtype=torch.float32, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    (ROOT / "model_structure.txt").write_text(str(model) + "\n", encoding="utf-8")

    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)
    ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([value]) for value in ids]

    captured: dict[str, torch.Tensor] = {}
    hooks = []
    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = tensor_from_output(output).detach().float().cpu()
        return hook
    hooks.append(model.gpt_neox.embed_in.register_forward_hook(capture("embedding")))
    for i, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.register_forward_hook(capture(f"block_{i}")))
    hooks.append(model.gpt_neox.final_layer_norm.register_forward_hook(capture("final_layer_norm")))
    hooks.append(model.embed_out.register_forward_hook(capture("lm_head_logits")))
    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    for handle in hooks:
        handle.remove()
    logits = output.logits.detach().float().cpu()

    token_rows = []
    for i, (token_id, token) in enumerate(zip(ids, tokens)):
        token_rows.append({
            "position_zero_based": i,
            "position_one_based": i + 1,
            "token_id": token_id,
            "token": token,
            "prefix_visible_to_logits": tokenizer.decode(ids[: i + 1]),
            "logits_index_zero_based": i,
            "predicts_input_position_zero_based": i + 1 if i + 1 < len(ids) else "",
            "actual_next_input_token": tokens[i + 1] if i + 1 < len(tokens) else "<new token after prompt>",
        })
    save_csv(ROOT / "processed/token_alignment.csv", token_rows)

    stage_rows = []
    for stage, values in captured.items():
        if values.ndim != 3 or values.shape[1] != len(ids):
            continue
        for i, token in enumerate(tokens):
            vector = values[0, i]
            stage_rows.append({
                "stage": stage,
                "position_zero_based": i,
                "token_id": ids[i],
                "token": token,
                "vector_dimension": vector.numel(),
                "l2_norm": float(torch.linalg.vector_norm(vector)),
                "mean": float(vector.mean()),
                "std": float(vector.std()),
            })
    save_csv(ROOT / "processed/token_hidden_flow.csv", stage_rows)

    prediction_rows = []
    for i in range(len(ids)):
        probability = torch.softmax(logits[0, i], dim=-1)
        values, indices = probability.topk(5)
        actual_next_id = ids[i + 1] if i + 1 < len(ids) else None
        prediction_rows.append({
            "logits_index_zero_based": i,
            "logits_index_one_based": i + 1,
            "context": tokenizer.decode(ids[: i + 1]),
            "actual_next_input_token": tokens[i + 1] if actual_next_id is not None else "<new token after prompt>",
            "actual_next_probability": float(probability[actual_next_id]) if actual_next_id is not None else "",
            "top1_token_id": int(indices[0]),
            "top1_token": tokenizer.decode([int(indices[0])]),
            "top1_probability": float(values[0]),
            "top5_json": json.dumps([
                {"token_id": int(idx), "token": tokenizer.decode([int(idx)]), "probability": float(value)}
                for value, idx in zip(values, indices)
            ], ensure_ascii=False),
        })
    save_csv(ROOT / "processed/per_position_predictions.csv", prediction_rows)

    metadata = {
        "model": "EleutherAI/pythia-70m",
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "input_ids_shape": list(input_ids.shape),
        "token_ids": ids,
        "tokens": tokens,
        "num_input_tokens": len(ids),
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "vocab_size": model.config.vocab_size,
        "logits_shape": list(logits.shape),
        "captured_shapes": {name: list(value.shape) for name, value in captured.items()},
        "lm_head_hook_matches_output_logits_max_abs_error": float((captured["lm_head_logits"] - logits).abs().max()),
        "alignment_conclusion": (
            f"For {len(ids)} input tokens, Python logits[:, {len(ids)-1}, :] (one-based output position {len(ids)}) "
            "predicts the new token after the full prompt; logits[:, 5, :] does not exist."
        ),
    }
    (ROOT / "processed/trace_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw_structure(
        ROOT / "figures/pythia70m_structure_and_token_alignment.png",
        tokens, model.config.num_hidden_layers, model.config.vocab_size, model.config.hidden_size,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({"predictions": prediction_rows}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
