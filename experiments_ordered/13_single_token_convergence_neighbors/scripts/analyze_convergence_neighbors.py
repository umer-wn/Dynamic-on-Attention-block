#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts._bootstrap import require_packages

require_packages(["datasets", "matplotlib", "numpy", "torch", "transformers"])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


CHECKPOINTS = ["step0", "step1000", "step16000", "step143000"]
COLORS = {
    "step0": "#9c3f35",
    "step1000": "#dc8a2e",
    "step16000": "#2878b5",
    "step143000": "#2f855a",
}
DEFAULT_SINGLE_ROOT = Path(
    "/home/luohaoming/model_feature_experiments/single_token_frequency_8bins/pilot"
)
DEFAULT_CACHE = Path("/home/luohaoming/model_feature_cache/hf_cache")
DEFAULT_ARROW = DEFAULT_CACHE / (
    "wikitext/wikitext-2-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifests(single_root: Path) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = {}
    for checkpoint in CHECKPOINTS:
        rows: list[dict] = []
        for path in sorted((single_root / checkpoint).glob("shard*/raw/*__manifest.jsonl")):
            rows.extend(row for row in read_jsonl(path) if row["group"] == "isolated_token")
        if len(rows) != 32 or len({int(row["token_id"]) for row in rows}) != 32:
            raise RuntimeError(f"{checkpoint}: expected 32 unique isolated-token states, got {len(rows)}")
        output[checkpoint] = sorted(rows, key=lambda row: int(row["token_id"]))
    return output


def wikitext_train_counts(arrow_path: Path, tokenizer) -> Counter[int]:
    if not arrow_path.is_file():
        raise FileNotFoundError(arrow_path)
    dataset = Dataset.from_file(str(arrow_path))
    text = "\n".join(str(row["text"]) for row in dataset)
    return Counter(int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"])


def confidence_label(cosine_z: float, cosine_margin: float) -> str:
    # A transparent descriptive tier, not a calibrated probability.
    if cosine_z >= 5.0 and cosine_margin >= 0.02:
        return "high_separation"
    if cosine_z >= 3.0 and cosine_margin >= 0.005:
        return "moderate_separation"
    return "low_separation"


def prediction_description(neighbor_id: int, prediction_id: int, probability: float) -> str:
    if neighbor_id == prediction_id:
        return f"geometry_and_LM_head_agree; LM_top1_p={probability:.4f}"
    return f"geometry_only_neighbor; LM_head_predicts_token{prediction_id}; LM_top1_p={probability:.4f}"


def decode_topk(tokenizer, ids: torch.Tensor, values: torch.Tensor) -> str:
    return json.dumps(
        [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "value": float(value),
            }
            for token_id, value in zip(ids.tolist(), values.tolist())
        ],
        ensure_ascii=False,
    )


def analyze_checkpoint(
    checkpoint: str,
    manifest_rows: list[dict],
    tokenizer,
    counts: Counter[int],
    cache_dir: Path,
    tail_size: int,
    device: torch.device,
) -> tuple[list[dict], torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        revision=checkpoint,
        cache_dir=str(cache_dir),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    input_weight = model.get_input_embeddings().weight.detach().float()
    output_weight = model.get_output_embeddings().weight.detach().float()
    special_ids = set(int(value) for value in tokenizer.all_special_ids)

    centers: list[torch.Tensor] = []
    state_metadata: list[dict] = []
    for manifest in manifest_rows:
        payload = torch.load(manifest["state_path"], map_location="cpu", weights_only=True)
        states = payload["target_states"].float()
        if states.ndim != 2 or states.shape[0] < tail_size:
            raise RuntimeError(f"invalid trajectory state tensor: {manifest['state_path']}")
        tail = states[-tail_size:]
        center = tail.mean(dim=0)
        centers.append(center)
        radius = torch.linalg.vector_norm(tail - center, dim=1)
        state_metadata.append(
            {
                "center_tail_start_step": int(states.shape[0] - tail_size),
                "center_tail_end_step": int(states.shape[0] - 1),
                "center_radius_mean": float(radius.mean()),
                "center_radius_max": float(radius.max()),
                "center_relative_radius_mean": float(radius.mean() / center.norm().clamp_min(1e-12)),
                "final_to_center_distance": float((states[-1] - center).norm()),
                "eval_start_to_center_distance": float((states[512] - center).norm()),
                "initial_to_center_distance": float((states[0] - center).norm()),
                "center_norm": float(center.norm()),
                "center_sha256": hashlib.sha256(center.numpy().tobytes()).hexdigest(),
            }
        )

    center_batch = torch.stack(centers).to(device)
    normalized_centers = torch.nn.functional.normalize(center_batch, dim=1)
    normalized_input = torch.nn.functional.normalize(input_weight, dim=1)
    cosine = normalized_centers @ normalized_input.T
    if special_ids:
        cosine[:, list(special_ids)] = -torch.inf
    cosine_top_values, cosine_top_ids = cosine.topk(5, dim=1)
    finite_cosine = torch.where(torch.isfinite(cosine), cosine, torch.nan)
    cosine_mean = torch.nanmean(finite_cosine, dim=1)
    cosine_std = torch.sqrt(torch.nanmean((finite_cosine - cosine_mean[:, None]).square(), dim=1))

    center_sq = center_batch.square().sum(dim=1, keepdim=True)
    input_sq = input_weight.square().sum(dim=1).unsqueeze(0)
    squared_distance = center_sq + input_sq - 2.0 * center_batch @ input_weight.T
    if special_ids:
        squared_distance[:, list(special_ids)] = torch.inf
    euclidean_values, euclidean_ids = squared_distance.topk(5, dim=1, largest=False)
    euclidean_values = euclidean_values.clamp_min(0).sqrt()

    logits = center_batch @ output_weight.T
    probabilities = torch.softmax(logits, dim=1)
    lm_values, lm_ids = probabilities.topk(5, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    normalized_entropy = -(probabilities * log_probabilities).sum(dim=1) / math.log(logits.shape[1])
    logit_top2 = logits.topk(2, dim=1).values

    rows: list[dict] = []
    for index, manifest in enumerate(manifest_rows):
        neighbor_id = int(cosine_top_ids[index, 0])
        prediction_id = int(lm_ids[index, 0])
        euclidean_id = int(euclidean_ids[index, 0])
        cosine_top1 = float(cosine_top_values[index, 0])
        cosine_margin = cosine_top1 - float(cosine_top_values[index, 1])
        cosine_z = (cosine_top1 - float(cosine_mean[index])) / max(float(cosine_std[index]), 1e-12)
        lm_probability = float(lm_values[index, 0])
        rows.append(
            {
                "checkpoint": checkpoint,
                "training_step": int(checkpoint.removeprefix("step")),
                "source_token_id": int(manifest["token_id"]),
                "source_token": manifest["decoded"],
                "source_wikitext_train_count": int(counts[int(manifest["token_id"])]),
                "frequency_bin": int(manifest["frequency_bin"]),
                **state_metadata[index],
                "cosine_neighbor_token_id": neighbor_id,
                "cosine_neighbor_token": tokenizer.decode([neighbor_id]),
                "cosine_neighbor_wikitext_train_count": int(counts[neighbor_id]),
                "cosine_similarity": cosine_top1,
                "cosine_top1_top2_margin": cosine_margin,
                "cosine_vocab_zscore": cosine_z,
                "geometry_confidence": confidence_label(cosine_z, cosine_margin),
                "cosine_neighbor_is_source": neighbor_id == int(manifest["token_id"]),
                "cosine_top5": decode_topk(tokenizer, cosine_top_ids[index], cosine_top_values[index]),
                "euclidean_neighbor_token_id": euclidean_id,
                "euclidean_neighbor_token": tokenizer.decode([euclidean_id]),
                "euclidean_neighbor_wikitext_train_count": int(counts[euclidean_id]),
                "euclidean_distance": float(euclidean_values[index, 0]),
                "euclidean_top5": decode_topk(tokenizer, euclidean_ids[index], euclidean_values[index]),
                "lm_top1_token_id": prediction_id,
                "lm_top1_token": tokenizer.decode([prediction_id]),
                "lm_top1_wikitext_train_count": int(counts[prediction_id]),
                "lm_top1_probability": lm_probability,
                "lm_top1_top2_logit_margin": float(logit_top2[index, 0] - logit_top2[index, 1]),
                "lm_normalized_entropy": float(normalized_entropy[index]),
                "lm_top5": decode_topk(tokenizer, lm_ids[index], lm_values[index]),
                "cosine_neighbor_matches_lm_top1": neighbor_id == prediction_id,
                "similarity_prediction_description": prediction_description(
                    neighbor_id, prediction_id, lm_probability
                ),
            }
        )
    del model
    return rows, torch.stack(centers)


def summarize(rows: list[dict], centers: dict[str, torch.Tensor]) -> list[dict]:
    summaries: list[dict] = []
    for checkpoint in CHECKPOINTS:
        group = [row for row in rows if row["checkpoint"] == checkpoint]
        checkpoint_centers = centers[checkpoint].float()
        centroid = checkpoint_centers.mean(dim=0)
        centroid_distances = torch.linalg.vector_norm(checkpoint_centers - centroid, dim=1)
        pairwise = torch.pdist(checkpoint_centers)
        summaries.append(
            {
                "checkpoint": checkpoint,
                "n_tokens": len(group),
                "median_cosine_similarity": float(np.median([row["cosine_similarity"] for row in group])),
                "median_cosine_margin": float(np.median([row["cosine_top1_top2_margin"] for row in group])),
                "median_cosine_vocab_zscore": float(np.median([row["cosine_vocab_zscore"] for row in group])),
                "high_separation_fraction": sum(
                    row["geometry_confidence"] == "high_separation" for row in group
                )
                / len(group),
                "neighbor_is_source_fraction": sum(row["cosine_neighbor_is_source"] for row in group)
                / len(group),
                "neighbor_matches_lm_top1_fraction": sum(
                    row["cosine_neighbor_matches_lm_top1"] for row in group
                )
                / len(group),
                "median_lm_top1_probability": float(
                    np.median([row["lm_top1_probability"] for row in group])
                ),
                "median_lm_normalized_entropy": float(
                    np.median([row["lm_normalized_entropy"] for row in group])
                ),
                "median_center_relative_radius": float(
                    np.median([row["center_relative_radius_mean"] for row in group])
                ),
                "across_token_center_distance_mean": float(centroid_distances.mean()),
                "across_token_center_distance_max": float(centroid_distances.max()),
                "across_token_center_distance_relative_mean": float(
                    centroid_distances.mean() / centroid.norm().clamp_min(1e-12)
                ),
                "across_token_pairwise_distance_max": float(pairwise.max()),
            }
        )
    return summaries


def plot_neighbor_summary(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    metrics = [
        ("cosine_similarity", "Nearest-token cosine"),
        ("cosine_top1_top2_margin", "Cosine top1−top2 margin"),
        ("lm_top1_probability", "LM-head top-1 probability"),
    ]
    for ax, (metric, label) in zip(axes, metrics):
        for checkpoint in CHECKPOINTS:
            group = [row for row in rows if row["checkpoint"] == checkpoint]
            ax.scatter(
                [math.log10(int(row["source_wikitext_train_count"]) + 1) for row in group],
                [float(row[metric]) for row in group],
                s=28,
                alpha=0.75,
                color=COLORS[checkpoint],
                label=checkpoint,
            )
        ax.set_xlabel("log10(source token WikiText-2 count + 1)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.suptitle("Convergence-center neighbor and prediction confidence")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_step16000_endpoints(single_root: Path, output: Path) -> None:
    rows: list[dict] = []
    for path in sorted((single_root / "step16000").glob("shard*/raw/*__trajectory.jsonl")):
        rows.extend(
            row
            for row in read_jsonl(path)
            if row["group"] == "isolated_token" and int(row["step"]) >= 512
        )
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["token_id"])].append(row)
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for bin_id, ax in enumerate(axes.flat):
        tokens = sorted(
            [token_id for token_id, group in grouped.items() if int(group[0]["frequency_bin"]) == bin_id],
            key=lambda token_id: int(grouped[token_id][0]["frequency_count"]),
        )
        for index, token_id in enumerate(tokens):
            group = sorted(grouped[token_id], key=lambda row: int(row["step"]))
            x = np.asarray([float(row["projection_0"]) for row in group])
            y = np.asarray([float(row["projection_1"]) for row in group])
            color = cmap(0.12 + 0.78 * index / max(len(tokens) - 1, 1))
            label = f"{group[0]['decoded']!r} (n={int(group[0]['frequency_count'])})"
            ax.plot(x, y, lw=1.15, color=color, alpha=0.88, label=label)
            ax.scatter(x[0], y[0], marker="o", s=42, color=color, edgecolor="black", zorder=4)
            ax.scatter(x[-1], y[-1], marker="X", s=58, color=color, edgecolor="black", zorder=5)
            ax.annotate("S512", (x[0], y[0]), xytext=(4, 4), textcoords="offset points", fontsize=6)
            if index == 0:
                ax.annotate(
                    "E768 cluster",
                    (x[-1], y[-1]),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=6,
                )
        ax.set_title(f"Frequency bin {bin_id}")
        ax.set_xlabel("projection 0")
        ax.set_ylabel("projection 1")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("step16000 isolated-token trajectories — ○ S512, × E768")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-root", type=Path, default=DEFAULT_SINGLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wikitext-train-arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--tail-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_root = args.output_root
    processed = output_root / "processed"
    figures = output_root / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        revision="step100000",
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    counts = wikitext_train_counts(args.wikitext_train_arrow, tokenizer)
    manifests = load_manifests(args.single_root)
    rows: list[dict] = []
    centers: dict[str, torch.Tensor] = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_rows, checkpoint_centers = analyze_checkpoint(
            checkpoint,
            manifests[checkpoint],
            tokenizer,
            counts,
            args.cache_dir,
            args.tail_size,
            device,
        )
        rows.extend(checkpoint_rows)
        centers[checkpoint] = checkpoint_centers
        print(json.dumps({"checkpoint": checkpoint, "completed_rows": len(rows)}), flush=True)

    summary_rows = summarize(rows, centers)
    write_csv(processed / "convergence_center_neighbors.csv", rows)
    write_csv(processed / "checkpoint_summary.csv", summary_rows)
    torch.save(
        {
            "checkpoints": CHECKPOINTS,
            "token_ids": {
                checkpoint: [int(row["token_id"]) for row in manifests[checkpoint]]
                for checkpoint in CHECKPOINTS
            },
            "tail_size": args.tail_size,
            "centers": centers,
        },
        processed / "convergence_centers.pt",
    )
    plot_neighbor_summary(rows, figures / "convergence_neighbor_confidence.png")
    plot_step16000_endpoints(
        args.single_root, figures / "single_token_frequency_projection_step16000_endpoints.png"
    )
    metadata = {
        "status": "complete",
        "checkpoints": CHECKPOINTS,
        "rows": len(rows),
        "tail_size": args.tail_size,
        "center_definition": f"arithmetic mean of target hidden states steps {769 - args.tail_size}..768",
        "frequency_source": str(args.wikitext_train_arrow),
        "frequency_token_count": int(sum(counts.values())),
        "frequency_vocab_observed": len(counts),
        "nearest_neighbor_metric": "cosine over checkpoint input-embedding vocabulary; special tokens excluded",
        "prediction_metric": "softmax(checkpoint output_head @ convergence_center)",
        "device": str(device),
    }
    (output_root / "run_complete.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
