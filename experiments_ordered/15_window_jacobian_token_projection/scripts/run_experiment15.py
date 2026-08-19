#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
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


CHECKPOINTS = [
    "step5000",
    "step7000",
    "step9000",
    "step13000",
    "step21000",
    "step29000",
    "step37000",
    "step53000",
    "step61000",
]
DEFAULT_CACHE = Path("/home/luohaoming/model_feature_cache/hf_cache")
DEFAULT_ARROW = DEFAULT_CACHE / (
    "wikitext/wikitext-2-raw-v1/0.0.0/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow"
)
DEFAULT_DATA_ROOT = Path(
    "/home/luohaoming/model_feature_experiments/"
    "experiment15_window_jacobian_token_projection"
)
DEFAULT_UPSTREAM_STATES_ROOT = Path(
    "/home/luohaoming/model_feature_experiments/"
    "experiment14_limit_cycle_jacobian/trace0_2048/states"
)
DEFAULT_REPORT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT13_TOKENS = Path(__file__).resolve().parents[2] / (
    "13_single_token_convergence_neighbors/processed/random8_selected_tokens.csv"
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_tokens(path: Path, count: int) -> list[dict]:
    rows = read_csv(path)[:count]
    if len(rows) != count:
        raise RuntimeError(f"expected {count} tokens in {path}, got {len(rows)}")
    return [
        {
            "selection_index": int(row["selection_index"]),
            "token_id": int(row["token_id"]),
            "token": row["token"],
            "wikitext_train_count": int(row["wikitext_train_count"]),
        }
        for row in rows
    ]


def projection_basis(hidden: int = 512, seed: int = 1414) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    raw = torch.randn((hidden, 3), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.float()


def load_model(checkpoint: str, cache: Path, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m",
        revision=checkpoint,
        cache_dir=str(cache),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def window_endpoints(steps: int, width: int) -> list[int]:
    if steps <= 0 or width <= 0 or steps % width:
        raise ValueError(f"steps={steps} must be positive and divisible by window={width}")
    return list(range(width, steps + 1, width))


def resolve_states_path(
    args,
    checkpoint: str,
    tokens: list[dict],
) -> Path:
    candidates = [
        args.data_root / "states" / f"{checkpoint}_states.pt",
        args.upstream_states_root / f"{checkpoint}_tail_states.pt",
    ]
    expected_shape = (args.steps + 1, len(tokens), 512)
    expected_ids = [row["token_id"] for row in tokens]
    for path in candidates:
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        states = payload.get("states", payload.get("recorded_states"))
        if states is None:
            continue
        if tuple(states.shape) != expected_shape:
            continue
        if [int(value) for value in payload["token_ids"]] != expected_ids:
            continue
        return path
    raise FileNotFoundError(
        f"no compatible state file for {checkpoint}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_states(args, checkpoint: str, tokens: list[dict]) -> torch.Tensor:
    path = resolve_states_path(args, checkpoint, tokens)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("states", payload.get("recorded_states")).float()


def dynamics_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    states_dir = args.data_root / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        output = states_dir / f"{checkpoint}_states.pt"
        if not args.overwrite:
            try:
                existing = resolve_states_path(args, checkpoint, tokens)
                print(
                    json.dumps(
                        {
                            "stage": "dynamics",
                            "checkpoint": checkpoint,
                            "status": "reuse",
                            "path": str(existing),
                        }
                    ),
                    flush=True,
                )
                continue
            except FileNotFoundError:
                pass
        started = time.perf_counter()
        model = load_model(checkpoint, args.cache_dir, args.device)
        ids = torch.tensor([row["token_id"] for row in tokens], device=args.device)
        state = model.get_input_embeddings()(ids).detach().float()
        attention_mask = torch.ones((len(tokens), 1), device=args.device, dtype=torch.long)
        position_ids = torch.zeros((1, 1), device=args.device, dtype=torch.long)
        states = [state.detach().cpu()]
        with torch.inference_mode():
            for _ in range(args.steps):
                state = model.gpt_neox(
                    inputs_embeds=state.unsqueeze(1),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, -1, :].float()
                states.append(state.detach().cpu())
        tensor = torch.stack(states)
        torch.save(
            {
                "checkpoint": checkpoint,
                "steps": args.steps,
                "token_ids": [row["token_id"] for row in tokens],
                "states": tensor,
            },
            output,
        )
        print(
            json.dumps(
                {
                    "stage": "dynamics",
                    "checkpoint": checkpoint,
                    "completed": checkpoint_index + 1,
                    "total": len(checkpoints),
                    "shape": list(tensor.shape),
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()


def write_projection_triples(
    csv_path: Path,
    jsonl_path: Path,
    checkpoints: list[str],
    tokens: list[dict],
    projected_by_checkpoint: dict[str, torch.Tensor],
) -> None:
    rows: list[dict] = []
    triples: list[str] = []
    for checkpoint in checkpoints:
        projected = projected_by_checkpoint[checkpoint].double().numpy()
        for dynamic_step in range(projected.shape[0]):
            for token_index, token in enumerate(tokens):
                p1 = float(projected[dynamic_step, token_index, 0])
                p2 = float(projected[dynamic_step, token_index, 1])
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "dynamic_step": dynamic_step,
                        **token,
                        "projection_1": p1,
                        "projection_2": p2,
                    }
                )
                triples.append(
                    json.dumps(
                        [
                            [checkpoint, dynamic_step],
                            [p1, p2],
                            {"token_id": token["token_id"], "token": token["token"]},
                        ],
                        ensure_ascii=False,
                    )
                )
    write_csv(csv_path, rows)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("\n".join(triples) + "\n", encoding="utf-8")


def plot_window_grid(
    checkpoints: list[str],
    tokens: list[dict],
    projected_by_checkpoint: dict[str, torch.Tensor],
    start: int,
    end: int,
    display_floor: float,
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    fig.subplots_adjust(top=0.91, bottom=0.075, hspace=0.30, wspace=0.16)
    colors = plt.get_cmap("tab10")
    for ax, checkpoint in zip(axes.flat, checkpoints):
        values = projected_by_checkpoint[checkpoint][start : end + 1, :, :2].double().numpy()
        values -= values.mean(axis=(0, 1), dtype=np.float64, keepdims=True)
        x_min, x_max = float(values[:, :, 0].min()), float(values[:, :, 0].max())
        y_min, y_max = float(values[:, :, 1].min()), float(values[:, :, 1].max())
        radius = np.linalg.norm(values, axis=2).reshape(-1)
        limit = 1.08 * max(
            abs(x_min), abs(x_max), abs(y_min), abs(y_max), display_floor
        )
        for token_index, token in enumerate(tokens):
            xy = values[:, token_index]
            color = colors(token_index)
            ax.plot(xy[:, 0], xy[:, 1], lw=0.9, color=color)
            ax.scatter(*xy[0], marker="o", s=20, color=color, edgecolor="black", linewidth=0.4)
            ax.scatter(*xy[-1], marker="X", s=30, color=color, edgecolor="black", linewidth=0.4)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.ticklabel_format(style="sci", scilimits=(-3, 3), useOffset=False)
        ax.set_title(f"{checkpoint} · r95={np.quantile(radius, 0.95):.3e}", fontsize=10)
        ax.text(
            0.02,
            0.98,
            f"x∈[{x_min:.2e}, {x_max:.2e}]\ny∈[{y_min:.2e}, {y_max:.2e}]",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.75,
                "edgecolor": "0.75",
            },
        )
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Δ projection 1")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ projection 2")
    handles = [
        plt.Line2D([0], [0], color=colors(index), label=repr(token["token"]))
        for index, token in enumerate(tokens)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(tokens), fontsize=9)
    fig.suptitle(
        f"Experiment 15 · dynamic steps {start}–{end} · shared fixed projection",
        fontsize=15,
        y=0.975,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def projection_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    figures = args.report_root / "figures"
    processed = args.report_root / "processed"
    figures.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    basis = projection_basis(seed=args.projection_seed)
    torch.save(
        {"seed": args.projection_seed, "basis": basis},
        processed / "projection_basis.pt",
    )
    projected: dict[str, torch.Tensor] = {}
    for checkpoint in checkpoints:
        states = load_states(args, checkpoint, tokens)
        projected[checkpoint] = torch.einsum("tbh,ph->tbp", states, basis)
    write_projection_triples(
        processed / "projection_trajectory.csv",
        processed / "projection_triples.jsonl",
        checkpoints,
        tokens,
        projected,
    )
    for end in window_endpoints(args.steps, args.window):
        plot_window_grid(
            checkpoints,
            tokens,
            projected,
            end - args.window,
            end,
            args.projection_display_floor,
            figures / f"projection_window_step{end - args.window:04d}_{end:04d}.png",
        )


def exact_jacobian(model, state: torch.Tensor, chunk_size: int) -> torch.Tensor:
    position_ids = torch.zeros((1, 1), device=state.device, dtype=torch.long)
    attention_mask = torch.ones((1, 1), device=state.device, dtype=torch.long)

    def mapping(value: torch.Tensor) -> torch.Tensor:
        return model.gpt_neox(
            inputs_embeds=value.reshape(1, 1, -1),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[0, -1].float()

    value = state.detach().float().requires_grad_(True)
    return torch.func.jacrev(mapping, chunk_size=chunk_size)(value).detach().float()


def topk_json(tokenizer, ids: torch.Tensor, values: torch.Tensor) -> str:
    return json.dumps(
        [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "value": float(value),
            }
            for token_id, value in zip(ids.cpu().tolist(), values.cpu().tolist())
        ],
        ensure_ascii=False,
    )


def wikitext_counts(path: Path, tokenizer) -> Counter[int]:
    dataset = Dataset.from_file(str(path))
    text = "\n".join(str(row["text"]) for row in dataset)
    return Counter(int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"])


def endpoint_token_metrics(
    state: torch.Tensor,
    model,
    tokenizer,
    counts: Counter[int],
) -> dict:
    value = state.reshape(1, -1).float()
    input_weight = model.get_input_embeddings().weight.detach().float()
    output_weight = model.get_output_embeddings().weight.detach().float()
    special = [int(token_id) for token_id in tokenizer.all_special_ids]

    cosine = torch.nn.functional.normalize(value, dim=1) @ torch.nn.functional.normalize(
        input_weight, dim=1
    ).T
    if special:
        cosine[:, special] = -torch.inf
    cosine_values, cosine_ids = cosine.topk(5, dim=1)

    squared_distance = (
        value.square().sum(dim=1, keepdim=True)
        + input_weight.square().sum(dim=1).unsqueeze(0)
        - 2.0 * value @ input_weight.T
    )
    if special:
        squared_distance[:, special] = torch.inf
    euclidean_values, euclidean_ids = squared_distance.topk(5, dim=1, largest=False)
    euclidean_values = euclidean_values.clamp_min(0).sqrt()

    logits = value @ output_weight.T
    probabilities = torch.softmax(logits, dim=1)
    lm_values, lm_ids = probabilities.topk(5, dim=1)
    lm_logits = logits.gather(1, lm_ids)
    normalized_entropy = (
        -(probabilities * torch.log_softmax(logits, dim=1)).sum() / math.log(logits.shape[1])
    )

    cosine_id = int(cosine_ids[0, 0])
    euclidean_id = int(euclidean_ids[0, 0])
    lm_id = int(lm_ids[0, 0])
    d1, d2 = float(euclidean_values[0, 0]), float(euclidean_values[0, 1])
    return {
        "hidden_state_norm": float(value.norm()),
        "cosine_token_id": cosine_id,
        "cosine_token": tokenizer.decode([cosine_id]),
        "cosine_wikitext_count": int(counts[cosine_id]),
        "cosine_similarity": float(cosine_values[0, 0]),
        "cosine_distance": 1.0 - float(cosine_values[0, 0]),
        "cosine_margin": float(cosine_values[0, 0] - cosine_values[0, 1]),
        "cosine_top5": topk_json(tokenizer, cosine_ids[0], cosine_values[0]),
        "euclidean_token_id": euclidean_id,
        "euclidean_token": tokenizer.decode([euclidean_id]),
        "euclidean_wikitext_count": int(counts[euclidean_id]),
        "euclidean_distance": d1,
        "euclidean_absolute_margin": d2 - d1,
        "euclidean_relative_margin": (d2 - d1) / max(d1, 1e-12),
        "euclidean_top5": topk_json(tokenizer, euclidean_ids[0], euclidean_values[0]),
        "lm_head_token_id": lm_id,
        "lm_head_token": tokenizer.decode([lm_id]),
        "lm_head_wikitext_count": int(counts[lm_id]),
        "lm_head_probability": float(lm_values[0, 0]),
        "lm_head_probability_margin": float(lm_values[0, 0] - lm_values[0, 1]),
        "lm_head_logit_margin": float(lm_logits[0, 0] - lm_logits[0, 1]),
        "lm_head_normalized_entropy": float(normalized_entropy),
        "lm_head_top5": topk_json(tokenizer, lm_ids[0], lm_values[0]),
        "cosine_equals_euclidean": cosine_id == euclidean_id,
        "cosine_equals_lm_head": cosine_id == lm_id,
        "euclidean_equals_lm_head": euclidean_id == lm_id,
    }


def add_endpoint_projections(
    args,
    rows: list[dict],
    tokens: list[dict],
) -> None:
    basis = projection_basis(seed=args.projection_seed)[:2].double()
    token_index = {int(token["token_id"]): index for index, token in enumerate(tokens)}
    by_checkpoint: dict[str, list[dict]] = {}
    for row in rows:
        by_checkpoint.setdefault(row["checkpoint"], []).append(row)
    for checkpoint, checkpoint_rows in by_checkpoint.items():
        states = load_states(args, checkpoint, tokens).double()
        for row in checkpoint_rows:
            state = states[
                int(row["dynamic_step"]),
                token_index[int(row["token_id"])],
            ]
            value = basis @ state
            row["projection_1"] = float(value[0])
            row["projection_2"] = float(value[1])


def plot_jacobian_spectral_radius(
    rows: list[dict],
    checkpoints: list[str],
    tokens: list[dict],
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.30, wspace=0.20)
    colors = plt.get_cmap("tab10")
    for ax, checkpoint in zip(axes.flat, checkpoints):
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        for token_index, token in enumerate(tokens):
            token_rows = sorted(
                (
                    row
                    for row in checkpoint_rows
                    if int(row["token_id"]) == int(token["token_id"])
                ),
                key=lambda row: int(row["dynamic_step"]),
            )
            ax.plot(
                [int(row["dynamic_step"]) for row in token_rows],
                [float(row["jacobian_lambda1_abs"]) for row in token_rows],
                marker="o",
                ms=3,
                lw=1.1,
                color=colors(token_index),
            )
        ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(checkpoint)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("dynamic step (window endpoint)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Jacobian spectral radius")
    handles = [
        plt.Line2D([0], [0], color=colors(index), marker="o", label=repr(token["token"]))
        for index, token in enumerate(tokens)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(tokens), fontsize=9)
    fig.suptitle(
        "Experiment 15 · endpoint Jacobian spectral radius by 256-step window",
        fontsize=15,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_jacobian_metric(
    rows: list[dict],
    tokens: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.30, wspace=0.20)
    colors = plt.get_cmap("tab10")
    for ax, checkpoint in zip(axes.flat, CHECKPOINTS):
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        for token_index, token in enumerate(tokens):
            token_rows = sorted(
                (
                    row
                    for row in checkpoint_rows
                    if int(row["token_id"]) == int(token["token_id"])
                ),
                key=lambda row: int(row["dynamic_step"]),
            )
            ax.plot(
                [int(row["dynamic_step"]) for row in token_rows],
                [float(row[metric]) for row in token_rows],
                marker="o",
                ms=3,
                lw=1.1,
                color=colors(token_index),
            )
        ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(checkpoint)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("dynamic step (window endpoint)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles = [
        plt.Line2D([0], [0], color=colors(index), marker="o", label=repr(token["token"]))
        for index, token in enumerate(tokens)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(tokens), fontsize=9)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_three_jacobian_metrics(
    rows: list[dict],
    tokens: list[dict],
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.30, wspace=0.20)
    definitions = [
        ("spectral_radius", "spectral radius"),
        ("spectral_abscissa", "max Re(eigenvalue)"),
        ("operator_norm_2", "operator 2-norm"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    endpoints = sorted({int(row["dynamic_step"]) for row in rows})
    for ax, checkpoint in zip(axes.flat, CHECKPOINTS):
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint]
        for (metric, label), color in zip(definitions, colors):
            medians = []
            lower = []
            upper = []
            for endpoint in endpoints:
                values = [
                    float(row[metric])
                    for row in checkpoint_rows
                    if int(row["dynamic_step"]) == endpoint
                ]
                medians.append(float(np.median(values)))
                lower.append(float(np.min(values)))
                upper.append(float(np.max(values)))
            ax.plot(endpoints, medians, marker="o", ms=3, lw=1.2, color=color, label=label)
            ax.fill_between(endpoints, lower, upper, color=color, alpha=0.12)
        ax.axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(checkpoint)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("dynamic step (window endpoint)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Jacobian metric")
    handles = [
        plt.Line2D([0], [0], color=color, marker="o", label=label)
        for (_, label), color in zip(definitions, colors)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(
        "Experiment 15 · Jacobian spectral radius, spectral abscissa, and operator norm\n"
        "lines: median across 4 tokens; bands: token min–max",
        fontsize=14,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def jacobian_geometry_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    processed = args.report_root / "processed"
    figures = args.report_root / "figures"
    parts = processed / "jacobian_geometry_parts"
    for directory in (processed, figures, parts):
        directory.mkdir(parents=True, exist_ok=True)
    endpoints = window_endpoints(args.steps, args.window)
    expected_per_checkpoint = len(endpoints) * len(tokens)

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        part_path = parts / f"{checkpoint}.csv"
        rows = [] if args.overwrite or not part_path.exists() else read_csv(part_path)
        completed = {
            (int(row["dynamic_step"]), int(row["token_id"]))
            for row in rows
        }
        if len(completed) == expected_per_checkpoint:
            print(
                json.dumps(
                    {
                        "stage": "jacobian_geometry",
                        "checkpoint": checkpoint,
                        "status": "skip",
                    }
                ),
                flush=True,
            )
            continue
        states = load_states(args, checkpoint, tokens)
        model = load_model(checkpoint, args.cache_dir, args.device)
        for dynamic_step in endpoints:
            for token_index, token in enumerate(tokens):
                key = (dynamic_step, int(token["token_id"]))
                if key in completed:
                    continue
                started = time.perf_counter()
                state = states[dynamic_step, token_index].to(args.device)
                jacobian = exact_jacobian(model, state, args.jacobian_chunk_size)
                eigenvalues = torch.linalg.eigvals(jacobian)
                absolute = eigenvalues.abs()
                radius_value, radius_index = absolute.max(dim=0)
                abscissa_value, abscissa_index = eigenvalues.real.max(dim=0)
                singular_values = torch.linalg.svdvals(jacobian)
                sigma_values = singular_values.topk(2).values
                radius_eigenvalue = eigenvalues[radius_index]
                abscissa_eigenvalue = eigenvalues[abscissa_index]
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "window_start": dynamic_step - args.window,
                        "window_end": dynamic_step,
                        "dynamic_step": dynamic_step,
                        **token,
                        "spectral_radius": float(radius_value),
                        "spectral_radius_eigenvalue_real": float(radius_eigenvalue.real),
                        "spectral_radius_eigenvalue_imag": float(radius_eigenvalue.imag),
                        "spectral_abscissa": float(abscissa_value),
                        "max_real_eigenvalue_imag": float(abscissa_eigenvalue.imag),
                        "max_real_eigenvalue_abs": float(abscissa_eigenvalue.abs()),
                        "operator_norm_2": float(sigma_values[0]),
                        "operator_norm_sigma2": float(sigma_values[1]),
                        "operator_norm_over_spectral_radius": float(
                            sigma_values[0] / radius_value.clamp_min(1e-12)
                        ),
                        "runtime_seconds": time.perf_counter() - started,
                    }
                )
                completed.add(key)
                write_csv(part_path, rows)
                print(
                    json.dumps(
                        {
                            "stage": "jacobian_geometry",
                            "checkpoint": checkpoint,
                            "dynamic_step": dynamic_step,
                            "token": token["token"],
                            "spectral_radius": rows[-1]["spectral_radius"],
                            "spectral_abscissa": rows[-1]["spectral_abscissa"],
                            "operator_norm_2": rows[-1]["operator_norm_2"],
                            "seconds": rows[-1]["runtime_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        del model
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "stage": "jacobian_geometry",
                    "checkpoint": checkpoint,
                    "completed": checkpoint_index + 1,
                    "total": len(checkpoints),
                }
            ),
            flush=True,
        )

    combined: list[dict] = []
    incomplete: list[str] = []
    for checkpoint in CHECKPOINTS:
        part_path = parts / f"{checkpoint}.csv"
        if not part_path.exists():
            incomplete.append(checkpoint)
            continue
        part_rows = read_csv(part_path)
        combined.extend(part_rows)
        if len(part_rows) != expected_per_checkpoint:
            incomplete.append(checkpoint)
    suffix = ".partial.csv" if incomplete else ".csv"
    write_csv(processed / f"window_endpoint_jacobian_three_metrics{suffix}", combined)
    if incomplete:
        print(
            json.dumps(
                {
                    "stage": "jacobian_geometry",
                    "status": "partial",
                    "incomplete": incomplete,
                }
            ),
            flush=True,
        )
        return
    plot_jacobian_metric(
        combined,
        tokens,
        "spectral_radius",
        "spectral radius ρ(J)",
        "Jacobian spectral radius: max |eigenvalue|",
        figures / "jacobian_spectral_radius_detailed.png",
    )
    plot_jacobian_metric(
        combined,
        tokens,
        "spectral_abscissa",
        "spectral abscissa α(J)",
        "Jacobian maximum eigenvalue real part: max Re(eigenvalue)",
        figures / "jacobian_max_real_eigenvalue_detailed.png",
    )
    plot_jacobian_metric(
        combined,
        tokens,
        "operator_norm_2",
        "operator 2-norm ||J||₂",
        "Jacobian operator norm: largest singular value",
        figures / "jacobian_operator_norm_detailed.png",
    )
    plot_three_jacobian_metrics(
        combined,
        tokens,
        figures / "jacobian_three_metrics_by_window.png",
    )


def jacobian_stage(args, checkpoints: list[str], tokens: list[dict]) -> None:
    processed = args.report_root / "processed"
    parts = processed / "checkpoint_parts"
    parts.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m",
        revision="step100000",
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )
    counts = wikitext_counts(args.wikitext_train_arrow, tokenizer)
    endpoints = window_endpoints(args.steps, args.window)
    expected_per_checkpoint = len(endpoints) * len(tokens)

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        part_path = parts / f"{checkpoint}.csv"
        rows = [] if args.overwrite or not part_path.exists() else read_csv(part_path)
        completed = {
            (int(row["dynamic_step"]), int(row["token_id"]))
            for row in rows
        }
        if len(completed) == expected_per_checkpoint:
            print(
                json.dumps(
                    {"stage": "jacobian", "checkpoint": checkpoint, "status": "skip"}
                ),
                flush=True,
            )
            continue
        states = load_states(args, checkpoint, tokens)
        model = load_model(checkpoint, args.cache_dir, args.device)
        for dynamic_step in endpoints:
            for token_index, token in enumerate(tokens):
                key = (dynamic_step, int(token["token_id"]))
                if key in completed:
                    continue
                state = states[dynamic_step, token_index].to(args.device)
                started = time.perf_counter()
                jacobian = exact_jacobian(model, state, args.jacobian_chunk_size)
                eigenvalues = torch.linalg.eigvals(jacobian)
                order = torch.argsort(eigenvalues.abs(), descending=True)
                first = eigenvalues[order[0]]
                second = eigenvalues[order[1]]
                metrics = endpoint_token_metrics(state, model, tokenizer, counts)
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "window_start": dynamic_step - args.window,
                        "window_end": dynamic_step,
                        "dynamic_step": dynamic_step,
                        **token,
                        "jacobian_lambda1_abs": float(first.abs()),
                        "jacobian_lambda1_real": float(first.real),
                        "jacobian_lambda1_imag": float(first.imag),
                        "jacobian_lambda2_abs": float(second.abs()),
                        **metrics,
                        "jacobian_runtime_seconds": time.perf_counter() - started,
                    }
                )
                completed.add(key)
                write_csv(part_path, rows)
                print(
                    json.dumps(
                        {
                            "stage": "jacobian",
                            "checkpoint": checkpoint,
                            "dynamic_step": dynamic_step,
                            "token": token["token"],
                            "lambda1_abs": rows[-1]["jacobian_lambda1_abs"],
                            "seconds": rows[-1]["jacobian_runtime_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        del model
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "stage": "jacobian",
                    "checkpoint": checkpoint,
                    "completed": checkpoint_index + 1,
                    "total": len(checkpoints),
                }
            ),
            flush=True,
        )

    combined: list[dict] = []
    incomplete: list[str] = []
    for checkpoint in CHECKPOINTS:
        part_path = parts / f"{checkpoint}.csv"
        if not part_path.exists():
            incomplete.append(checkpoint)
            continue
        rows = read_csv(part_path)
        if len(rows) != expected_per_checkpoint:
            incomplete.append(checkpoint)
            combined.extend(rows)
            continue
        combined.extend(rows)
    if incomplete:
        add_endpoint_projections(args, combined, tokens)
        write_csv(processed / "window_endpoint_jacobian_token_metrics.partial.csv", combined)
        print(
            json.dumps(
                {"stage": "jacobian", "status": "partial", "incomplete": incomplete}
            ),
            flush=True,
        )
    else:
        add_endpoint_projections(args, combined, tokens)
        write_csv(processed / "window_endpoint_jacobian_token_metrics.csv", combined)
        plot_jacobian_spectral_radius(
            combined,
            CHECKPOINTS,
            tokens,
            args.report_root / "figures" / "jacobian_spectral_radius_by_window.png",
        )


def markdown_text(value: str) -> str:
    return repr(value).replace("|", "\\|")


def neighbor_table(
    rows: list[dict],
    tokens: list[dict],
    endpoints: list[int],
    metric: str,
) -> str:
    token_ids = [int(token["token_id"]) for token in tokens]
    token_headers = [markdown_text(token["token"]) for token in tokens]
    lines = [
        "| checkpoint | dynamic step | " + " | ".join(token_headers) + " |",
        "|---|---:|" + "|".join("---" for _ in tokens) + "|",
    ]
    indexed = {
        (row["checkpoint"], int(row["dynamic_step"]), int(row["token_id"])): row
        for row in rows
    }
    for checkpoint in CHECKPOINTS:
        for dynamic_step in endpoints:
            cells: list[str] = []
            for token_id in token_ids:
                row = indexed[(checkpoint, dynamic_step, token_id)]
                if metric == "cosine":
                    cell = (
                        f"{markdown_text(row['cosine_token'])}<br>"
                        f"sim={float(row['cosine_similarity']):.5g}; "
                        f"margin={float(row['cosine_margin']):.3g}; "
                        f"freq={int(row['cosine_wikitext_count'])}"
                    )
                elif metric == "euclidean":
                    cell = (
                        f"{markdown_text(row['euclidean_token'])}<br>"
                        f"d={float(row['euclidean_distance']):.5g}; "
                        f"rel-margin={float(row['euclidean_relative_margin']):.3g}; "
                        f"freq={int(row['euclidean_wikitext_count'])}"
                    )
                elif metric == "lm_head":
                    cell = (
                        f"{markdown_text(row['lm_head_token'])}<br>"
                        f"p={float(row['lm_head_probability']):.4g}; "
                        f"Δp={float(row['lm_head_probability_margin']):.3g}; "
                        f"Δlogit={float(row['lm_head_logit_margin']):.3g}; "
                        f"freq={int(row['lm_head_wikitext_count'])}"
                    )
                else:
                    raise ValueError(f"unknown neighbor metric: {metric}")
                cells.append(cell)
            lines.append(
                f"| {checkpoint} | {dynamic_step} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def jacobian_geometry_report(args, tokens: list[dict]) -> str:
    path = (
        args.report_root
        / "processed"
        / "window_endpoint_jacobian_three_metrics.csv"
    )
    if not path.exists():
        return ""
    rows = read_csv(path)
    lines = [
        "## 4. Jacobian的三个不同尺度",
        "",
        "对实Jacobian，特征值仍可能是复数。写作 `λᵢ=aᵢ+bᵢi`，其中",
        "`aᵢ=Re(λᵢ)` 是实部，`bᵢ=Im(λᵢ)` 是虚部，",
        "`|λᵢ|=sqrt(aᵢ²+bᵢ²)` 是模。三个尺度定义如下。",
        "",
        "### 4.1 谱半径：最大特征值模",
        "",
        "`ρ(J)=maxᵢ|λᵢ|`。它先计算每个复特征值的模，再取最大值；因此",
        "`spectral_radius` 是非负实数，而不是某个特征值的实部。",
        "",
        "- `spectral_radius`：最大模；",
        "- `spectral_radius_eigenvalue_real`：达到最大模的那个特征值的实部；",
        "- `spectral_radius_eigenvalue_imag`：同一个特征值的虚部。",
        "",
        "离散动力系统的局部渐近稳定性主要看谱半径：`ρ(J)<1` 表示所有",
        "线性特征模态渐近收缩，`ρ(J)>1` 表示至少一个模态局部扩张。",
        "",
        "### 4.2 最大特征值：最大实部（谱横坐标）",
        "",
        "复数没有天然的大小顺序，因此报告把“最大特征值”明确定义为",
        "`α(J)=maxᵢ Re(λᵢ)`，也称谱横坐标。它按实部排序，不按模排序；",
        "达到最大实部的特征值不一定是达到最大模的同一个特征值。",
        "",
        "- `spectral_abscissa`：最大实部，本身就是对应特征值的实部；",
        "- `max_real_eigenvalue_imag`：达到最大实部的那个特征值的虚部；",
        "- `max_real_eigenvalue_abs`：该特征值的模。",
        "",
        "虚部不表示扩张强度，而表示线性化模态中的旋转/振荡成分。对当前",
        "离散dynamic step，是否收缩仍应优先看模和谱半径，不能只看实部。",
        "",
        "### 4.3 算子二范数：最大奇异值",
        "",
        "`||J||₂=σmax(J)`。奇异值是非负实数，所以算子范数没有实部和",
        "虚部。它给出任意单位扰动经过单次线性映射后可能达到的最大长度。",
        "",
        "- `operator_norm_2`：最大奇异值 `σ₁`；",
        "- `operator_norm_sigma2`：第二大奇异值 `σ₂`；",
        "- `operator_norm_over_spectral_radius`：`σ₁/ρ(J)`。",
        "",
        "即使 `ρ(J)<1`，也可能有 `||J||₂>1`：这表示所有特征模态最终收缩，",
        "但某些方向上的扰动可以先发生单步或短期瞬时放大。这通常来自Jacobian",
        "的非正规性。`||J||₂/ρ(J)` 越大，这种瞬时放大与渐近尺度的差异越明显。",
        "",
        "### 4.4 三者对照",
        "",
        "| 指标 | 取什么量的最大值 | 模/实部/虚部 | 主要含义 |",
        "|---|---|---|---|",
        "| `spectral_radius` | `max |λᵢ|` | 特征值的**模** | 离散系统渐近收缩或扩张 |",
        "| `spectral_abscissa` | `max Re(λᵢ)` | 特征值的**实部** | 最靠右的复谱位置 |",
        "| `operator_norm_2` | `max σᵢ` | 奇异值；无实部/虚部 | 单步最大扰动放大率 |",
        "| `*_eigenvalue_imag` | 不取最大；记录被选中特征值 | 特征值的**虚部** | 旋转或振荡成分 |",
        "",
        "### 4.5 各checkpoint观测范围",
        "",
        "下表中每个range都汇总该checkpoint的 `8个dynamic-step窗口端点 × "
        "4个token = 32个Jacobian`。`[a,b]` 表示这32个样本中的最小值为",
        "`a`、最大值为 `b`，不是置信区间，也不是误差条。",
        "",
        "- **spectral radius range**：32个 `ρ(J)=max|λᵢ|` 的",
        "  `[最小谱半径, 最大谱半径]`，两端都是特征值模的统计；",
        "- **max Re(eigenvalue) range**：32个 `α(J)=max Re(λᵢ)` 的",
        "  `[最小谱横坐标, 最大谱横坐标]`，两端都是实部的统计；",
        "- **operator norm range**：32个 `||J||₂=σmax(J)` 的",
        "  `[最小算子二范数, 最大算子二范数]`，两端都是奇异值；",
        "- **norm/radius range**：对每个Jacobian先计算",
        "  `||J||₂/ρ(J)`，再报告32个逐样本比值的 `[最小值, 最大值]`；",
        "  它不是 `operator norm range` 的端点除以 `spectral radius range`",
        "  的端点。比值接近1表示接近正规算子的尺度关系；明显大于1表示",
        "  单步最大放大显著强于特征值给出的渐近尺度。",
        "",
        "| checkpoint | spectral radius range | max Re(eigenvalue) range | operator norm range | norm/radius range |",
        "|---|---:|---:|---:|---:|",
    ]
    for checkpoint in CHECKPOINTS:
        group = [row for row in rows if row["checkpoint"] == checkpoint]

        def value_range(field: str) -> str:
            values = [float(row[field]) for row in group]
            return f"[{min(values):.5g}, {max(values):.5g}]"

        lines.append(
            f"| {checkpoint} | {value_range('spectral_radius')} | "
            f"{value_range('spectral_abscissa')} | "
            f"{value_range('operator_norm_2')} | "
            f"{value_range('operator_norm_over_spectral_radius')} |"
        )
    final_step = max(int(row["dynamic_step"]) for row in rows)
    lines.extend(
        [
            "",
            f"### 4.6 最后一个窗口端点的range（t={final_step}）",
            "",
            f"最后一个窗口是 `{final_step - args.window}–{final_step}`，当前实验的",
            f"Jacobian取在其右端点 `t={final_step}`。下表只汇总该端点的4个",
            "token Jacobian；每个 `[a,b]` 分别是4个token中的最小值和最大值。",
            "列结构与上面的全8端点range表完全一致。",
            "",
            "| checkpoint | spectral radius range | max Re(eigenvalue) range | operator norm range | norm/radius range |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for checkpoint in CHECKPOINTS:
        final_rows = [
            row
            for row in rows
            if row["checkpoint"] == checkpoint
            and int(row["dynamic_step"]) == final_step
        ]

        def final_range(field: str) -> str:
            values = [float(row[field]) for row in final_rows]
            return f"[{min(values):.6g}, {max(values):.6g}]"

        lines.append(
            f"| {checkpoint} | {final_range('spectral_radius')} | "
            f"{final_range('spectral_abscissa')} | "
            f"{final_range('operator_norm_2')} | "
            f"{final_range('operator_norm_over_spectral_radius')} |"
        )
    lines.extend(
        [
            "",
            "对应图：",
            "",
            "- [`jacobian_spectral_radius_detailed.png`](figures/jacobian_spectral_radius_detailed.png)",
            "- [`jacobian_max_real_eigenvalue_detailed.png`](figures/jacobian_max_real_eigenvalue_detailed.png)",
            "- [`jacobian_operator_norm_detailed.png`](figures/jacobian_operator_norm_detailed.png)",
            "- [`jacobian_three_metrics_by_window.png`](figures/jacobian_three_metrics_by_window.png)",
            "",
        ]
    )
    return "\n".join(lines)


def report_stage(args, tokens: list[dict]) -> None:
    metrics_path = (
        args.report_root
        / "processed"
        / "window_endpoint_jacobian_token_metrics.csv"
    )
    rows = read_csv(metrics_path)
    expected = len(CHECKPOINTS) * len(window_endpoints(args.steps, args.window)) * len(tokens)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete metrics table: {len(rows)} != {expected}")
    endpoints = window_endpoints(args.steps, args.window)
    report = f"""# 实验15报告：窗口Jacobian与最近词

每行对应一个 `checkpoint / dynamic step`，四个数据列对应四个固定初始token。
所有最近词均由该窗口右端点的同一个hidden state计算。`freq` 是该最近词在
WikiText-2 train tokenization中的出现次数。

## 1. Cosine最近词

- `sim`：hidden state与input embedding的cosine similarity，越大越近；
- `margin`：top1 similarity减top2 similarity，越大表示top1分离越明确。

{neighbor_table(rows, tokens, endpoints, "cosine")}

## 2. Euclidean最近词

- `d`：hidden state到input embedding的欧式距离，越小越近；
- `rel-margin=(d2-d1)/d1`，越大表示top1分离越明确。

{neighbor_table(rows, tokens, endpoints, "euclidean")}

## 3. LM-head最近词

- `p`：LM-head top1 softmax概率；
- `Δp`：top1与top2概率差；
- `Δlogit`：top1与top2 logit差。

{neighbor_table(rows, tokens, endpoints, "lm_head")}

{jacobian_geometry_report(args, tokens)}
"""
    (args.report_root / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "dynamics",
            "projection",
            "jacobian",
            "jacobian_geometry",
            "report",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--upstream-states-root",
        type=Path,
        default=DEFAULT_UPSTREAM_STATES_ROOT,
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--wikitext-train-arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--token-manifest", type=Path, default=EXPERIMENT13_TOKENS)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:7"))
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--checkpoints", nargs="+", choices=CHECKPOINTS)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--projection-seed", type=int, default=1414)
    parser.add_argument("--projection-display-floor", type=float, default=1e-6)
    parser.add_argument("--jacobian-chunk-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    checkpoints = (args.checkpoints or CHECKPOINTS)[: args.max_checkpoints]
    tokens = load_tokens(args.token_manifest, args.max_tokens)
    window_endpoints(args.steps, args.window)
    if args.stage in {"dynamics", "all"}:
        dynamics_stage(args, checkpoints, tokens)
    if args.stage in {"projection", "all"}:
        projection_stage(args, checkpoints, tokens)
    if args.stage in {"jacobian", "all"}:
        jacobian_stage(args, checkpoints, tokens)
    if args.stage in {"jacobian_geometry", "all"}:
        jacobian_geometry_stage(args, checkpoints, tokens)
    if args.stage in {"report", "all"}:
        report_stage(args, tokens)


if __name__ == "__main__":
    main()
