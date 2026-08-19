#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts._bootstrap import require_packages

require_packages(["numpy", "torch", "transformers", "yaml"])

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM

from src.attractor_validation import (
    atomic_torch_save,
    best_two_lags,
    classify_screen,
    cyclic_orbit_distance,
    final_classification,
    finite_time_lyapunov,
    floquet_summary,
    interpolate_state_dict,
    lag_curve,
    minimal_repeated_period,
    multiple_shooting,
    perturbation_recovery,
    phase_invariant_orbit_distance,
    sampled_orbit_scale,
    tensor_sha256,
)
from src.experiment_io import atomic_json


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/experiment19.yaml"


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return value


def read_tokens(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    tokens = [
        {
            "selection_index": int(row["selection_index"]),
            "token_id": int(row["token_id"]),
            "token": row["token"],
            "wikitext_train_count": int(row["wikitext_train_count"]),
            "frequency_bin": int(row["frequency_bin"]),
        }
        for row in rows
    ]
    if len(tokens) != 8 or sorted(row["frequency_bin"] for row in tokens) != list(range(8)):
        raise RuntimeError("experiment19 requires exactly one token from every frequency bin")
    return tokens


def atomic_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def projection_basis(hidden: int, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn((hidden, count), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.T.float()


class PythiaAttractorOperator(torch.nn.Module):
    """Pythia-70m single-token map with explicit architecture controls."""

    VALID_MODES = {
        "full",
        "mlp_residual",
        "vo_residual",
        "no_internal_ln",
        "no_final_ln",
        "no_residual",
        "layer_shuffle",
    }

    def __init__(
        self,
        model: torch.nn.Module,
        mode: str = "full",
        layer_order: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown operator mode: {mode}")
        if not model.config.use_parallel_residual:
            raise ValueError("experiment19 assumes Pythia parallel residual blocks")
        self.model = model
        self.mode = mode
        self.layer_order = tuple(layer_order or range(len(model.gpt_neox.layers)))
        if sorted(self.layer_order) != list(range(len(model.gpt_neox.layers))):
            raise ValueError("layer_order must be a permutation of every layer")

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _vo(self, layer, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        heads = layer.attention.config.num_attention_heads
        head_size = layer.attention.head_size
        qkv = layer.attention.query_key_value(hidden).view(batch, length, heads, 3 * head_size)
        value = qkv.transpose(1, 2).chunk(3, dim=-1)[2]
        value = value.transpose(1, 2).reshape(batch, length, -1).contiguous()
        return layer.attention.dense(value)

    def _custom_forward(self, states: torch.Tensor) -> torch.Tensor:
        body = self.model.gpt_neox
        hidden = states.to(dtype=self.dtype).unsqueeze(1)
        position_ids = torch.zeros((1, 1), device=hidden.device, dtype=torch.long)
        position_embeddings = body.rotary_emb(hidden, position_ids)
        for index in self.layer_order:
            layer = body.layers[index]
            residual = hidden
            attention_input = residual if self.mode == "no_internal_ln" else layer.input_layernorm(residual)
            if self.mode == "mlp_residual":
                attention_output = torch.zeros_like(residual)
            elif self.mode == "vo_residual":
                attention_output = self._vo(layer, attention_input)
            else:
                attention_output = layer.attention(
                    attention_input,
                    attention_mask=None,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False,
                    cache_position=torch.zeros(1, device=hidden.device, dtype=torch.long),
                    position_embeddings=position_embeddings,
                )[0]
            mlp_input = residual if self.mode == "no_internal_ln" else layer.post_attention_layernorm(residual)
            mlp_output = torch.zeros_like(residual) if self.mode == "vo_residual" else layer.mlp(mlp_input)
            if self.mode == "mlp_residual":
                hidden = residual + mlp_output
            elif self.mode == "vo_residual":
                hidden = residual + attention_output
            elif self.mode == "no_residual":
                hidden = attention_output + mlp_output
            else:
                hidden = residual + attention_output + mlp_output
        if self.mode != "no_final_ln":
            hidden = body.final_layer_norm(hidden)
        return hidden[:, 0, :]

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2:
            raise ValueError("single-token states must have shape [batch,hidden]")
        if self.mode == "full":
            batch = len(states)
            output = self.model.gpt_neox(
                inputs_embeds=states.to(dtype=self.dtype).unsqueeze(1),
                attention_mask=torch.ones((batch, 1), device=states.device, dtype=torch.long),
                position_ids=torch.zeros((1, 1), device=states.device, dtype=torch.long),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, 0, :]
        return output

    def jvp_map(self, states: torch.Tensor) -> torch.Tensor:
        """JVP-safe map: explicit blocks avoid the transformer cache's fp16 dual tensors."""
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2:
            raise ValueError("single-token states must have shape [batch,hidden]")
        return self._custom_forward(states)
        return self._custom_forward(states)


def load_model(config: dict, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        revision=checkpoint,
        cache_dir=config["cache_dir"],
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=torch.float32)
    # Some cached Pythia revisions retain fp16 checkpoint tensors despite
    # torch_dtype.  The JVP/Floquet path requires a uniform fp32 operator.
    model.float()
    model.eval()
    model.set_attn_implementation("eager")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def random_model(config: dict, seed: int, device: torch.device) -> torch.nn.Module:
    specification = AutoConfig.from_pretrained(
        config["model"],
        revision="step0",
        cache_dir=config["cache_dir"],
        local_files_only=True,
    )
    torch.manual_seed(int(seed))
    model = AutoModelForCausalLM.from_config(specification).to(device)
    model.eval()
    model.set_attn_implementation("eager")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def token_embeddings(model: torch.nn.Module, tokens: Sequence[dict]) -> torch.Tensor:
    ids = torch.tensor([row["token_id"] for row in tokens], device=next(model.parameters()).device)
    return model.get_input_embeddings()(ids).detach()


def prepare_reference_banks(config: dict, tokens: Sequence[dict], device: torch.device) -> dict[str, torch.Tensor]:
    path = Path(config["data_root"]) / "manifests/reference_state_banks.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["token_ids"] == [row["token_id"] for row in tokens]:
            return {name: value.to(device) for name, value in payload["banks"].items()}
    model = load_model(config, "step0", device)
    common = token_embeddings(model, tokens).float()
    generator = torch.Generator(device="cpu").manual_seed(int(config["seeds"]["random_states"]))
    random = torch.randn(common.shape, generator=generator).to(device)
    random = torch.nn.functional.normalize(random, dim=-1) * torch.linalg.vector_norm(common, dim=-1).median()
    payload = {
        "token_ids": [row["token_id"] for row in tokens],
        "banks": {"common_step0": common.cpu(), "random_matched": random.cpu()},
        "seed": int(config["seeds"]["random_states"]),
        "common_sha256": tensor_sha256(common),
        "random_sha256": tensor_sha256(random),
    }
    atomic_torch_save(path, payload)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {name: value.to(device) for name, value in payload["banks"].items()}


def combine_banks(banks: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[tuple[str, int]]]:
    values: list[torch.Tensor] = []
    labels: list[tuple[str, int]] = []
    for name, bank in banks.items():
        values.append(bank)
        labels.extend((name, index) for index in range(len(bank)))
    return torch.cat(values, dim=0), labels


def run_trace(
    operator: PythiaAttractorOperator,
    initial: torch.Tensor,
    *,
    steps: int,
    projection: torch.Tensor,
    perturbation_seed: int,
    perturbation_scale: float = 1e-6,
) -> dict[str, torch.Tensor | float]:
    state = initial.detach()
    generator = torch.Generator(device="cpu").manual_seed(int(perturbation_seed))
    noise = torch.randn(state.shape, generator=generator).to(state.device)
    noise = torch.nn.functional.normalize(noise, dim=-1)
    scale = torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1.0)
    nearby = state + float(perturbation_scale) * scale * noise
    states = torch.empty((steps + 1, *state.shape), dtype=torch.float32, device="cpu")
    projections = torch.empty((steps + 1, len(state), projection.shape[0]), dtype=torch.float32)
    nearby_distance = torch.empty((steps + 1, len(state)), dtype=torch.float32)
    states[0] = state.detach().float().cpu()
    projections[0] = state.detach().float().cpu() @ projection.T
    nearby_distance[0] = torch.linalg.vector_norm(nearby - state, dim=-1).float().cpu()
    with torch.inference_mode():
        for step in range(1, int(steps) + 1):
            joined = operator(torch.cat([state, nearby], dim=0))
            state, nearby = joined.chunk(2, dim=0)
            states[step] = state.float().cpu()
            projections[step] = state.float().cpu() @ projection.T
            nearby_distance[step] = torch.linalg.vector_norm(nearby - state, dim=-1).float().cpu()
    norm = torch.linalg.vector_norm(states, dim=-1)
    delta = torch.linalg.vector_norm(states[1:] - states[:-1], dim=-1)
    return {
        "states": states,
        "projections": projections,
        "state_norm": norm,
        "step_delta": delta,
        "relative_step_delta": delta / norm[1:].clamp_min(1e-12),
        "nearby_distance": nearby_distance,
        "initial_nearby_distance": float(nearby_distance[0].median()),
    }


def logit_recurrence(
    model: torch.nn.Module,
    states: torch.Tensor,
    lag: int,
    *,
    chunk_size: int,
) -> dict[str, float]:
    output = model.get_output_embeddings()
    distances: list[torch.Tensor] = []
    cosine: list[torch.Tensor] = []
    js_values: list[torch.Tensor] = []
    for start in range(0, len(states) - lag, int(chunk_size)):
        stop = min(start + int(chunk_size), len(states) - lag)
        left = states[start:stop].to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        right = states[start + lag : stop + lag].to(left.device, dtype=left.dtype)
        with torch.no_grad():
            left_logits = output(left).float()
            right_logits = output(right).float()
            distances.append(torch.linalg.vector_norm(left_logits - right_logits, dim=-1).cpu())
            cosine.append((1.0 - torch.nn.functional.cosine_similarity(left_logits, right_logits, dim=-1)).cpu())
            left_logp = torch.log_softmax(left_logits, dim=-1)
            right_logp = torch.log_softmax(right_logits, dim=-1)
            left_p, right_p = left_logp.exp(), right_logp.exp()
            mean_logp = torch.logaddexp(left_logp, right_logp) - math.log(2.0)
            js = 0.5 * ((left_p * (left_logp - mean_logp)).sum(-1) + (right_p * (right_logp - mean_logp)).sum(-1))
            js_values.append(js.cpu())
    distance = torch.cat(distances)
    cosine_value = torch.cat(cosine)
    js = torch.cat(js_values)
    return {
        "logit_l2_median": float(distance.median()),
        "logit_l2_p95": float(torch.quantile(distance, 0.95)),
        "logit_cosine_median": float(cosine_value.median()),
        "logit_cosine_p95": float(torch.quantile(cosine_value, 0.95)),
        "logit_js_median": float(js.median()),
        "logit_js_p95": float(torch.quantile(js, 0.95)),
    }


def analyze_trace(
    config: dict,
    model: torch.nn.Module,
    operator: PythiaAttractorOperator,
    trace: dict,
    labels: Sequence[tuple[str, int]],
    tokens: Sequence[dict],
    *,
    tail_start: int,
    tail_end: int,
    max_lag: int,
    condition: str,
    checkpoint: str,
) -> tuple[list[dict], list[dict]]:
    states = trace["states"]
    if not isinstance(states, torch.Tensor):
        raise TypeError("trace states missing")
    tail = states[tail_start : tail_end + 1]
    lyapunov, _ = finite_time_lyapunov(
        operator.jvp_map,
        tail.to(next(model.parameters()).device),
        seed=int(config["seeds"]["perturbation"]),
    )
    summaries: list[dict] = []
    recurrence_rows: list[dict] = []
    half = len(tail) // 2
    for batch_index, (bank, token_index) in enumerate(labels):
        trajectory = tail[:, batch_index]
        scale = sampled_orbit_scale(
            trajectory,
            seed=int(config["seeds"]["recurrence"]) + batch_index,
        )
        curve = lag_curve(trajectory, max_lag=max_lag, normalization_scale=scale["normalization_scale"])
        best, second = best_two_lags(curve)
        window_lags: list[int] = []
        for segment in (trajectory[: half + 1], trajectory[half:]):
            segment_curve = lag_curve(segment, max_lag=max_lag, normalization_scale=scale["normalization_scale"])
            window_lags.append(int(best_two_lags(segment_curve)[0]["candidate_lag"]))
        lag1 = curve[0]
        screen_label = classify_screen(
            lag1_normalized_p95=float(lag1["normalized_p95"]),
            best_lag=int(best["candidate_lag"]),
            best_normalized_p95=float(best["normalized_p95"]),
            second_normalized_p95=float(second["normalized_p95"]),
            lyapunov=float(lyapunov[batch_index]),
            window_lags=window_lags,
        )
        token = tokens[token_index]
        common = {
            "checkpoint": checkpoint,
            "condition": condition,
            "initial_state_bank": bank,
            "tail_start": tail_start,
            "tail_end": tail_end,
            **token,
        }
        for row in curve:
            recurrence_rows.append({**common, **row})
        logits = logit_recurrence(
            model,
            trajectory,
            int(best["candidate_lag"]),
            chunk_size=int(config["analysis"]["logit_chunk_size"]),
        )
        nearby = trace["nearby_distance"]
        summaries.append(
            {
                **common,
                **scale,
                "lag1_normalized_p95": lag1["normalized_p95"],
                "best_lag": best["candidate_lag"],
                "best_normalized_p95": best["normalized_p95"],
                "second_lag": second["candidate_lag"],
                "second_normalized_p95": second["normalized_p95"],
                "lag_prominence_ratio": float(second["normalized_p95"]) / max(float(best["normalized_p95"]), 1e-12),
                "first_half_best_lag": window_lags[0],
                "second_half_best_lag": window_lags[1],
                "tail_lyapunov": float(lyapunov[batch_index]),
                "nearby_final_over_initial": float(nearby[tail_end, batch_index]) / max(float(nearby[tail_start, batch_index]), 1e-30),
                "screen_classification": screen_label,
                **logits,
            }
        )
    return summaries, recurrence_rows


def condition_paths(config: dict, stage: str, condition: str) -> tuple[Path, Path, Path]:
    root = Path(config["data_root"]) / stage / slug(condition)
    return root / "trajectory.pt", root / "summary.csv", root / "complete.json"


def execute_condition(
    config: dict,
    model: torch.nn.Module,
    operator: PythiaAttractorOperator,
    banks: dict[str, torch.Tensor],
    tokens: Sequence[dict],
    *,
    stage: str,
    condition: str,
    checkpoint: str,
    steps: int,
    tail_start: int,
    tail_end: int,
    max_lag: int,
    overwrite: bool,
) -> list[dict]:
    trajectory_path, summary_path, complete_path = condition_paths(config, stage, condition)
    if complete_path.exists() and summary_path.exists() and not overwrite:
        print(json.dumps({"stage": stage, "condition": condition, "status": "skip"}), flush=True)
        return read_csv(summary_path)
    initial, labels = combine_banks(banks)
    projection = projection_basis(initial.shape[-1], 4, int(config["seeds"]["projection"]))
    started = time.perf_counter()
    trace = run_trace(
        operator,
        initial,
        steps=int(steps),
        projection=projection,
        perturbation_seed=int(config["seeds"]["perturbation"]),
    )
    metadata = {
        "model": config["model"],
        "checkpoint": checkpoint,
        "condition": condition,
        "operator": operator.mode,
        "layer_order": list(operator.layer_order),
        "initial_state_labels": labels,
        "token_ids": [row["token_id"] for row in tokens],
        "dtype": str(operator.dtype),
        "seeds": config["seeds"],
        "projection_sha256": tensor_sha256(projection),
        "weight_sha256": model_hash(model),
        "steps": int(steps),
    }
    atomic_torch_save(trajectory_path, {**metadata, **trace})
    summaries, recurrence = analyze_trace(
        config,
        model,
        operator,
        trace,
        labels,
        tokens,
        tail_start=int(tail_start),
        tail_end=int(tail_end),
        max_lag=int(max_lag),
        condition=condition,
        checkpoint=checkpoint,
    )
    atomic_csv(summary_path, summaries)
    atomic_csv(summary_path.with_name("recurrence.csv"), recurrence)
    atomic_json(
        complete_path,
        {
            **metadata,
            "trajectory_path": str(trajectory_path),
            "summary_path": str(summary_path),
            "runtime_seconds": time.perf_counter() - started,
            "status": "complete",
        },
    )
    print(json.dumps({"stage": stage, "condition": condition, "status": "complete"}), flush=True)
    return summaries


def screen_stage(args, config: dict, tokens: Sequence[dict]) -> None:
    device = torch.device(args.device)
    references = prepare_reference_banks(config, tokens, device)
    all_rows: list[dict] = []
    for checkpoint in args.checkpoints:
        model = load_model(config, checkpoint, device)
        banks = {"native": token_embeddings(model, tokens), **references}
        rows = execute_condition(
            config,
            model,
            PythiaAttractorOperator(model, "full"),
            banks,
            tokens,
            stage="screen",
            condition=checkpoint,
            checkpoint=checkpoint,
            steps=int(config["screen"]["steps"]),
            tail_start=int(config["screen"]["tail_start"]),
            tail_end=int(config["screen"]["steps"]),
            max_lag=int(config["analysis"]["max_lag"]),
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_csv(ROOT / "processed/screen_summary.csv", all_rows)


def selected_validation_checkpoints(config: dict) -> list[str]:
    mandatory = list(config["validation"]["mandatory_checkpoints"])
    screen_rows = read_csv(ROOT / "processed/screen_summary.csv")
    candidates: dict[str, float] = {}
    for row in screen_rows:
        checkpoint = row["checkpoint"]
        if checkpoint in mandatory or row["screen_classification"] != "recurrent_candidate":
            continue
        score = float(row["best_normalized_p95"])
        candidates[checkpoint] = min(score, candidates.get(checkpoint, float("inf")))
    extra = [key for key, _ in sorted(candidates.items(), key=lambda item: item[1])[:2]]
    return mandatory + [value for value in extra if value not in mandatory]


def validate_condition(
    args,
    config: dict,
    model: torch.nn.Module,
    operator: PythiaAttractorOperator,
    tokens: Sequence[dict],
    *,
    checkpoint: str,
    condition: str,
    banks: dict[str, torch.Tensor],
    source_stage: str = "validate",
) -> tuple[list[dict], list[dict], dict[str, torch.Tensor]]:
    summaries = execute_condition(
        config,
        model,
        operator,
        banks,
        tokens,
        stage=source_stage,
        condition=condition,
        checkpoint=checkpoint,
        steps=int(config["validation"]["steps"]),
        tail_start=int(config["validation"]["tail_start"]),
        tail_end=int(config["validation"]["steps"]),
        max_lag=int(config["analysis"]["max_lag"]),
        overwrite=args.overwrite,
    )
    trajectory_path, _, _ = condition_paths(config, source_stage, condition)
    trace = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    labels = [(str(name), int(index)) for name, index in trace["initial_state_labels"]]
    solutions: dict[str, torch.Tensor] = {}
    orbit_rows: list[dict] = []
    floquet_rows: list[dict] = []
    orbit_path = trajectory_path.with_name("orbit_solutions.pt")
    if orbit_path.exists() and not args.overwrite:
        stored = torch.load(orbit_path, map_location="cpu", weights_only=True)
        return summaries, stored["rows"], stored["solutions"]
    for row in summaries:
        if row["screen_classification"] not in {"fixed_candidate", "recurrent_candidate"}:
            continue
        bank = row["initial_state_bank"]
        token_index = int(row["selection_index"])
        batch_index = labels.index((bank, token_index))
        candidate_period = 1 if row["screen_classification"] == "fixed_candidate" else int(row["best_lag"])
        trajectory = trace["states"][:, batch_index].to(next(model.parameters()).device)
        scale = float(row["normalization_scale"])
        start = int(config["validation"]["tail_start"])
        for phase_run, offset in enumerate((0, candidate_period // 3, 2 * candidate_period // 3)):
            initial = trajectory[start + offset : start + offset + candidate_period]
            if len(initial) != candidate_period:
                continue
            result = multiple_shooting(
                operator,
                initial,
                orbit_scale=scale,
                max_iterations=int(config["shooting"]["max_iterations"]),
                tolerance=float(config["shooting"]["tolerance"]),
            )
            key = f"{condition}__{bank}__token{row['token_id']}__phase{phase_run}"
            solutions[key] = result.orbit.detach().cpu()
            minimal = result.minimal_period
            floquet = floquet_summary(
                operator,
                result.orbit[:minimal],
                seed=int(config["seeds"]["floquet"]) + phase_run,
                dimensions=tuple(config["floquet"]["krylov_dimensions"]),
            )
            recovery = perturbation_recovery(
                operator,
                result.orbit[:minimal],
                seed=int(config["seeds"]["perturbation"]) + phase_run,
                directions=int(config["perturbation"]["directions"]),
                relative_scales=tuple(config["perturbation"]["relative_scales"]),
                periods=int(config["perturbation"]["periods"]),
            )
            common = {
                "checkpoint": checkpoint,
                "condition": condition,
                "initial_state_bank": bank,
                "token_id": int(row["token_id"]),
                "token": row["token"],
                "phase_run": phase_run,
                "candidate_period": candidate_period,
                "minimal_period": minimal,
                "shooting_loss": result.loss,
                "shooting_normalized_residual_p95": result.normalized_residual_p95,
                "shooting_converged": result.converged,
                "shooting_iterations": result.iterations,
                "solution_key": key,
            }
            orbit_rows.append({**common, **floquet, **recovery})
            floquet_rows.append({**common, **floquet})
    atomic_torch_save(orbit_path, {"rows": orbit_rows, "solutions": solutions})
    atomic_csv(orbit_path.with_name("orbit_candidates.csv"), orbit_rows)
    atomic_csv(orbit_path.with_name("floquet_metrics.csv"), floquet_rows)
    return summaries, orbit_rows, solutions


def validate_stage(args, config: dict, tokens: Sequence[dict]) -> None:
    device = torch.device(args.device)
    references = prepare_reference_banks(config, tokens, device)
    checkpoints = selected_validation_checkpoints(config)
    all_orbits: list[dict] = []
    all_floquet: list[dict] = []
    solution_bank: dict[str, torch.Tensor] = {}
    precision_rows: list[dict] = []
    for checkpoint in checkpoints:
        model = load_model(config, checkpoint, device)
        banks = {"native": token_embeddings(model, tokens), **references}
        _, rows, solutions = validate_condition(
            args,
            config,
            model,
            PythiaAttractorOperator(model, "full"),
            tokens,
            checkpoint=checkpoint,
            condition=checkpoint,
            banks=banks,
        )
        all_orbits.extend(rows)
        all_floquet.extend(rows)
        solution_bank.update(solutions)
        native_rows = [
            row for row in rows
            if row["initial_state_bank"] == "native" and bool(row["shooting_converged"])
        ]
        if native_rows:
            best = min(native_rows, key=lambda row: float(row["shooting_normalized_residual_p95"]))
            orbit = solutions[best["solution_key"]]
            period = int(best["minimal_period"])
            scale = max(float(torch.linalg.vector_norm(orbit - orbit.mean(0), dim=1).median()), 1e-12)
            fp32_residual = float(best["shooting_normalized_residual_p95"])
            model.double()
            double_operator = PythiaAttractorOperator(model, "full")
            state = orbit[0].double().to(device).unsqueeze(0)
            returns: list[float] = []
            with torch.inference_mode():
                for step in range(1, 4 * period + 1):
                    state = double_operator(state)
                    if step % period == 0:
                        returns.append(float(torch.linalg.vector_norm(state[0].cpu() - orbit[0].double())) / scale)
            fp64_residual = max(returns)
            precision_rows.append(
                {
                    "checkpoint": checkpoint,
                    "token_id": best["token_id"],
                    "period": period,
                    "fp32_normalized_residual_p95": fp32_residual,
                    "fp64_max_return_error_4p": fp64_residual,
                    "precision_consistent": fp64_residual <= max(10.0 * fp32_residual, 1e-4),
                    "projection_seed_a": int(config["seeds"]["projection"]),
                    "projection_seed_b": int(config["seeds"]["projection"]) + 1,
                    "classification_uses_projection": False,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_csv(ROOT / "processed/orbit_candidates.csv", all_orbits)
    atomic_csv(ROOT / "processed/floquet_metrics.csv", all_floquet)
    atomic_csv(ROOT / "processed/precision_audit.csv", precision_rows)
    atomic_torch_save(Path(config["data_root"]) / "validate/orbit_solutions.pt", solution_bank)


def controls_stage(args, config: dict, tokens: Sequence[dict]) -> None:
    device = torch.device(args.device)
    references = prepare_reference_banks(config, tokens, device)
    rows: list[dict] = []
    validation_rows: list[dict] = []
    modes = list(config["controls"]["modes"])
    permutations = config["controls"]["layer_orders"]
    for checkpoint in config["controls"]["checkpoints"]:
        model = load_model(config, checkpoint, device)
        banks = {"native": token_embeddings(model, tokens), **references}
        conditions: list[tuple[str, Sequence[int] | None]] = []
        for mode in modes:
            if mode == "layer_shuffle":
                conditions.extend((f"layer_shuffle_{index}", order) for index, order in enumerate(permutations))
            else:
                conditions.append((mode, None))
        for mode_label, order in conditions:
            mode = "layer_shuffle" if mode_label.startswith("layer_shuffle") else mode_label
            condition = f"{checkpoint}__{mode_label}"
            operator = PythiaAttractorOperator(model, mode, order)
            current = execute_condition(
                config,
                model,
                operator,
                banks,
                tokens,
                stage="controls_screen",
                condition=condition,
                checkpoint=checkpoint,
                steps=int(config["screen"]["steps"]),
                tail_start=int(config["screen"]["tail_start"]),
                tail_end=int(config["screen"]["steps"]),
                max_lag=int(config["analysis"]["max_lag"]),
                overwrite=args.overwrite,
            )
            for row in current:
                row["operator_mode"] = mode_label
            rows.extend(current)
            if any(row["screen_classification"] == "recurrent_candidate" for row in current):
                _, strict_rows, _ = validate_condition(
                    args,
                    config,
                    model,
                    operator,
                    tokens,
                    checkpoint=checkpoint,
                    condition=condition,
                    banks=banks,
                    source_stage="controls_validate",
                )
                for row in strict_rows:
                    row["operator_mode"] = mode_label
                validation_rows.extend(strict_rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for seed in config["controls"]["random_init_seeds"]:
        model = random_model(config, int(seed), device)
        banks = {"native": token_embeddings(model, tokens), **references}
        condition = f"random_init_seed{seed}"
        current = execute_condition(
            config,
            model,
            PythiaAttractorOperator(model, "full"),
            banks,
            tokens,
            stage="controls_screen",
            condition=condition,
            checkpoint="random_init",
            steps=int(config["screen"]["steps"]),
            tail_start=int(config["screen"]["tail_start"]),
            tail_end=int(config["screen"]["steps"]),
            max_lag=int(config["analysis"]["max_lag"]),
            overwrite=args.overwrite,
        )
        for row in current:
            row["operator_mode"] = condition
        rows.extend(current)
        if any(row["screen_classification"] == "recurrent_candidate" for row in current):
            _, strict_rows, _ = validate_condition(
                args,
                config,
                model,
                PythiaAttractorOperator(model, "full"),
                tokens,
                checkpoint="random_init",
                condition=condition,
                banks=banks,
                source_stage="controls_validate",
            )
            validation_rows.extend(strict_rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_csv(ROOT / "processed/architecture_controls.csv", rows)
    atomic_csv(ROOT / "processed/architecture_control_orbits.csv", validation_rows)


def interpolated_model(
    config: dict,
    specification,
    left_state: dict[str, torch.Tensor],
    right_state: dict[str, torch.Tensor],
    alpha: float,
    device: torch.device,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_config(specification)
    model.load_state_dict(interpolate_state_dict(left_state, right_state, float(alpha)), strict=True)
    model = model.to(device)
    model.eval()
    model.set_attn_implementation("eager")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def endpoint_output_error(
    work: torch.nn.Module,
    reference: torch.nn.Module,
    states: torch.Tensor,
) -> float:
    work_device = next(work.parameters()).device
    reference_device = next(reference.parameters()).device
    with torch.no_grad():
        expected = PythiaAttractorOperator(reference, "full")(states.to(reference_device)).float().cpu()
        actual = PythiaAttractorOperator(work, "full")(states.to(work_device)).float().cpu()
    return float((actual - expected).abs().max())


def dominant_labels(rows: Sequence[dict]) -> tuple[str, ...]:
    return tuple(sorted(str(row["screen_classification"]) for row in rows if row["initial_state_bank"] == "joint"))


def interpolation_alphas(config: dict) -> list[float]:
    step = float(config["bifurcation"]["coarse_alpha_step"])
    count = round(1.0 / step)
    return [round(index * step, 10) for index in range(count + 1)]


def _bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _checkpoint_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _bifurcation_alpha(
    args,
    config: dict,
    tokens: Sequence[dict],
    *,
    pair_name: str,
    left_checkpoint: str,
    right_checkpoint: str,
    alpha: float,
    specification,
    left_state: dict[str, torch.Tensor],
    right_state: dict[str, torch.Tensor],
    left_embedding: torch.Tensor,
    right_embedding: torch.Tensor,
    references: dict[str, torch.Tensor],
    left_reference: torch.nn.Module,
    right_reference: torch.nn.Module,
    device: torch.device,
) -> list[dict]:
    model = interpolated_model(
        config, specification, left_state, right_state, alpha, device
    )
    joint = token_embeddings(model, tokens)
    banks = {
        "joint": joint,
        "left_embedding": left_embedding.to(device),
        "right_embedding": right_embedding.to(device),
        **references,
    }
    condition = f"{pair_name}__alpha{alpha:.4f}"
    rows = execute_condition(
        config,
        model,
        PythiaAttractorOperator(model, "full"),
        banks,
        tokens,
        stage="bifurcation",
        condition=condition,
        checkpoint=condition,
        steps=int(config["screen"]["steps"]),
        tail_start=int(config["screen"]["tail_start"]),
        tail_end=int(config["screen"]["steps"]),
        max_lag=int(config["analysis"]["max_lag"]),
        overwrite=args.overwrite,
    )
    endpoint_error = None
    if abs(alpha) <= 1e-12:
        endpoint_error = endpoint_output_error(model, left_reference, joint)
    elif abs(alpha - 1.0) <= 1e-12:
        endpoint_error = endpoint_output_error(model, right_reference, joint)
    for row in rows:
        row["transition"] = pair_name
        row["left_checkpoint"] = left_checkpoint
        row["right_checkpoint"] = right_checkpoint
        row["alpha"] = alpha
        row["interpolation_is_diagnostic"] = True
        row["endpoint_max_abs_error"] = endpoint_error
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def bifurcation_stage(args, config: dict, tokens: Sequence[dict]) -> None:
    device = torch.device(args.device)
    references = prepare_reference_banks(config, tokens, device)
    output = ROOT / "processed/bifurcation_branches.csv"
    accumulated = [] if args.overwrite else read_csv(output)
    row_index = {
        (
            row.get("transition"),
            row.get("alpha"),
            row.get("initial_state_bank"),
            row.get("token_id"),
        ): row
        for row in accumulated
    }
    for left_checkpoint, right_checkpoint in config["bifurcation"]["pairs"]:
        pair_name = f"{left_checkpoint}_to_{right_checkpoint}"
        left_model = load_model(config, left_checkpoint, torch.device("cpu"))
        right_model = load_model(config, right_checkpoint, torch.device("cpu"))
        specification = left_model.config
        left_state = _checkpoint_state(left_model)
        right_state = _checkpoint_state(right_model)
        left_embedding = token_embeddings(left_model, tokens).cpu()
        right_embedding = token_embeddings(right_model, tokens).cpu()
        coarse_rows: dict[float, list[dict]] = {}
        for alpha in interpolation_alphas(config):
            current = _bifurcation_alpha(
                args,
                config,
                tokens,
                pair_name=pair_name,
                left_checkpoint=left_checkpoint,
                right_checkpoint=right_checkpoint,
                alpha=alpha,
                specification=specification,
                left_state=left_state,
                right_state=right_state,
                left_embedding=left_embedding,
                right_embedding=right_embedding,
                references=references,
                left_reference=left_model,
                right_reference=right_model,
                device=device,
            )
            coarse_rows[alpha] = current
            for row in current:
                key = (
                    row["transition"],
                    str(row["alpha"]),
                    row["initial_state_bank"],
                    str(row["token_id"]),
                )
                row_index[key] = row
            atomic_csv(output, row_index.values())
        fine_step = float(config["bifurcation"]["fine_alpha_step"])
        fine_alphas: set[float] = set()
        ordered = sorted(coarse_rows)
        for lower, upper in zip(ordered, ordered[1:]):
            if dominant_labels(coarse_rows[lower]) == dominant_labels(coarse_rows[upper]):
                continue
            count = round((upper - lower) / fine_step)
            fine_alphas.update(
                round(lower + index * fine_step, 10)
                for index in range(1, count)
            )
        for alpha in sorted(fine_alphas):
            current = _bifurcation_alpha(
                args,
                config,
                tokens,
                pair_name=pair_name,
                left_checkpoint=left_checkpoint,
                right_checkpoint=right_checkpoint,
                alpha=alpha,
                specification=specification,
                left_state=left_state,
                right_state=right_state,
                left_embedding=left_embedding,
                right_embedding=right_embedding,
                references=references,
                left_reference=left_model,
                right_reference=right_model,
                device=device,
            )
            for row in current:
                row["refined_alpha"] = True
                key = (
                    row["transition"],
                    str(row["alpha"]),
                    row["initial_state_bank"],
                    str(row["token_id"]),
                )
                row_index[key] = row
            atomic_csv(output, row_index.values())
        del left_model, right_model, left_state, right_state
        if device.type == "cuda":
            torch.cuda.empty_cache()


def analyze_stage(config: dict) -> None:
    screen_rows = read_csv(ROOT / "processed/screen_summary.csv")
    orbit_rows = read_csv(ROOT / "processed/orbit_candidates.csv")
    precision_rows = read_csv(ROOT / "processed/precision_audit.csv")
    orbit_index: dict[tuple[str, str, str], dict] = {}
    for row in orbit_rows:
        key = (
            row.get("checkpoint", ""),
            row.get("initial_state_bank", ""),
            row.get("token_id", ""),
        )
        previous = orbit_index.get(key)
        if previous is None or float(row.get("shooting_normalized_residual_p95", "inf")) < float(
            previous.get("shooting_normalized_residual_p95", "inf")
        ):
            orbit_index[key] = row
    precision_index = {
        (row.get("checkpoint", ""), row.get("token_id", "")): row
        for row in precision_rows
    }
    classified: list[dict] = []
    for row in screen_rows:
        key = (
            row.get("checkpoint", ""),
            row.get("initial_state_bank", ""),
            row.get("token_id", ""),
        )
        orbit = orbit_index.get(key)
        precision = precision_index.get((row.get("checkpoint", ""), row.get("token_id", "")))
        period = int(orbit["minimal_period"]) if orbit and orbit.get("minimal_period") else None
        residual = (
            float(orbit["shooting_normalized_residual_p95"])
            if orbit and orbit.get("shooting_normalized_residual_p95")
            else None
        )
        recovery = (
            float(orbit["recovery_fraction"])
            if orbit and orbit.get("recovery_fraction")
            else None
        )
        label = final_classification(
            screen_label=row["screen_classification"],
            period=period,
            shooting_residual=residual,
            floquet_stability=orbit.get("stability") if orbit else None,
            recovery_fraction=recovery,
            precision_consistent=_bool(precision.get("precision_consistent")) if precision else None,
        )
        classified.append(
            {
                **row,
                "minimal_period": period,
                "shooting_normalized_residual_p95": residual,
                "floquet_stability": orbit.get("stability") if orbit else None,
                "leading_multiplier_modulus": orbit.get("leading_multiplier_modulus") if orbit else None,
                "recovery_fraction": recovery,
                "precision_consistent": precision.get("precision_consistent") if precision else None,
                "final_classification": label,
            }
        )
    atomic_csv(ROOT / "processed/classification_summary.csv", classified)
    data_root = Path(config["data_root"])
    trajectory_rows: list[dict] = []
    for marker in data_root.rglob("complete.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trajectory_rows.append(
            {
                "stage": marker.parent.parent.name,
                "condition": marker.parent.name,
                "status": payload.get("status"),
                "checkpoint": payload.get("checkpoint"),
                "operator": payload.get("operator"),
                "steps": payload.get("steps"),
                "runtime_seconds": payload.get("runtime_seconds"),
                "trajectory_path": payload.get("trajectory_path"),
                "summary_path": payload.get("summary_path"),
                "complete_marker": str(marker),
            }
        )
    atomic_csv(ROOT / "processed/trajectory_index.csv", trajectory_rows)


def report_stage(config: dict) -> None:
    classifications = read_csv(ROOT / "processed/classification_summary.csv")
    controls = read_csv(ROOT / "processed/architecture_controls.csv")
    bifurcations = read_csv(ROOT / "processed/bifurcation_branches.csv")
    counts: dict[str, int] = defaultdict(int)
    for row in classifications:
        counts[row.get("final_classification", "unknown")] += 1
    control_counts: dict[str, int] = defaultdict(int)
    for row in controls:
        control_counts[row.get("screen_classification", "unknown")] += 1
    transition_changes: list[str] = []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in bifurcations:
        grouped[(row.get("transition", ""), row.get("initial_state_bank", ""), row.get("token_id", ""))].append(row)
    for (transition, bank, token), rows in grouped.items():
        ordered = sorted(rows, key=lambda row: float(row.get("alpha", 0)))
        labels = [row.get("screen_classification", "") for row in ordered]
        if len(set(labels)) > 1:
            transition_changes.append(f"- {transition}, {bank}, token {token}: " + " -> ".join(dict.fromkeys(labels)))
    lines = [
        "# Experiment 19: single-token attractor validation",
        "",
        "This report is generated from full-dimensional hidden-state evidence. Random projections are never used for classification.",
        "",
        "## Chapter 1: attractor validity",
        "",
    ]
    if counts:
        lines.extend(f"- `{label}`: {count}" for label, count in sorted(counts.items()))
    else:
        lines.append("- No validated classifications are available yet.")
    lines.extend(["", "## Chapter 3: architecture controls", ""])
    if control_counts:
        lines.extend(f"- `{label}`: {count}" for label, count in sorted(control_counts.items()))
    else:
        lines.append("- Architecture controls have not completed.")
    lines.extend(["", "## Chapter 4: diagnostic interpolation", ""])
    lines.extend(transition_changes or ["- No coarse classification transition has been recorded yet."])
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Weight interpolation is a diagnostic path, not the SGD trajectory.",
            "- Failed shooting, unstable Floquet estimates, and fp32/fp64 disagreement remain negative evidence.",
            "- A projected loop alone is not evidence of a periodic orbit.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("screen", "validate", "controls", "bifurcation", "analyze", "report", "all"),
        default="screen",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-manifest", type=Path)
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = read_config(args.config)
    args.checkpoints = args.checkpoints or list(config["checkpoints"])
    manifest = args.token_manifest or Path(config["token_manifest"])
    tokens = read_tokens(manifest)
    ROOT.joinpath("processed").mkdir(parents=True, exist_ok=True)
    Path(config["data_root"]).mkdir(parents=True, exist_ok=True)
    stages = (
        ("screen", "validate", "controls", "bifurcation", "analyze", "report")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        print(json.dumps({"stage": stage, "status": "start"}), flush=True)
        if stage == "screen":
            screen_stage(args, config, tokens)
        elif stage == "validate":
            validate_stage(args, config, tokens)
        elif stage == "controls":
            controls_stage(args, config, tokens)
        elif stage == "bifurcation":
            bifurcation_stage(args, config, tokens)
        elif stage == "analyze":
            analyze_stage(config)
        elif stage == "report":
            report_stage(config)
        print(json.dumps({"stage": stage, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
