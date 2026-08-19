#!/usr/bin/env python3
"""Dense checkpoint extension for Experiment 19.

Select checkpoint-level nontrivial periods from the existing dense screen,
refine one representative orbit per checkpoint, estimate Floquet stability,
and measure 256-step perturbation gain with 8 directions x 3 scales.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.attractor_validation import floquet_summary, multiple_shooting, sampled_orbit_scale

DENSE = REPO / "experiments_ordered/25_dense_checkpoint_suite"
DENSE_SCREEN = DENSE / "processed/period_screen_summary.csv"
DENSE_STATES = DENSE / "raw/states8"
ORIGINAL_SCREEN = EXP / "processed/screen_summary.csv"
TOKEN_MANIFEST = REPO / "experiments_ordered/18_fine_grained_window_jacobian/manifests/frequency_stratified_tokens_8.csv"
CACHE = Path("/home/luohaoming/model_feature_cache/hf_cache")
EPSILONS = (1e-6, 1e-4, 1e-2)
DIRECTIONS = 8
HORIZON = 256


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(value)
    return value


base = module("exp16_base_for_exp19_dense", REPO / "experiments_ordered/16_frequency_stratified_window_jacobian/scripts/run_experiment16.py")


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def token_names() -> dict[int, str]:
    return {int(row["token_id"]): row["token"] for row in read(TOKEN_MANIFEST)}


def selection() -> list[dict]:
    names = token_names()
    grouped: dict[str, list[dict]] = {}
    for row in read(DENSE_SCREEN):
        grouped.setdefault(row["checkpoint"], []).append(row)
    selected: list[dict] = []
    for checkpoint, rows in grouped.items():
        periods = [int(float(row["best_period"])) for row in rows]
        if statistics.median(periods) <= 1:
            continue
        candidates = [row for row in rows if int(float(row["best_period"])) > 1]
        representative = min(candidates, key=lambda row: float(row["best_period_error"]))
        token_id = int(representative["token_id"])
        selected.append({
            "checkpoint": checkpoint,
            "training_step": int(checkpoint[4:]),
            "selection_source": "experiment25_dense_period_screen",
            "checkpoint_period_median": statistics.median(periods),
            "nontrivial_token_count": len(candidates),
            "screen_periods": json.dumps(periods),
            "representative_token_id": token_id,
            "representative_token": names[token_id],
            "candidate_period": int(float(representative["best_period"])),
            "candidate_period_error": float(representative["best_period_error"]),
            "period_improvement_vs_p1": float(representative["period_improvement_vs_p1"]),
        })
    original = [row for row in read(ORIGINAL_SCREEN) if row["checkpoint"] == "step10000" and row["initial_state_bank"] == "native" and row["screen_classification"] == "recurrent_candidate"]
    representative = min(original, key=lambda row: float(row["best_normalized_p95"]))
    token_id = int(representative["token_id"])
    selected.append({
        "checkpoint": "step10000",
        "training_step": 10000,
        "selection_source": "experiment19_original_screen",
        "checkpoint_period_median": 101,
        "nontrivial_token_count": 8,
        "screen_periods": json.dumps([101] * 8),
        "representative_token_id": token_id,
        "representative_token": representative["token"],
        "candidate_period": int(float(representative["best_lag"])),
        "candidate_period_error": float(representative["best_normalized_p95"]),
        "period_improvement_vs_p1": float(representative["lag_prominence_ratio"]),
    })
    return sorted(selected, key=lambda row: row["training_step"])


def model_step(model, state: torch.Tensor) -> torch.Tensor:
    return model.gpt_neox(
        inputs_embeds=state.unsqueeze(1),
        attention_mask=torch.ones((len(state), 1), device=state.device, dtype=torch.long),
        position_ids=torch.zeros((1, 1), device=state.device, dtype=torch.long),
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[:, 0, :].float()


def load_representative_states(model, selected: dict, device: torch.device) -> torch.Tensor:
    checkpoint = selected["checkpoint"]
    token_id = int(selected["representative_token_id"])
    if checkpoint != "step10000":
        payload = torch.load(DENSE_STATES / f"{checkpoint}.pt", map_location="cpu", weights_only=True)
        index = payload["token_ids"].index(token_id)
        return payload["states"][:, index].float().to(device)
    state = model.get_input_embeddings()(torch.tensor([token_id], device=device)).detach().float()
    states = [state[0].cpu()]
    with torch.inference_mode():
        for _ in range(1024):
            state = model_step(model, state)
            states.append(state[0].cpu())
    return torch.stack(states).to(device)


def noise_seed(checkpoint: str, token_id: int, epsilon: float, direction: int) -> int:
    text = f"19dense256|{checkpoint}|{token_id}|{epsilon:.1e}|{direction}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)


def perturb_256(model, start: torch.Tensor, selected: dict) -> list[dict]:
    checkpoint = selected["checkpoint"]
    token_id = int(selected["representative_token_id"])
    descriptors = [(epsilon, direction) for epsilon in EPSILONS for direction in range(DIRECTIONS)]
    noise = []
    for epsilon, direction in descriptors:
        generator = torch.Generator(device="cpu").manual_seed(noise_seed(checkpoint, token_id, epsilon, direction))
        noise.append(torch.randn(start.numel(), generator=generator))
    unit = torch.nn.functional.normalize(torch.stack(noise), dim=-1).to(start.device)
    eps = torch.tensor([item[0] for item in descriptors], device=start.device).unsqueeze(1)
    reference = start.reshape(1, -1)
    perturbed = reference + eps * reference.norm(dim=-1, keepdim=True).clamp_min(1.0) * unit
    initial = torch.linalg.vector_norm(perturbed - reference, dim=-1)
    with torch.inference_mode():
        for _ in range(HORIZON):
            joined = model_step(model, torch.cat([reference, perturbed], dim=0))
            reference, perturbed = joined[:1], joined[1:]
    final = torch.linalg.vector_norm(perturbed - reference, dim=-1)
    reference_norm = reference.norm().clamp_min(1e-30)
    gain = final / initial.clamp_min(1e-30)
    output = []
    for index, (epsilon, direction) in enumerate(descriptors):
        output.append({
            "checkpoint": checkpoint,
            "training_step": selected["training_step"],
            "representative_token_id": token_id,
            "representative_token": selected["representative_token"],
            "candidate_period": selected["candidate_period"],
            "epsilon": epsilon,
            "direction_id": direction,
            "seed": noise_seed(checkpoint, token_id, epsilon, direction),
            "horizon_steps": HORIZON,
            "initial_distance": float(initial[index]),
            "final_distance": float(final[index]),
            "endpoint_relative_distance": float(final[index] / reference_norm),
            "response_gain": float(gain[index]),
            "log_response_gain": float(torch.log(gain[index].clamp_min(1e-30))),
            "response": "contracting" if float(gain[index]) < 1 else "amplifying",
        })
    return output


def summarize_perturbation(rows: list[dict]) -> dict:
    gains = [float(row["response_gain"]) for row in rows]
    logs = [math.log(max(value, 1e-30)) for value in gains]
    result = {
        "perturbation_count": len(rows),
        "perturbation_directions": DIRECTIONS,
        "perturbation_scales": json.dumps(EPSILONS),
        "perturbation_horizon_steps": HORIZON,
        "response_gain_mean": statistics.mean(gains),
        "response_gain_geometric_mean": math.exp(statistics.mean(logs)),
        "response_gain_median": statistics.median(gains),
        "response_gain_min": min(gains),
        "response_gain_max": max(gains),
        "mean_log_response_gain": statistics.mean(logs),
        "contraction_fraction": sum(value < 1 for value in gains) / len(gains),
    }
    for epsilon in EPSILONS:
        subset = [float(row["response_gain"]) for row in rows if math.isclose(float(row["epsilon"]), epsilon)]
        tag = f"eps{epsilon:.0e}"
        result[f"{tag}_gain_mean"] = statistics.mean(subset)
        result[f"{tag}_gain_geometric_mean"] = math.exp(statistics.mean(math.log(max(value, 1e-30)) for value in subset))
        result[f"{tag}_contraction_fraction"] = sum(value < 1 for value in subset) / len(subset)
    arithmetic = result["response_gain_mean"]
    geometric = result["response_gain_geometric_mean"]
    # The requested headline statistic is the arithmetic mean gain after 256
    # steps.  Keep the geometric mean as a multiplicative/robust companion,
    # but do not let it silently determine the headline label.
    result["perturbation_response_label"] = "contracting" if arithmetic < 1.0 else "amplifying" if arithmetic > 1.0 else "neutral"
    result["response_gain_geometric_label"] = "contracting" if geometric < 1.0 else "amplifying" if geometric > 1.0 else "neutral"
    return result


def run_checkpoint(selected: dict, device: torch.device) -> None:
    checkpoint = selected["checkpoint"]
    summary_path = EXP / "processed/dense_extension_summary_parts" / f"{checkpoint}.csv"
    perturb_path = EXP / "processed/dense_extension_perturbation_parts" / f"{checkpoint}.csv"
    orbit_path = EXP / "raw/dense_extension_orbits" / f"{checkpoint}.pt"
    if summary_path.exists() and perturb_path.exists() and len(read(perturb_path)) == DIRECTIONS * len(EPSILONS):
        print(json.dumps({"checkpoint": checkpoint, "status": "skip_complete"}), flush=True)
        return
    started = time.time()
    model = base.base.load_model(checkpoint, base.DEFAULT_CACHE, device)
    model.set_attn_implementation("eager")
    states = load_representative_states(model, selected, device)
    period = int(selected["candidate_period"])
    tail = states[512:]
    phase = torch.arange(len(tail), device=device) % period
    initial_orbit = torch.stack([tail[phase == index].mean(0) for index in range(period)])
    orbit_scale = sampled_orbit_scale(tail.detach().cpu(), seed=1919 + int(selected["training_step"]))["normalization_scale"]
    shooting = multiple_shooting(
        lambda value: model_step(model, value), initial_orbit,
        orbit_scale=orbit_scale, max_iterations=200, tolerance=1e-9,
    )
    floquet = {
        "leading_multiplier_modulus": "",
        "stability": "shooting_failed",
        "krylov_dimensions": "",
        "spectral_radius_estimates": "",
        "relative_disagreement": "",
    }
    if shooting.converged and shooting.minimal_period > 1:
        floquet = floquet_summary(
            lambda value: model_step(model, value), shooting.orbit[:shooting.minimal_period],
            seed=1905 + int(selected["training_step"]), dimensions=(16, 32),
        )
    perturb_rows = perturb_256(model, states[768], selected)
    perturb_summary = summarize_perturbation(perturb_rows)
    summary = {
        **selected,
        "initial_orbit_period": period,
        "minimal_period": shooting.minimal_period,
        "orbit_scale": orbit_scale,
        "shooting_loss": shooting.loss,
        "shooting_normalized_residual_p95": shooting.normalized_residual_p95,
        "shooting_converged": shooting.converged,
        "shooting_iterations": shooting.iterations,
        "krylov_dimensions": json.dumps(floquet.get("krylov_dimensions", "")),
        "spectral_radius_estimates": json.dumps(floquet.get("spectral_radius_estimates", "")),
        "relative_disagreement": floquet.get("relative_disagreement", ""),
        "leading_multiplier_modulus": floquet.get("leading_multiplier_modulus", ""),
        "stability": floquet.get("stability", "shooting_failed"),
        **perturb_summary,
        "runtime_seconds": time.time() - started,
    }
    write(summary_path, [summary])
    write(perturb_path, perturb_rows)
    orbit_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"selection": selected, "orbit": shooting.orbit.cpu(), "summary": summary}, orbit_path)
    print(json.dumps({"checkpoint": checkpoint, "period": shooting.minimal_period, "residual": shooting.normalized_residual_p95, "stability": summary["stability"], "gain_geomean": summary["response_gain_geometric_mean"], "seconds": round(summary["runtime_seconds"], 1)}), flush=True)
    del model, states, tail, initial_orbit
    torch.cuda.empty_cache()


def finalize() -> None:
    selected = selection()
    summaries = []
    perturbations = []
    for row in selected:
        summary_path = EXP / "processed/dense_extension_summary_parts" / f"{row['checkpoint']}.csv"
        perturb_path = EXP / "processed/dense_extension_perturbation_parts" / f"{row['checkpoint']}.csv"
        summaries.extend(read(summary_path) if summary_path.exists() else [])
        perturbations.extend(read(perturb_path) if perturb_path.exists() else [])
    if len(summaries) != len(selected) or len(perturbations) != len(selected) * DIRECTIONS * len(EPSILONS):
        raise RuntimeError(f"incomplete extension: summaries={len(summaries)}/{len(selected)}, perturbations={len(perturbations)}/{len(selected)*24}")
    for row in summaries:
        arithmetic = float(row["response_gain_mean"])
        geometric = float(row["response_gain_geometric_mean"])
        row["perturbation_response_label"] = "contracting" if arithmetic < 1.0 else "amplifying" if arithmetic > 1.0 else "neutral"
        row["response_gain_geometric_label"] = "contracting" if geometric < 1.0 else "amplifying" if geometric > 1.0 else "neutral"
        write(EXP / "processed/dense_extension_summary_parts" / f"{row['checkpoint']}.csv", [row])
    summaries.sort(key=lambda row: int(row["training_step"]))
    perturbations.sort(key=lambda row: (int(row["training_step"]), float(row["epsilon"]), int(row["direction_id"])))
    write(EXP / "processed/dense_periodic_checkpoint_summary.csv", summaries)
    write(EXP / "processed/dense_periodic_checkpoint_perturbation_256.csv", perturbations)
    write(EXP / "processed/dense_periodic_checkpoint_selection.csv", selected)
    print(json.dumps({"status": "complete", "checkpoints": len(summaries), "perturbations": len(perturbations)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--list-selection", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    selected = selection()
    if args.list_selection:
        print(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    if args.finalize:
        finalize()
        return
    requested = {item.strip() for item in args.checkpoints.split(",") if item.strip()}
    for row in selected:
        if not requested or row["checkpoint"] in requested:
            run_checkpoint(row, torch.device(args.device))


if __name__ == "__main__":
    main()
