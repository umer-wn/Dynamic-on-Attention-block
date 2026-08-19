#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._bootstrap import require_packages

require_packages(["torch", "transformers", "datasets", "tqdm", "yaml"])

import torch
from tqdm import tqdm

from src.data_utils import load_text_samples, tokenize_samples
from src.dynamics import (
    apply_operator_update,
    estimate_normalized_frobenius,
    expand_mask,
    lagged_state_distance_metrics,
    make_projection_bank,
    make_projection_vector,
    maximal_lyapunov_metrics,
    multi_step_jacobian_product_metrics,
    nearby_growth_metrics,
    project_state,
    run_feedback_trajectory,
    trajectory_summary_rows,
)
from src.io_utils import base_metadata, load_config, sanitize_name, setup_storage_env, should_skip, write_jsonl
from src.model_utils import iter_model_revisions, load_model_and_tokenizer


def _target_from_outputs(
    outputs: Any,
    target: str,
    embedding: torch.nn.Module | None = None,
    logit_temperature: float = 1.0,
) -> torch.Tensor:
    if target == "final_hidden":
        return outputs.hidden_states[-1].float()
    if target == "embedding_expectation":
        if embedding is None or not hasattr(embedding, "weight"):
            raise ValueError("embedding_expectation target requires an embedding matrix")
        temperature = max(float(logit_temperature), 1e-6)
        probs = torch.softmax(outputs.logits.float() / temperature, dim=-1)
        return probs @ embedding.weight.float()
    raise ValueError(f"Unsupported dynamics target: {target}")


def _make_llm_operator(
    model: torch.nn.Module,
    attention_mask: torch.Tensor | None,
    original_dtype: torch.dtype,
    target: str,
    update_mode: str,
    residual_alpha: float,
    mask: torch.Tensor | None,
    logit_temperature: float,
    output_scale: float,
):
    def operator(x: torch.Tensor) -> torch.Tensor:
        x_in = x
        expanded_mask = expand_mask(mask, x_in)
        if expanded_mask is not None:
            x_in = x_in * expanded_mask
        outputs = model(
            inputs_embeds=x_in.to(dtype=original_dtype),
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        raw_next = _target_from_outputs(outputs, target, embedding=model.get_input_embeddings(), logit_temperature=logit_temperature)
        next_x = apply_operator_update(
            x.float(),
            raw_next.float(),
            mode=update_mode,
            residual_alpha=residual_alpha,
            output_scale=output_scale,
        )
        if expanded_mask is not None:
            next_x = next_x * expanded_mask
        return next_x

    return operator


def _finite_list(values: list[float]) -> list[float]:
    return [float(x) if math.isfinite(float(x)) else 0.0 for x in values]


def _add_projection_values(
    rows: list[dict[str, Any]],
    states: list[torch.Tensor],
    projection_mode: str,
    projection_vector: torch.Tensor | None,
    mask: torch.Tensor | None,
) -> list[dict[str, Any]]:
    projected = [
        project_state(state, projection_mode, projection_vector=projection_vector, mask=mask)
        for state in states
    ]
    for idx, row in enumerate(rows):
        z_t = projected[idx] if idx < len(projected) else 0.0
        z_next = projected[idx + 1] if idx + 1 < len(projected) else None
        row["projection_mode"] = projection_mode
        row["projection_value"] = z_t
        row["projection_next"] = z_next
    return rows


def _add_projection_bank_values(
    rows: list[dict[str, Any]],
    states: list[torch.Tensor],
    projection_mode: str,
    projection_vectors: list[torch.Tensor | None],
    mask: torch.Tensor | None,
) -> list[dict[str, Any]]:
    for projection_index, vector in enumerate(projection_vectors):
        projected = [
            project_state(state, projection_mode, projection_vector=vector, mask=mask)
            for state in states
        ]
        for idx, row in enumerate(rows):
            value = projected[idx] if idx < len(projected) else 0.0
            next_value = projected[idx + 1] if idx + 1 < len(projected) else None
            row[f"projection_{projection_index}"] = value
            row[f"projection_{projection_index}_next"] = next_value
            if projection_index == 0:
                row["projection_mode"] = projection_mode
                row["projection_value"] = value
                row["projection_next"] = next_value
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_storage_env(config)
    torch.manual_seed(int(config.get("seed", 1234)))

    dynamics_cfg = config.get("dynamics", {})
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
            embedding = model.get_input_embeddings()
            target = str(dynamics_cfg.get("target", "final_hidden"))
            update_mode = str(dynamics_cfg.get("operator_update", "direct"))
            residual_alpha = float(dynamics_cfg.get("residual_alpha", 1.0))
            logit_temperature = float(dynamics_cfg.get("logit_temperature", 1.0))
            output_scale = float(dynamics_cfg.get("output_scale", 1.0))
            token_mode = str(dynamics_cfg.get("token_mode", config.get("jacobian_token_mode", "nonpad_flattened")))

            for seq_len in config["dataset"].get("sequence_lengths", [128]):
                out = raw_dir / (
                    f"{config['experiment_name']}__{sanitize_name(name)}__{revision}"
                    f"__seq{seq_len}__dynamical_edge.jsonl"
                )
                if should_skip(out, bool(config.get("skip_existing", True))):
                    print(f"skip existing {out}")
                    continue

                batches = tokenize_samples(
                    tokenizer,
                    texts,
                    int(seq_len),
                    int(config["dataset"].get("num_samples", len(texts))),
                )
                metadata = base_metadata(config, name, revision, tokenizer.name_or_path, int(seq_len))
                rows = []
                trajectory_rows = []
                distance_rows = []
                product_rows = []
                for batch_idx, batch in enumerate(tqdm(batches, desc=f"dynamics {name}@{revision} seq{seq_len}")):
                    sample_index = batch.pop("sample_index")
                    batch = {k: v.to(device) for k, v in batch.items()}
                    attention_mask = batch.get("attention_mask")
                    with torch.no_grad():
                        inputs_embeds = embedding(batch["input_ids"]).detach().float()
                    mask = attention_mask if token_mode == "nonpad_flattened" else None
                    expanded_mask = expand_mask(mask, inputs_embeds)
                    if expanded_mask is not None:
                        inputs_embeds = inputs_embeds * expanded_mask
                    projection_mode = str(dynamics_cfg.get("trajectory_projection", "state_norm"))
                    projection_seed = int(dynamics_cfg.get("projection_seed", config.get("seed", 1234))) + int(sample_index)
                    projection_vector = make_projection_vector(
                        inputs_embeds,
                        mask,
                        projection_mode,
                        seed=projection_seed,
                        token_index=int(dynamics_cfg.get("projection_token_index", 0)),
                        hidden_index=int(dynamics_cfg.get("projection_hidden_index", 0)),
                    )
                    projection_bank_cfg = dynamics_cfg.get("trajectory_projection_bank")
                    projection_bank: list[torch.Tensor | None] = []
                    if projection_bank_cfg:
                        projection_mode = str(projection_bank_cfg.get("mode", "fixed_random"))
                        projection_seed = int(projection_bank_cfg.get("seed", config.get("seed", 1234)))
                        projection_bank = make_projection_bank(
                            inputs_embeds,
                            mask,
                            projection_mode,
                            count=int(projection_bank_cfg.get("count", 3)),
                            seed=projection_seed,
                            sample_index=int(sample_index),
                            shared_across_samples=bool(projection_bank_cfg.get("shared_across_samples", True)),
                        )

                    operator = _make_llm_operator(
                        model,
                        attention_mask,
                        inputs_embeds.dtype,
                        target,
                        update_mode,
                        residual_alpha,
                        mask,
                        logit_temperature,
                        output_scale,
                    )
                    trajectory = run_feedback_trajectory(
                        operator,
                        inputs_embeds,
                        burn_in_steps=int(dynamics_cfg.get("burn_in_steps", 8)),
                        eval_steps=int(dynamics_cfg.get("eval_steps", 4)),
                        mask=mask,
                        perturbation_epsilon=float(dynamics_cfg.get("perturbation_epsilon", 1e-5)),
                        divergence_threshold=float(dynamics_cfg.get("divergence_threshold", 1e6)),
                        collapse_threshold=float(dynamics_cfg.get("collapse_threshold", 1e-8)),
                        convergence_tol=float(dynamics_cfg.get("convergence_tol", 1e-6)),
                        sensitivity_growth=float(dynamics_cfg.get("sensitivity_growth", 10.0)),
                    )

                    eval_states = trajectory.eval_states
                    max_eval_states = int(dynamics_cfg.get("frobenius_eval_states", len(eval_states)))
                    eval_states = eval_states[-max_eval_states:] if max_eval_states > 0 else []
                    frob = estimate_normalized_frobenius(
                        operator,
                        eval_states,
                        probes=int(dynamics_cfg.get("frobenius_probes", 4)),
                        mask=mask,
                        probe_distribution=str(dynamics_cfg.get("probe_distribution", "rademacher")),
                    )
                    nearby_growth = nearby_growth_metrics(trajectory.nearby_distances, len(trajectory.nearby_distances))
                    lyapunov = maximal_lyapunov_metrics(
                        operator,
                        trajectory.eval_states,
                        probes=int(dynamics_cfg.get("lyapunov_probes", 0)),
                        mask=mask,
                        probe_distribution=str(dynamics_cfg.get("probe_distribution", "rademacher")),
                    )
                    lag_metrics = lagged_state_distance_metrics(
                        trajectory.eval_states,
                        [int(x) for x in dynamics_cfg.get("lag_distance_windows", [])],
                        mask=mask,
                    )
                    product_metrics = multi_step_jacobian_product_metrics(
                        operator,
                        trajectory.eval_states,
                        [int(x) for x in dynamics_cfg.get("product_jacobian_windows", [])],
                        probes=int(dynamics_cfg.get("product_jacobian_probes", 0)),
                        mask=mask,
                        probe_distribution=str(dynamics_cfg.get("probe_distribution", "rademacher")),
                    ) if int(dynamics_cfg.get("product_jacobian_probes", 0)) > 0 else []

                    row_base = {
                        **metadata,
                        "batch_index": batch_idx,
                        "sample_index": sample_index,
                        "method": "paper_style_feedback_dynamics",
                        "input_space": "inputs_embeds",
                        "target": target,
                        "operator_update": update_mode,
                        "residual_alpha": residual_alpha,
                        "logit_temperature": logit_temperature,
                        "output_scale": output_scale,
                        "token_mode": token_mode,
                        "input_shape": list(inputs_embeds.shape),
                        "active_dim": frob.active_dim,
                        "burn_in_steps": int(dynamics_cfg.get("burn_in_steps", 8)),
                        "eval_steps": int(dynamics_cfg.get("eval_steps", 4)),
                        "actual_eval_states": len(trajectory.eval_states),
                    }

                    if bool(dynamics_cfg.get("save_trajectory_summary", False)):
                        summary_rows = trajectory_summary_rows(trajectory)
                        if projection_bank:
                            projected_rows = _add_projection_bank_values(
                                summary_rows,
                                trajectory.eval_states,
                                projection_mode,
                                projection_bank,
                                mask,
                            )
                        else:
                            projected_rows = _add_projection_values(
                                summary_rows,
                                trajectory.eval_states,
                                projection_mode,
                                projection_vector,
                                mask,
                            )
                        for trajectory_row in projected_rows:
                            trajectory_rows.append({**row_base, **trajectory_row})
                    for lag_row in lag_metrics:
                        distance_rows.append({**row_base, **nearby_growth, **lag_row})
                    for product_row in product_metrics:
                        product_rows.append({**row_base, **product_row})

                    rows.append(
                        {
                            **row_base,
                            "frobenius_eval_states": len(eval_states),
                            "frobenius_probes": frob.probes,
                            "normalized_frobenius_geomean": frob.geometric_mean_normalized_frobenius,
                            "normalized_frobenius_mean": frob.arithmetic_mean_normalized_frobenius,
                            "normalized_frobenius_local": _finite_list(frob.local_normalized_frobenius),
                            "edge_distance_log": abs(math.log(max(frob.geometric_mean_normalized_frobenius, 1e-12))),
                            "phase_label": trajectory.phase_label,
                            "diverged": int(trajectory.diverged),
                            "collapsed": int(trajectory.collapsed),
                            "state_norms": _finite_list(trajectory.state_norms),
                            "step_deltas": _finite_list(trajectory.step_deltas),
                            "relative_step_deltas": _finite_list([
                                delta / max(norm, 1e-12)
                                for delta, norm in zip(trajectory.step_deltas, trajectory.state_norms)
                            ]),
                            "nearby_distances": _finite_list(trajectory.nearby_distances),
                            "initial_perturbation_distance": trajectory.initial_perturbation_distance,
                            "final_asymptotic_distance": (
                                trajectory.nearby_distances[-1] if trajectory.nearby_distances else 0.0
                            ),
                            "final_to_initial_separation": (
                                trajectory.nearby_distances[-1] / max(trajectory.initial_perturbation_distance, 1e-12)
                                if trajectory.nearby_distances else 0.0
                            ),
                            **lyapunov,
                            **nearby_growth,
                        }
                    )
                write_jsonl(out, rows)
                print(f"wrote {len(rows)} rows to {out}")
                if trajectory_rows:
                    trajectory_out = raw_dir / (
                        f"{config['experiment_name']}__{sanitize_name(name)}__{revision}"
                        f"__seq{seq_len}__dynamics_trajectory.jsonl"
                    )
                    write_jsonl(trajectory_out, trajectory_rows)
                    print(f"wrote {len(trajectory_rows)} rows to {trajectory_out}")
                if distance_rows:
                    distance_out = raw_dir / (
                        f"{config['experiment_name']}__{sanitize_name(name)}__{revision}"
                        f"__seq{seq_len}__state_distance_metrics.jsonl"
                    )
                    write_jsonl(distance_out, distance_rows)
                    print(f"wrote {len(distance_rows)} rows to {distance_out}")
                if product_rows:
                    product_out = raw_dir / (
                        f"{config['experiment_name']}__{sanitize_name(name)}__{revision}"
                        f"__seq{seq_len}__product_jacobian_metrics.jsonl"
                    )
                    write_jsonl(product_out, product_rows)
                    print(f"wrote {len(product_rows)} rows to {product_out}")


if __name__ == "__main__":
    main()
