from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch


TensorMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class TrajectoryResult:
    eval_states: list[torch.Tensor]
    state_norms: list[float]
    step_deltas: list[float]
    nearby_distances: list[float]
    initial_perturbation_distance: float
    diverged: bool
    collapsed: bool
    phase_label: str


@dataclass
class FrobeniusResult:
    local_normalized_frobenius: list[float]
    geometric_mean_normalized_frobenius: float
    arithmetic_mean_normalized_frobenius: float
    active_dim: int
    probes: int


def trajectory_summary_rows(result: TrajectoryResult) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for idx, (norm, delta, distance) in enumerate(
        zip(result.state_norms, result.step_deltas, result.nearby_distances)
    ):
        rows.append(
            {
                "step_index": idx,
                "state_norm": float(norm),
                "step_delta": float(delta),
                "relative_step_delta": float(delta / max(norm, 1e-12)),
                "nearby_distance": float(distance),
            }
        )
    return rows


def make_projection_vector(
    reference: torch.Tensor,
    mask: Optional[torch.Tensor],
    mode: str,
    seed: int,
    token_index: int = 0,
    hidden_index: int = 0,
) -> Optional[torch.Tensor]:
    if mode in {"none", "state_norm"}:
        return None
    vector = torch.zeros_like(reference, dtype=torch.float32)
    expanded_mask = expand_mask(mask, reference)
    if mode == "fixed_random":
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(int(seed))
        vector = torch.randn(reference.shape, device=reference.device, dtype=torch.float32, generator=generator)
        if expanded_mask is not None:
            vector = vector * expanded_mask
        return vector / (vector.norm() + 1e-12)
    if mode == "fixed_coordinate":
        token = max(0, min(int(token_index), reference.shape[-2] - 1))
        hidden = max(0, min(int(hidden_index), reference.shape[-1] - 1))
        index = [0] * reference.ndim
        index[-2] = token
        index[-1] = hidden
        vector[tuple(index)] = 1.0
        if expanded_mask is not None:
            vector = vector * expanded_mask
        return vector / (vector.norm() + 1e-12)
    raise ValueError(f"Unsupported trajectory projection mode: {mode}")


def make_projection_bank(
    reference: torch.Tensor,
    mask: Optional[torch.Tensor],
    mode: str,
    count: int,
    seed: int,
    sample_index: int = 0,
    shared_across_samples: bool = True,
) -> list[Optional[torch.Tensor]]:
    """Create deterministic projection vectors shared across checkpoints."""
    base = int(seed) if shared_across_samples else int(seed) + int(sample_index) * 100_000
    return [
        make_projection_vector(reference, mask, mode, seed=base + idx)
        for idx in range(max(0, int(count)))
    ]


def project_state(
    state: torch.Tensor,
    mode: str,
    projection_vector: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> float:
    if mode == "none":
        return 0.0
    if mode == "state_norm":
        return float(masked_norm(state, mask).detach().cpu())
    if projection_vector is None:
        raise ValueError(f"Projection mode {mode} requires a projection vector")
    expanded_mask = expand_mask(mask, state)
    x = state.float()
    if expanded_mask is not None:
        x = x * expanded_mask
    return float(torch.sum(x * projection_vector.to(device=state.device, dtype=torch.float32)).detach().cpu())


def expand_mask(mask: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    out = mask.detach().to(device=reference.device, dtype=reference.dtype)
    while out.ndim < reference.ndim:
        out = out.unsqueeze(-1)
    return out.expand_as(reference)


def active_dimension(reference: torch.Tensor, mask: Optional[torch.Tensor]) -> int:
    expanded = expand_mask(mask, reference)
    if expanded is None:
        return int(reference.numel())
    return max(1, int(expanded.sum().item()))


def masked_norm(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    expanded = expand_mask(mask, x)
    if expanded is not None:
        x = x * expanded
    return x.float().norm()


def apply_operator_update(
    current: torch.Tensor,
    raw_next: torch.Tensor,
    mode: str = "direct",
    residual_alpha: float = 1.0,
    output_scale: float = 1.0,
    norm_eps: float = 1e-12,
) -> torch.Tensor:
    raw_next = float(output_scale) * raw_next
    if mode == "direct":
        return raw_next
    if mode == "residual":
        return (1.0 - residual_alpha) * current + residual_alpha * raw_next
    if mode == "norm_matched":
        target_norm = current.float().norm().detach()
        next_norm = raw_next.float().norm().detach()
        return raw_next * (target_norm / (next_norm + norm_eps))
    if mode == "residual_norm_matched":
        mixed = (1.0 - residual_alpha) * current + residual_alpha * raw_next
        target_norm = current.float().norm().detach()
        next_norm = mixed.float().norm().detach()
        return mixed * (target_norm / (next_norm + norm_eps))
    raise ValueError(f"Unsupported operator update mode: {mode}")


def classify_phase(
    state_norms: list[float],
    step_deltas: list[float],
    nearby_distances: list[float],
    diverged: bool,
    collapsed: bool,
    convergence_tol: float,
    sensitivity_growth: float,
) -> str:
    if diverged:
        return "divergent"
    if collapsed:
        return "collapsed"
    if not step_deltas:
        return "unknown"
    tail_count = max(1, min(5, len(step_deltas)))
    tail_delta = step_deltas[-tail_count:]
    tail_norm = state_norms[-tail_count:]
    tail_relative_delta = sum(
        delta / max(norm, 1e-12) for delta, norm in zip(tail_delta, tail_norm)
    ) / tail_count
    if tail_relative_delta < convergence_tol:
        return "stable_fixed_like"
    if len(nearby_distances) >= 2:
        first = max(nearby_distances[0], 1e-12)
        last = nearby_distances[-1]
        if last / first > sensitivity_growth:
            return "sensitive_or_chaotic_like"
    if state_norms:
        return "bounded_nonfixed_like"
    return "unknown"


def run_feedback_trajectory(
    operator: TensorMap,
    x0: torch.Tensor,
    burn_in_steps: int,
    eval_steps: int,
    mask: Optional[torch.Tensor] = None,
    perturbation_epsilon: float = 1e-5,
    divergence_threshold: float = 1e6,
    collapse_threshold: float = 1e-8,
    convergence_tol: float = 1e-6,
    sensitivity_growth: float = 10.0,
) -> TrajectoryResult:
    expanded_mask = expand_mask(mask, x0)
    x = x0.detach().float()
    if expanded_mask is not None:
        x = x * expanded_mask

    noise = torch.randn_like(x)
    if expanded_mask is not None:
        noise = noise * expanded_mask
    noise = noise / (noise.norm() + 1e-12)
    x_near = x + perturbation_epsilon * noise
    initial_perturbation_distance = float(masked_norm(x_near - x, mask).cpu())

    eval_states: list[torch.Tensor] = []
    state_norms: list[float] = []
    step_deltas: list[float] = []
    nearby_distances: list[float] = []
    diverged = False
    collapsed = False

    total_steps = int(burn_in_steps) + int(eval_steps)
    with torch.no_grad():
        for step in range(total_steps):
            previous = x
            x = operator(x).detach().float()
            x_near = operator(x_near).detach().float()
            if expanded_mask is not None:
                x = x * expanded_mask
                x_near = x_near * expanded_mask

            norm = float(masked_norm(x, expanded_mask).cpu())
            delta = float(masked_norm(x - previous, expanded_mask).cpu())
            distance = float(masked_norm(x - x_near, expanded_mask).cpu())

            if not math.isfinite(norm) or norm > divergence_threshold:
                diverged = True
            if norm < collapse_threshold:
                collapsed = True

            if step >= burn_in_steps:
                state_norms.append(norm)
                step_deltas.append(delta)
                nearby_distances.append(distance)
                eval_states.append(x.detach().clone())

            if diverged or collapsed:
                break

    phase_label = classify_phase(
        state_norms,
        step_deltas,
        nearby_distances,
        diverged,
        collapsed,
        convergence_tol,
        sensitivity_growth,
    )
    return TrajectoryResult(
        eval_states=eval_states,
        state_norms=state_norms,
        step_deltas=step_deltas,
        nearby_distances=nearby_distances,
        initial_perturbation_distance=initial_perturbation_distance,
        diverged=diverged,
        collapsed=collapsed,
        phase_label=phase_label,
    )


def _sample_probe(reference: torch.Tensor, distribution: str) -> torch.Tensor:
    if distribution == "rademacher":
        return torch.empty_like(reference).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    if distribution == "gaussian":
        return torch.randn_like(reference)
    raise ValueError(f"Unsupported probe distribution: {distribution}")


def estimate_normalized_frobenius(
    operator: TensorMap,
    states: list[torch.Tensor],
    probes: int,
    mask: Optional[torch.Tensor] = None,
    probe_distribution: str = "rademacher",
) -> FrobeniusResult:
    if not states:
        return FrobeniusResult([], 0.0, 0.0, 0, int(probes))

    local_values: list[float] = []
    active_dim = active_dimension(states[0], mask)
    for state in states:
        expanded_mask = expand_mask(mask, state)
        probe_norm_sq: list[torch.Tensor] = []
        for _ in range(int(probes)):
            v = _sample_probe(state, probe_distribution)
            if expanded_mask is not None:
                v = v * expanded_mask
            x = state.detach().requires_grad_(True)
            _, jv = torch.autograd.functional.jvp(
                operator,
                (x,),
                (v,),
                create_graph=False,
                strict=False,
            )
            if expanded_mask is not None:
                jv = jv * expanded_mask
            probe_norm_sq.append(jv.float().pow(2).sum().detach())
        mean_norm_sq = torch.stack(probe_norm_sq).mean()
        normalized = torch.sqrt(mean_norm_sq / max(active_dim, 1))
        local_values.append(float(normalized.cpu()))

    positive = [max(x, 1e-12) for x in local_values]
    geometric = math.exp(sum(math.log(x) for x in positive) / len(positive))
    arithmetic = sum(local_values) / len(local_values)
    return FrobeniusResult(
        local_normalized_frobenius=local_values,
        geometric_mean_normalized_frobenius=float(geometric),
        arithmetic_mean_normalized_frobenius=float(arithmetic),
        active_dim=active_dim,
        probes=int(probes),
    )


def lagged_state_distance_metrics(
    states: list[torch.Tensor],
    windows: list[int],
    mask: Optional[torch.Tensor] = None,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for window in windows:
        k = int(window)
        if k <= 0 or len(states) <= k:
            continue
        distances = []
        for idx in range(len(states) - k):
            distances.append(masked_norm(states[idx + k] - states[idx], mask).detach().float())
        if not distances:
            continue
        d = torch.stack(distances)
        rows.append(
            {
                "lag_window": k,
                "lag_distance_mean": float(d.mean().cpu()),
                "lag_distance_min": float(d.min().cpu()),
                "lag_distance_std": float(d.std(unbiased=False).cpu()),
            }
        )
    return rows


def nearby_growth_metrics(nearby_distances: list[float], steps: int) -> dict[str, float]:
    if len(nearby_distances) < 2:
        return {"nearby_growth_ratio": 0.0, "nearby_log_growth_per_step": 0.0}
    first = max(float(nearby_distances[0]), 1e-12)
    last = max(float(nearby_distances[-1]), 1e-12)
    ratio = last / first
    denom = max(1, int(steps))
    return {
        "nearby_growth_ratio": float(ratio),
        "nearby_log_growth_per_step": float(math.log(ratio) / denom),
    }


def _jvp_at_state(
    operator: TensorMap,
    state: torch.Tensor,
    tangent: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    expanded_mask = expand_mask(mask, state)
    v = tangent
    if expanded_mask is not None:
        v = v * expanded_mask
    x = state.detach().requires_grad_(True)
    _, jv = torch.autograd.functional.jvp(
        operator,
        (x,),
        (v,),
        create_graph=False,
        strict=False,
    )
    if expanded_mask is not None:
        jv = jv * expanded_mask
    return jv.detach().float()


def multi_step_jacobian_product_metrics(
    operator: TensorMap,
    states: list[torch.Tensor],
    windows: list[int],
    probes: int,
    mask: Optional[torch.Tensor] = None,
    probe_distribution: str = "rademacher",
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    if not states:
        return rows
    for window in windows:
        k = int(window)
        if k <= 0 or len(states) < k:
            continue
        gains: list[float] = []
        start_count = len(states) - k + 1
        for start in range(start_count):
            for _ in range(int(probes)):
                v0 = _sample_probe(states[start], probe_distribution)
                expanded_mask = expand_mask(mask, states[start])
                if expanded_mask is not None:
                    v0 = v0 * expanded_mask
                v0_norm = float(v0.float().norm().cpu())
                v = v0
                for offset in range(k):
                    v = _jvp_at_state(operator, states[start + offset], v, mask)
                gain = float(v.float().norm().cpu()) / max(v0_norm, 1e-12)
                if math.isfinite(gain):
                    gains.append(gain)
        if not gains:
            continue
        gain_tensor = torch.tensor(gains, dtype=torch.float64)
        gain_mean = float(gain_tensor.mean())
        gain_max = float(gain_tensor.max())
        rows.append(
            {
                "product_window": k,
                "product_probes": int(probes),
                "product_start_count": int(start_count),
                "product_gain_mean": gain_mean,
                "product_gain_max": gain_max,
                "product_log_gain_mean": float(math.log(max(gain_mean, 1e-12)) / k),
                "product_log_gain_max": float(math.log(max(gain_max, 1e-12)) / k),
            }
        )
    return rows


def maximal_lyapunov_metrics(
    operator: TensorMap,
    states: list[torch.Tensor],
    probes: int,
    mask: Optional[torch.Tensor] = None,
    probe_distribution: str = "rademacher",
) -> dict[str, float | int | list[float]]:
    """Estimate the leading finite-time Lyapunov exponent with JVP renormalization."""
    if not states or probes <= 0:
        return {"lyapunov_steps": 0, "lyapunov_probes": int(probes), "lyapunov_exponents": []}
    exponents: list[float] = []
    for _ in range(int(probes)):
        v = _sample_probe(states[0], probe_distribution)
        expanded_mask = expand_mask(mask, states[0])
        if expanded_mask is not None:
            v = v * expanded_mask
        v = v / (v.float().norm() + 1e-12)
        log_growth = 0.0
        valid_steps = 0
        for state in states:
            v = _jvp_at_state(operator, state, v, mask)
            norm = float(v.float().norm().cpu())
            if not math.isfinite(norm) or norm <= 0.0:
                log_growth = float("-inf")
                valid_steps += 1
                break
            log_growth += math.log(norm)
            valid_steps += 1
            v = v / norm
        if valid_steps:
            exponents.append(log_growth / valid_steps)
    finite = [x for x in exponents if math.isfinite(x)]
    mean = sum(finite) / len(finite) if finite else float("-inf")
    std = math.sqrt(sum((x - mean) ** 2 for x in finite) / len(finite)) if finite else 0.0
    return {
        "lyapunov_steps": len(states),
        "lyapunov_probes": int(probes),
        "lyapunov_exponents": exponents,
        "maximal_lyapunov_mean": mean,
        "maximal_lyapunov_std": std,
        "maximal_lyapunov_max": max(finite) if finite else float("-inf"),
    }
