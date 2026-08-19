from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import torch


def _unit_random(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    value = torch.randn(reference.shape, device=reference.device, dtype=torch.float32, generator=generator)
    return value / value.norm().clamp_min(1e-12)


class SingleTokenOperator:
    """Hidden-state feedback operators used by the three single-token groups.

    G1: x -> transformer(x)[-1]
    G2: x -> transformer(concat(frozen_prefix, x))[-1]
    G3: X -> transformer(X), while target_map differentiates only the final
        output token with respect to the final input token and freezes the
        current prefix for that local derivative.
    """

    VALID_GROUPS = {"isolated_token", "frozen_context", "dynamic_context"}

    def __init__(self, model: torch.nn.Module, group: str, prefix: torch.Tensor | None = None):
        if group not in self.VALID_GROUPS:
            raise ValueError(f"unknown group: {group}")
        if group != "isolated_token" and prefix is None:
            raise ValueError(f"{group} requires a prefix")
        self.model = model
        self.group = group
        self.prefix = None if prefix is None else prefix.detach().float()
        self.model_dtype = next(model.parameters()).dtype

    def _forward_hidden(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2:
            raise ValueError(f"expected [L,H], got {tuple(sequence.shape)}")
        length = sequence.shape[0]
        attention_mask = torch.ones((1, length), device=sequence.device, dtype=torch.long)
        position_ids = torch.arange(length, device=sequence.device, dtype=torch.long).unsqueeze(0)
        inputs = sequence.to(dtype=self.model_dtype).unsqueeze(0)
        if hasattr(self.model, "gpt_neox"):
            output = self.model.gpt_neox(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        else:
            output = self.model(
                inputs_embeds=inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            ).hidden_states[-1]
        return output[0].float()

    def initial_state(self, target: torch.Tensor) -> torch.Tensor:
        target = target.detach().float().reshape(-1)
        if self.group == "dynamic_context":
            return torch.cat([self.prefix, target.unsqueeze(0)], dim=0)
        return target

    def full_step(self, state: torch.Tensor) -> torch.Tensor:
        if self.group == "isolated_token":
            return self._forward_hidden(state.reshape(1, -1))[-1]
        if self.group == "frozen_context":
            sequence = torch.cat([self.prefix, state.reshape(1, -1)], dim=0)
            return self._forward_hidden(sequence)[-1]
        return self._forward_hidden(state)

    def target(self, state: torch.Tensor) -> torch.Tensor:
        return state[-1] if self.group == "dynamic_context" else state

    def current_prefix(self, state: torch.Tensor) -> torch.Tensor | None:
        if self.group == "isolated_token":
            return None
        if self.group == "frozen_context":
            return self.prefix
        return state[:-1]

    def target_map(self, state: torch.Tensor):
        """Return an H->H function at the current full state.

        For G3 the prefix is intentionally captured as a constant. Thus its
        Jacobian is the registered target-to-target block, not the full LH map.
        """
        prefix = self.current_prefix(state)

        def mapping(target: torch.Tensor) -> torch.Tensor:
            if prefix is None:
                sequence = target.reshape(1, -1)
            else:
                sequence = torch.cat([prefix, target.reshape(1, -1)], dim=0)
            return self._forward_hidden(sequence)[-1]

        return mapping


@dataclass
class TrajectoryResult:
    states: list[torch.Tensor]
    rows: list[dict[str, float | int]]
    initial_nearby_distance: float


def fixed_projection_vectors(reference: torch.Tensor, count: int = 4, seed: int = 1234) -> list[torch.Tensor]:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed))
    vectors: list[torch.Tensor] = []
    for _ in range(int(count)):
        q = torch.randn(reference.shape, device=reference.device, dtype=torch.float32, generator=generator)
        vectors.append(q / q.norm().clamp_min(1e-12))
    return vectors


def run_trajectory(
    operator: SingleTokenOperator,
    initial_state: torch.Tensor,
    steps: int,
    epsilon: float,
    perturbation_seed: int,
    projections: list[torch.Tensor],
) -> TrajectoryResult:
    state = initial_state.detach().float()
    target0 = operator.target(state)
    direction = _unit_random(target0, perturbation_seed)
    nearby = state.clone()
    if operator.group == "dynamic_context":
        nearby[-1] = nearby[-1] + float(epsilon) * direction
    else:
        nearby = nearby + float(epsilon) * direction
    initial_distance = float((operator.target(nearby) - target0).norm().cpu())
    rows: list[dict[str, float | int]] = []
    states: list[torch.Tensor] = [state.detach().clone()]

    with torch.no_grad():
        for step in range(int(steps)):
            previous = state
            state = operator.full_step(state).float()
            nearby = operator.full_step(nearby).float()
            target = operator.target(state)
            previous_target = operator.target(previous)
            delta = float((target - previous_target).norm().cpu())
            norm = float(target.norm().cpu())
            distance = float((operator.target(nearby) - target).norm().cpu())
            row: dict[str, float | int] = {
                "step": step + 1,
                "state_norm": norm,
                "step_delta": delta,
                "relative_step_delta": delta / max(norm, 1e-12),
                "nearby_distance": distance,
            }
            if operator.group == "dynamic_context":
                prefix = state[:-1]
                old_prefix = previous[:-1]
                prefix_norm = float(prefix.norm().cpu())
                prefix_delta = float((prefix - old_prefix).norm().cpu())
                row["prefix_norm"] = prefix_norm
                row["prefix_relative_delta"] = prefix_delta / max(prefix_norm, 1e-12)
            for index, vector in enumerate(projections):
                row[f"projection_{index}"] = float(torch.dot(target.float(), vector).cpu())
            rows.append(row)
            states.append(state.detach().clone())
    return TrajectoryResult(states=states, rows=rows, initial_nearby_distance=initial_distance)


def target_jvp(operator: SingleTokenOperator, state: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    mapping = operator.target_map(state)
    target = operator.target(state).detach().requires_grad_(True)
    _, jv = torch.autograd.functional.jvp(mapping, (target,), (vector,), create_graph=False, strict=False)
    return jv.detach().float()


def estimate_hutchinson_frobenius(
    operator: SingleTokenOperator,
    states: Iterable[torch.Tensor],
    probes: int,
    seed: int,
) -> list[float]:
    states = list(states)
    if not states or probes <= 0:
        return []
    generator = torch.Generator(device=states[0].device)
    generator.manual_seed(int(seed))
    hidden = int(operator.target(states[0]).numel())
    estimates: list[float] = []
    for state in states:
        norms: list[torch.Tensor] = []
        for _ in range(int(probes)):
            v = torch.randint(0, 2, (hidden,), device=state.device, generator=generator).float().mul_(2).sub_(1)
            norms.append(target_jvp(operator, state, v).pow(2).sum())
        estimates.append(math.sqrt(float(torch.stack(norms).mean().cpu()) / hidden))
    return estimates


def estimate_conditional_lyapunov(
    operator: SingleTokenOperator,
    states: Iterable[torch.Tensor],
    probes: int,
    seed: int,
) -> list[float]:
    states = list(states)
    if not states or probes <= 0:
        return []
    exponents: list[float] = []
    for probe in range(int(probes)):
        v = _unit_random(operator.target(states[0]), int(seed) + probe)
        total = 0.0
        valid = 0
        for state in states:
            jv = target_jvp(operator, state, v)
            growth = float(jv.norm().cpu())
            if not math.isfinite(growth) or growth <= 0:
                total = float("-inf")
                valid += 1
                break
            total += math.log(growth)
            valid += 1
            v = jv / growth
        exponents.append(total / max(valid, 1))
    return exponents


def exact_target_jacobian(operator: SingleTokenOperator, state: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
    target = operator.target(state).detach().requires_grad_(True)
    mapping = operator.target_map(state)
    if hasattr(torch, "func") and hasattr(torch.func, "jacrev"):
        jacobian = torch.func.jacrev(mapping, chunk_size=int(chunk_size))(target)
    else:
        jacobian = torch.autograd.functional.jacobian(mapping, target, vectorize=True)
    if jacobian.shape != (target.numel(), target.numel()):
        raise RuntimeError(f"target Jacobian must be square HxH; got {tuple(jacobian.shape)}")
    return jacobian.detach().float()


def jacobian_summary(jacobian: torch.Tensor) -> dict[str, float | int | list[int] | str]:
    hidden = int(jacobian.shape[0])
    singular = torch.linalg.svdvals(jacobian.float())
    frobenius = float(jacobian.norm().cpu()) / math.sqrt(hidden)
    rms = float(torch.sqrt(torch.mean(singular.square())).cpu())
    checksum = hashlib.sha256(jacobian.detach().cpu().numpy().tobytes()).hexdigest()
    stable_rank = float(jacobian.square().sum().cpu()) / max(float(singular[0].square().cpu()), 1e-30)
    return {
        "shape": list(jacobian.shape),
        "normalized_frobenius": frobenius,
        "sigma_max": float(singular[0].cpu()),
        "sigma_mean": float(singular.mean().cpu()),
        "sigma_median": float(singular.median().cpu()),
        "sigma_rms": rms,
        "sigma_max_over_rms": float(singular[0].cpu()) / max(rms, 1e-30),
        "stable_rank": stable_rank,
        "checksum_sha256": checksum,
    }


def classify_convergence(
    tail_relative_step_delta: float,
    nearby_log_growth: float | None,
    lyapunov_exponents: list[float],
    nearby_numerical_floor: bool = False,
) -> str:
    if nearby_numerical_floor or nearby_log_growth is None or len(lyapunov_exponents) < 2:
        return "unresolved"
    all_negative = all(value < 0 for value in lyapunov_exponents)
    all_positive = all(value > 0 for value in lyapunov_exponents)
    if tail_relative_step_delta < 1e-6 and nearby_log_growth < 0 and all_negative:
        return "stable_fixed_point_candidate"
    if tail_relative_step_delta >= 1e-6 and nearby_log_growth < 0 and all_negative:
        return "stable_nonfixed_candidate"
    if tail_relative_step_delta >= 1e-6 and nearby_log_growth > 0 and all_positive:
        return "expanding_or_chaotic_candidate"
    return "unresolved"


def causal_cross_gradient_max(operator: SingleTokenOperator, state: torch.Tensor) -> float:
    if operator.group != "dynamic_context" or state.shape[0] <= 1:
        return 0.0
    prefix = state[:-1].detach()
    target = state[-1].detach().requires_grad_(True)

    def prefix_output(value: torch.Tensor) -> torch.Tensor:
        sequence = torch.cat([prefix, value.reshape(1, -1)], dim=0)
        return operator._forward_hidden(sequence)[:-1]

    direction = _unit_random(target, 9981)
    _, jv = torch.autograd.functional.jvp(prefix_output, (target,), (direction,), strict=False)
    return float(jv.abs().max().cpu())


def projected_poincare_points(rows: list[dict], projection_count: int = 4) -> list[dict[str, float]]:
    """Upward crossings of the per-trajectory median z0 section.

    Coordinates are linearly interpolated between adjacent discrete states.
    This is a projected diagnostic; it is deliberately not a convergence test.
    """
    if len(rows) < 2:
        return []
    z0 = torch.tensor([float(row["projection_0"]) for row in rows], dtype=torch.float64)
    section = float(z0.median())
    output: list[dict[str, float]] = []
    for index in range(len(rows) - 1):
        left = float(rows[index]["projection_0"])
        right = float(rows[index + 1]["projection_0"])
        if not (left < section <= right) or right == left:
            continue
        fraction = (section - left) / (right - left)
        point = {"section_z0": section, "crossing_step": float(rows[index]["step"]) + fraction}
        for projection in range(1, int(projection_count)):
            key = f"projection_{projection}"
            a = float(rows[index][key])
            b = float(rows[index + 1][key])
            point[f"z{projection}"] = a + fraction * (b - a)
        output.append(point)
    return output
