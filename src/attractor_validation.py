from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch


TensorMap = Callable[[torch.Tensor], torch.Tensor]


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def divisors(value: int) -> list[int]:
    if value <= 0:
        raise ValueError("value must be positive")
    return [candidate for candidate in range(1, value + 1) if value % candidate == 0]


def sampled_orbit_scale(
    states: torch.Tensor,
    *,
    seed: int,
    pairs: int = 8192,
) -> dict[str, float]:
    """Robust full-state scale used to normalize recurrence errors."""
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("states must have shape [time, hidden] with time >= 2")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    count = min(int(pairs), max(len(states) * 4, 256))
    left = torch.randint(0, len(states), (count,), generator=generator)
    right = torch.randint(0, len(states), (count,), generator=generator)
    cpu = states.detach().float().cpu()
    distances = torch.linalg.vector_norm(cpu[left] - cpu[right], dim=-1)
    norms = torch.linalg.vector_norm(cpu, dim=-1)
    numerical_floor = 8.0 * torch.finfo(torch.float32).eps * max(float(norms.median()), 1.0)
    d95 = float(torch.quantile(distances, 0.95))
    return {
        "orbit_pairwise_d95": d95,
        "orbit_pairwise_rms": float(torch.sqrt(distances.square().mean())),
        "orbit_diameter_approx": float(distances.max()),
        "median_state_norm": float(norms.median()),
        "numerical_floor": numerical_floor,
        "normalization_scale": max(d95, numerical_floor, 1e-12),
    }


def lag_curve(
    states: torch.Tensor,
    *,
    max_lag: int,
    normalization_scale: float | None = None,
) -> list[dict[str, float | int]]:
    """Full-dimensional Euclidean and cosine recurrence for every lag."""
    if states.ndim != 2:
        raise ValueError("states must have shape [time, hidden]")
    if max_lag < 1 or len(states) <= max_lag:
        raise ValueError("trajectory must be longer than max_lag")
    scale = float(normalization_scale or sampled_orbit_scale(states, seed=1901)["normalization_scale"])
    rows: list[dict[str, float | int]] = []
    values = states.detach().float()
    normalized_states = torch.nn.functional.normalize(values, dim=-1)
    for lag in range(1, int(max_lag) + 1):
        difference = torch.linalg.vector_norm(values[lag:] - values[:-lag], dim=-1)
        cosine = 1.0 - (normalized_states[lag:] * normalized_states[:-lag]).sum(dim=-1)
        rows.append(
            {
                "candidate_lag": lag,
                "absolute_median": float(difference.median()),
                "absolute_p95": float(torch.quantile(difference, 0.95)),
                "normalized_median": float(difference.median()) / scale,
                "normalized_p95": float(torch.quantile(difference, 0.95)) / scale,
                "cosine_distance_median": float(cosine.median()),
                "cosine_distance_p95": float(torch.quantile(cosine, 0.95)),
            }
        )
    return rows


def best_two_lags(rows: Sequence[dict[str, float | int]], *, exclude_lag1: bool = True) -> tuple[dict, dict]:
    candidates = [row for row in rows if not exclude_lag1 or int(row["candidate_lag"]) > 1]
    if len(candidates) < 2:
        raise ValueError("at least two lag candidates are required")
    ordered = sorted(candidates, key=lambda row: float(row["normalized_p95"]))
    return ordered[0], ordered[1]


def phase_invariant_orbit_distance(points: torch.Tensor, orbit: torch.Tensor) -> torch.Tensor:
    """Distance from each point to the nearest phase of an orbit."""
    if points.ndim == 1:
        points = points.unsqueeze(0)
    if orbit.ndim != 2 or points.ndim != 2 or points.shape[-1] != orbit.shape[-1]:
        raise ValueError("points and orbit must be [N,H] and [P,H]")
    return torch.cdist(points.float(), orbit.float()).min(dim=1).values


def cyclic_orbit_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    """Mean point distance after the best cyclic alignment."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("orbits must have identical [period, hidden] shapes")
    return min(float(torch.linalg.vector_norm(left - right.roll(shift, 0), dim=1).mean()) for shift in range(len(left)))


def minimal_repeated_period(orbit: torch.Tensor, *, relative_tolerance: float = 1e-5) -> int:
    if orbit.ndim != 2 or len(orbit) < 1:
        raise ValueError("orbit must have shape [period, hidden]")
    scale = max(float(torch.linalg.vector_norm(orbit - orbit.mean(0), dim=1).median()), 1e-12)
    for candidate in divisors(len(orbit)):
        if candidate == len(orbit):
            return candidate
        error = torch.linalg.vector_norm(orbit - orbit.roll(-candidate, 0), dim=1)
        if float(torch.quantile(error, 0.95)) / scale <= relative_tolerance:
            return candidate
    return len(orbit)


def finite_time_lyapunov(
    mapping: TensorMap,
    states: torch.Tensor,
    *,
    seed: int,
    epsilon: float = 1e-30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Largest finite-time exponent for a batch of trajectories.

    states has shape [time, batch, hidden]. The returned tensors are the
    per-trajectory mean exponent and final unit tangent.
    """
    if states.ndim != 3 or len(states) < 2:
        raise ValueError("states must have shape [time,batch,hidden]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    tangent = torch.randn(states.shape[1:], generator=generator, dtype=torch.float32).to(states.device)
    tangent = torch.nn.functional.normalize(tangent, dim=-1)
    total = torch.zeros(states.shape[1], device=states.device, dtype=torch.float64)
    for state in states[:-1]:
        _, next_tangent = torch.func.jvp(mapping, (state,), (tangent,))
        growth = torch.linalg.vector_norm(next_tangent.float(), dim=-1).clamp_min(float(epsilon))
        total += growth.double().log()
        tangent = (next_tangent / growth.unsqueeze(-1)).detach()
    return (total / (len(states) - 1)).float(), tangent


@dataclass
class ShootingResult:
    orbit: torch.Tensor
    loss: float
    normalized_residual_p95: float
    iterations: int
    converged: bool
    minimal_period: int


def multiple_shooting(
    mapping: TensorMap,
    initial_orbit: torch.Tensor,
    *,
    orbit_scale: float,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
    history_size: int = 20,
) -> ShootingResult:
    """Fit F(x_i)=x_(i+1) without backpropagating through F**p."""
    if initial_orbit.ndim != 2 or len(initial_orbit) < 1:
        raise ValueError("initial_orbit must have shape [period, hidden]")
    value = torch.nn.Parameter(initial_orbit.detach().clone())
    denominator = max(float(orbit_scale) ** 2, 1e-24)
    optimizer = torch.optim.LBFGS(
        [value],
        lr=0.8,
        max_iter=int(max_iterations),
        tolerance_grad=float(tolerance),
        tolerance_change=float(tolerance),
        history_size=int(history_size),
        line_search_fn="strong_wolfe",
    )
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        optimizer.zero_grad(set_to_none=True)
        predicted = mapping(value)
        residual = predicted - value.roll(-1, 0)
        loss = residual.square().sum(dim=-1).mean() / denominator
        loss.backward()
        calls += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        orbit = value.detach()
        residual = torch.linalg.vector_norm(mapping(orbit) - orbit.roll(-1, 0), dim=-1)
        normalized = residual / max(float(orbit_scale), 1e-12)
        loss = float(residual.square().mean()) / denominator
        p95 = float(torch.quantile(normalized.float(), 0.95))
    return ShootingResult(
        orbit=orbit,
        loss=loss,
        normalized_residual_p95=p95,
        iterations=calls,
        converged=math.isfinite(loss) and p95 <= max(math.sqrt(float(tolerance)), 1e-6),
        minimal_period=minimal_repeated_period(orbit),
    )


def monodromy_action(mapping: TensorMap, orbit: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    result = vector
    for state in orbit:
        _, result = torch.func.jvp(mapping, (state.unsqueeze(0),), (result.unsqueeze(0),))
        result = result[0]
    return result.detach().float()


def arnoldi_eigenvalues(
    action: Callable[[torch.Tensor], torch.Tensor],
    dimension: int,
    *,
    krylov_dimension: int,
    seed: int,
    tolerance: float = 1e-10,
) -> torch.Tensor:
    """Ritz eigenvalues of an implicit square linear operator."""
    maximum = min(int(krylov_dimension), int(dimension))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    first = torch.randn(dimension, generator=generator, dtype=torch.float32)
    first = first / first.norm().clamp_min(1e-12)
    basis = [first]
    hessenberg = torch.zeros((maximum + 1, maximum), dtype=torch.float64)
    used = 0
    for column in range(maximum):
        value = action(basis[column]).detach().float().cpu()
        for row in range(column + 1):
            coefficient = torch.dot(basis[row], value)
            hessenberg[row, column] = coefficient.double()
            value = value - coefficient * basis[row]
        norm = value.norm()
        hessenberg[column + 1, column] = norm.double()
        used = column + 1
        if float(norm) <= tolerance or column + 1 == maximum:
            break
        basis.append(value / norm)
    return torch.linalg.eigvals(hessenberg[:used, :used])


def floquet_summary(
    mapping: TensorMap,
    orbit: torch.Tensor,
    *,
    seed: int,
    dimensions: Sequence[int] = (16, 32),
) -> dict[str, float | str | list[float]]:
    radii: list[float] = []
    leading_real: list[float] = []
    leading_imag: list[float] = []
    for size in dimensions:
        eigenvalues = arnoldi_eigenvalues(
            lambda vector: monodromy_action(mapping, orbit, vector.to(orbit.device)).cpu(),
            orbit.shape[-1],
            krylov_dimension=int(size),
            seed=int(seed),
        )
        index = int(torch.argmax(eigenvalues.abs()))
        leading = eigenvalues[index]
        radii.append(float(leading.abs()))
        leading_real.append(float(leading.real))
        leading_imag.append(float(leading.imag))
    agreement = abs(radii[-1] - radii[0]) / max(radii[-1], radii[0], 1e-12)
    radius = radii[-1]
    stability = "stable" if radius < 0.98 and agreement < 0.05 else "unstable" if radius > 1.02 else "boundary"
    return {
        "krylov_dimensions": [int(value) for value in dimensions],
        "spectral_radius_estimates": radii,
        "leading_real_estimates": leading_real,
        "leading_imag_estimates": leading_imag,
        "relative_disagreement": agreement,
        "leading_multiplier_modulus": radius,
        "stability": stability,
    }


def perturbation_recovery(
    mapping: TensorMap,
    orbit: torch.Tensor,
    *,
    seed: int,
    directions: int = 16,
    relative_scales: Sequence[float] = (1e-6, 1e-4, 1e-2),
    periods: int = 10,
) -> dict[str, float | int]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    scale = max(float(torch.linalg.vector_norm(orbit - orbit.mean(0), dim=1).median()), 1e-12)
    initial: list[torch.Tensor] = []
    for relative in relative_scales:
        noise = torch.randn((int(directions), orbit.shape[-1]), generator=generator).to(orbit.device)
        noise = torch.nn.functional.normalize(noise, dim=-1) * (float(relative) * scale)
        initial.append(orbit[0].unsqueeze(0) + noise)
    states = torch.cat(initial, dim=0)
    initial_distance = phase_invariant_orbit_distance(states, orbit)
    with torch.no_grad():
        for _ in range(int(periods) * len(orbit)):
            states = mapping(states)
    final_distance = phase_invariant_orbit_distance(states, orbit)
    recovered = final_distance <= initial_distance
    return {
        "perturbation_count": int(len(states)),
        "recovered_count": int(recovered.sum()),
        "recovery_fraction": float(recovered.float().mean()),
        "initial_distance_median": float(initial_distance.median()),
        "final_distance_median": float(final_distance.median()),
    }


def classify_screen(
    *,
    lag1_normalized_p95: float,
    best_lag: int,
    best_normalized_p95: float,
    second_normalized_p95: float,
    lyapunov: float | None,
    window_lags: Sequence[int] = (),
    numerical_floor_ratio: float = 1e-5,
) -> str:
    if lag1_normalized_p95 <= numerical_floor_ratio:
        return "fixed_candidate"
    if lyapunov is not None and lyapunov > 0.01 and best_normalized_p95 > 0.05:
        return "expanding_candidate"
    stable_windows = len(window_lags) >= 2 and len(set(int(value) for value in window_lags)) == 1
    prominence = second_normalized_p95 / max(best_normalized_p95, 1e-12)
    if best_lag > 1 and best_normalized_p95 <= 0.05 and prominence >= 1.1 and stable_windows:
        return "recurrent_candidate"
    if best_lag > 1 and best_normalized_p95 <= 0.1 and len(set(int(value) for value in window_lags)) > 1:
        return "quasiperiodic_candidate"
    return "transient_or_unresolved"


def final_classification(
    *,
    screen_label: str,
    period: int | None,
    shooting_residual: float | None,
    floquet_stability: str | None,
    recovery_fraction: float | None,
    precision_consistent: bool | None,
) -> str:
    if precision_consistent is False:
        return "numerical_cycle"
    if period is not None and shooting_residual is not None and shooting_residual <= 1e-5:
        if period == 1 and floquet_stability == "stable":
            return "stable_fixed_point"
        if period > 1 and floquet_stability == "stable" and (recovery_fraction or 0.0) >= 0.9:
            return f"stable_periodic_orbit({period})"
        if period > 1 and floquet_stability == "unstable":
            return f"unstable_periodic_orbit({period})"
    mapping = {
        "quasiperiodic_candidate": "quasiperiodic_candidate",
        "expanding_candidate": "expanding_or_chaotic_candidate",
        "recurrent_candidate": "long_transient",
        "transient_or_unresolved": "unresolved",
        "fixed_candidate": "unresolved",
    }
    return mapping.get(screen_label, "unresolved")


def interpolate_state_dict(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if left.keys() != right.keys():
        raise ValueError("state dict keys differ")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    output: dict[str, torch.Tensor] = {}
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"incompatible tensor for {key}")
        if torch.is_floating_point(a) or torch.is_complex(a):
            output[key] = torch.lerp(a, b, float(alpha))
        else:
            if not torch.equal(a, b):
                raise ValueError(f"non-floating buffer differs for {key}")
            output[key] = a.clone()
    return output


def atomic_torch_save(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
