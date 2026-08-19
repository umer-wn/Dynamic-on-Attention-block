from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch


TensorMap = Callable[[torch.Tensor], torch.Tensor]


def rademacher_like(reference: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    values = torch.randint(
        0,
        2,
        reference.shape,
        device=reference.device,
        generator=generator,
        dtype=torch.int64,
    )
    return values.to(torch.float32).mul_(2.0).sub_(1.0)


class SoftNextTokenRollingOperator:
    """Fixed-length differentiable rolling-window operator for a causal LM."""

    def __init__(self, model: torch.nn.Module, temperature: float = 1.0, position_mode: str = "reset"):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if position_mode != "reset":
            raise ValueError("pilot JVP path currently supports only autonomous reset positions")
        self.model = model
        self.temperature = float(temperature)
        self.position_mode = position_mode
        self.input_embedding = model.get_input_embeddings()
        self.output_embedding = model.get_output_embeddings()
        if self.input_embedding is None or self.output_embedding is None:
            raise ValueError("model must expose input and output embeddings")

    def _last_hidden(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        attention_mask = torch.ones((batch, length), device=x.device, dtype=torch.long)
        position_ids = torch.arange(length, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        model_dtype = next(self.model.parameters()).dtype
        x_model = x.to(dtype=model_dtype)
        if hasattr(self.model, "gpt_neox"):
            outputs = self.model.gpt_neox(
                inputs_embeds=x_model,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            return outputs.last_hidden_state[:, -1, :]
        outputs = self.model(
            inputs_embeds=x_model,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return outputs.hidden_states[-1][:, -1, :]

    def next_embedding_and_probs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._last_hidden(x)
        logits = self.output_embedding(hidden).float()
        probs = torch.softmax(logits / self.temperature, dim=-1)
        embedding = probs @ self.input_embedding.weight.float()
        return embedding, probs

    def next_embedding(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.next_embedding_and_probs(x)
        return embedding

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.next_embedding(x)
        return torch.cat([x[:, 1:, :], embedding.unsqueeze(1)], dim=1)


def shift_only_operator(x: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(x[:, :1, :])
    return torch.cat([x[:, 1:, :], zeros], dim=1)


def analytic_shift_normalized_frobenius(window_length: int) -> float:
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    return math.sqrt(max(window_length - 1, 0) / window_length)


@dataclass
class SoftTrajectoryResult:
    states: list[torch.Tensor]
    rows: list[dict[str, float | int]]
    initial_nearby_distance: float


def run_soft_trajectory(
    operator: SoftNextTokenRollingOperator,
    x0: torch.Tensor,
    burn_in_steps: int,
    eval_steps: int,
    epsilon: float,
    seed: int,
) -> SoftTrajectoryResult:
    generator = torch.Generator(device=x0.device)
    generator.manual_seed(int(seed))
    x = x0.detach().float()
    noise = torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)
    noise = noise / (noise.norm() + 1e-12)
    x_near = x + float(epsilon) * noise
    initial_distance = float((x_near - x).norm().cpu())
    states: list[torch.Tensor] = []
    rows: list[dict[str, float | int]] = []
    total = int(burn_in_steps) + int(eval_steps)
    with torch.no_grad():
        for step in range(total):
            previous = x
            next_embedding, probs = operator.next_embedding_and_probs(x)
            x = torch.cat([x[:, 1:, :], next_embedding.unsqueeze(1)], dim=1).float()
            x_near = operator(x_near).float()
            if step >= burn_in_steps:
                norm = float(x.norm().cpu())
                delta = float((x - previous).norm().cpu())
                distance = float((x_near - x).norm().cpu())
                entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)).mean().cpu())
                top1_probability = float(probs.max(dim=-1).values.mean().cpu())
                rows.append(
                    {
                        "step_index": step - int(burn_in_steps),
                        "state_norm": norm,
                        "step_delta": delta,
                        "relative_step_delta": delta / max(norm, 1e-12),
                        "nearby_distance": distance,
                        "soft_entropy": entropy,
                        "soft_top1_probability": top1_probability,
                        "new_embedding_norm": float(next_embedding.norm().cpu()),
                    }
                )
                states.append(x.detach().clone())
    return SoftTrajectoryResult(states=states, rows=rows, initial_nearby_distance=initial_distance)


def estimate_innovation_frobenius(
    operator: SoftNextTokenRollingOperator,
    states: list[torch.Tensor],
    probes: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    if not states or probes <= 0:
        return {"frobenius_states": 0, "frobenius_probes": int(probes), "innovation_local": []}
    generator = torch.Generator(device=states[0].device)
    generator.manual_seed(int(seed))
    local_innovation_total: list[float] = []
    local_innovation_output: list[float] = []
    local_total: list[float] = []
    _, length, hidden = states[0].shape
    active_dim = int(length * hidden)
    shift_sq = float((length - 1) * hidden)
    for state in states:
        norm_sq: list[torch.Tensor] = []
        for _ in range(int(probes)):
            v = rademacher_like(state, generator=generator)
            _, jv = torch.autograd.functional.jvp(
                operator.next_embedding,
                (state.detach().requires_grad_(True),),
                (v,),
                create_graph=False,
                strict=False,
            )
            norm_sq.append(jv.float().pow(2).sum().detach())
        mean_sq = float(torch.stack(norm_sq).mean().cpu())
        innovation_total = math.sqrt(mean_sq / active_dim)
        innovation_output = math.sqrt(mean_sq / hidden)
        total = math.sqrt((shift_sq + mean_sq) / active_dim)
        local_innovation_total.append(innovation_total)
        local_innovation_output.append(innovation_output)
        local_total.append(total)

    def geometric(values: list[float]) -> float:
        return math.exp(sum(math.log(max(v, 1e-12)) for v in values) / len(values))

    shift = analytic_shift_normalized_frobenius(length)
    return {
        "frobenius_states": len(states),
        "frobenius_probes": int(probes),
        "active_dim": active_dim,
        "shift_normalized_frobenius": shift,
        "innovation_local": local_innovation_total,
        "innovation_output_local": local_innovation_output,
        "total_local": local_total,
        "innovation_geomean": geometric(local_innovation_total),
        "innovation_output_geomean": geometric(local_innovation_output),
        "total_geomean": geometric(local_total),
        "shift_fraction_of_total_squared": (shift * shift) / max(geometric(local_total) ** 2, 1e-12),
    }


def estimate_maximal_lyapunov(
    operator: TensorMap,
    states: list[torch.Tensor],
    probes: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    if not states or probes <= 0:
        return {"lyapunov_steps": 0, "lyapunov_probes": int(probes), "lyapunov_exponents": []}
    generator = torch.Generator(device=states[0].device)
    generator.manual_seed(int(seed))
    exponents: list[float] = []
    for _ in range(int(probes)):
        v = rademacher_like(states[0], generator=generator)
        v = v / (v.norm() + 1e-12)
        log_growth = 0.0
        valid = 0
        for state in states:
            _, jv = torch.autograd.functional.jvp(
                operator,
                (state.detach().requires_grad_(True),),
                (v,),
                create_graph=False,
                strict=False,
            )
            norm = float(jv.float().norm().cpu())
            if not math.isfinite(norm) or norm <= 0:
                log_growth = float("-inf")
                valid += 1
                break
            log_growth += math.log(norm)
            valid += 1
            v = jv.detach().float() / norm
        exponents.append(log_growth / max(valid, 1))
    finite = [v for v in exponents if math.isfinite(v)]
    mean = sum(finite) / len(finite) if finite else float("-inf")
    std = math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite)) if finite else 0.0
    return {
        "lyapunov_steps": len(states),
        "lyapunov_probes": int(probes),
        "lyapunov_exponents": exponents,
        "maximal_lyapunov_mean": mean,
        "maximal_lyapunov_std": std,
        "maximal_lyapunov_max": max(finite) if finite else float("-inf"),
    }


def hard_argmax_rollout(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    steps: int,
) -> dict[str, object]:
    window = input_ids.detach().clone()
    generated: list[int] = []
    seen: dict[tuple[int, ...], int] = {tuple(int(v) for v in window[0].tolist()): 0}
    cycle_start: int | None = None
    cycle_length: int | None = None
    with torch.no_grad():
        for step in range(int(steps)):
            attention_mask = torch.ones_like(window)
            position_ids = torch.arange(window.shape[1], device=window.device).unsqueeze(0)
            outputs = model(
                input_ids=window,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            token = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
            generated.append(token)
            window = torch.cat(
                [window[:, 1:], torch.tensor([[token]], device=window.device, dtype=window.dtype)], dim=1
            )
            key = tuple(int(v) for v in window[0].tolist())
            if key in seen and cycle_length is None:
                cycle_start = seen[key]
                cycle_length = step + 1 - seen[key]
            seen.setdefault(key, step + 1)
    unique_ratio = len(set(generated)) / max(len(generated), 1)
    adjacent_repeat_fraction = (
        sum(int(a == b) for a, b in zip(generated[:-1], generated[1:])) / max(len(generated) - 1, 1)
    )
    return {
        "generated_token_ids": generated,
        "hard_unique_token_ratio": unique_ratio,
        "hard_adjacent_repeat_fraction": adjacent_repeat_fraction,
        "hard_cycle_start": cycle_start,
        "hard_cycle_length": cycle_length,
        "hard_unique_windows": len(seen),
    }
