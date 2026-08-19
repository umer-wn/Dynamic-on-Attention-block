# Qwen Dynamical Edge Normal Long Plan

## Objective

Resolve the main caveat in the first Qwen direct-dynamics row: the 64-step evaluation trajectory did not settle, so the normalized Jacobian estimate was not fully paper-comparable.

## Method

- Model: `Qwen/Qwen2.5-0.5B`, revision `main`, offline cache.
- Operator: `inputs_embeds -> final_hidden`, iterated as `x_{t+1}=f(x_t)`.
- Scale: `sequence_length=128`, `num_samples=4`.
- Longer trajectory: `burn_in_steps=64`, `eval_steps=128`.
- Metrics: normalized Jacobian Frobenius norm, nearby trajectory distance, lagged state-space distance, fixed-projection return map, approximate Poincare section, and short-window Jacobian-product gain.

## Decision Rule

The result can replace the provisional Qwen row only if final trajectory step deltas are small enough to mark the run as settled in `paper_alignment_matrix.csv`.
