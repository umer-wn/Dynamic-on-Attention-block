# Qwen Dynamical Edge Burn256 Plan

## Objective

Check whether the unsettled Qwen samples in the normal-long run are simply caused by insufficient burn-in.

## Method

- Model: `Qwen/Qwen2.5-0.5B`, revision `main`, offline cache.
- Operator: `inputs_embeds -> final_hidden`, iterated as `x_{t+1}=f(x_t)`.
- Scale: `sequence_length=128`, `num_samples=4`.
- Longer burn-in: `burn_in_steps=256`, `eval_steps=128`.
- Metrics: normalized Jacobian Frobenius norm, nearby trajectory distance, lagged state-space distance, fixed-projection return map, and approximate Poincare section.
- Product-Jacobian probes are disabled in this check so runtime focuses on convergence and the direct paper criterion.

## Decision Rule

If this run still has unsettled samples, the Qwen row should remain marked as only partially asymptotic rather than treated as a clean paper-comparable attractor estimate.
