# Qwen Dynamical Edge Normal-Dimension Plan

## Objective

Fill the Qwen row of the paper-alignment matrix using the same direct feedback-dynamics protocol, with a smaller first sample count because the model is larger.

## Method

- Model: `Qwen/Qwen2.5-0.5B`, revision `main`, offline cache.
- State/operator: `inputs_embeds -> final_hidden`, iterated as `x_{t+1}=f(x_t)`.
- Scale: `sequence_length=128`, `num_samples=4`, `burn_in_steps=16`, `eval_steps=64`.
- Metrics: normalized Jacobian Frobenius norm, nearby trajectory distance, lagged state-space distance, fixed-projection return map, approximate Poincare section, and short-window Jacobian-product gain.

## Expected Output

The run should populate `results/`, then `scripts/build_paper_alignment_matrix.py` should update `results/processed/paper_alignment_matrix.csv` with a direct Qwen dynamics row.
