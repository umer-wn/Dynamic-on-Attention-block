# Qwen Dynamical Edge Normal-Dimension Findings

## Run

- Date: 2026-07-10
- Config: `experiments/qwen_dynamical_edge_normal/configs/qwen_dynamical_edge_normal.yaml`
- Plan: `experiments/qwen_dynamical_edge_normal/plan/qwen_dynamical_edge_normal_plan.md`
- Results: `experiments/qwen_dynamical_edge_normal/results/`
- Model: `Qwen/Qwen2.5-0.5B`, revision `main`, offline cache
- Scale: `sequence_length=128`, `num_samples=4`, `burn_in_steps=16`, `eval_steps=64`

## Main Results

- Mean normalized Jacobian Frobenius norm: `0.472415`.
- Median normalized Jacobian Frobenius norm: `0.470640`.
- Mean edge distance log: `0.750191`.
- Diverged fraction: `0.0`.
- Collapsed fraction: `0.0`.
- Final step delta mean: `36.720736`.
- Final step delta max: `101.925064`.
- Nearby perturbation average growth ratio: `1.830085`.
- Nearby perturbation average log growth per step: `-0.012734`.

## Multi-Step Jacobian Product

- Window 2: `product_log_gain_max=-0.560696`, `product_gain_max=0.326312`.
- Window 4: `product_log_gain_max=-0.366034`, `product_gain_max=0.263965`.

## Interpretation

Qwen has direct dynamics output, but this run is not asymptotically settled by the configured 64 evaluation steps. The normalized Frobenius value is below 1, but because the final step deltas remain large, this row should not yet be treated as a fully paper-comparable asymptotic result.

The next Qwen run should increase burn-in/evaluation length before using the row to decide whether Qwen is near the edge of chaos.

Figures and per-figure explanations are in `results/figures/qwen_dynamical_edge_normal__figure_manifest.md`.
