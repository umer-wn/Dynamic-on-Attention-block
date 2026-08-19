# Pythia Dynamical Edge Normal-Dimension Findings

## Run

- Date: 2026-07-10
- Config: `experiments/pythia_dynamical_edge_normal/configs/pythia_dynamical_edge_normal.yaml`
- Plan: `experiments/pythia_dynamical_edge_normal/plan/pythia_dynamical_edge_normal_plan.md`
- Results: `experiments/pythia_dynamical_edge_normal/results/`
- Model: `EleutherAI/pythia-70m`, revision `main`, offline cache
- Scale: `sequence_length=128`, `num_samples=8`, `burn_in_steps=16`, `eval_steps=64`

## Pipeline Notes

The experiment follows the new directory convention:

- Plan files are under `plan/`.
- Raw, processed, and figure outputs are under `results/`.
- This report is under `reports/`.
- Each generated figure has a neighboring `.md` explanation, and the full image index is `results/figures/pythia_dynamical_edge_normal__figure_manifest.md`.

## Main Results

- Mean normalized Jacobian Frobenius norm: `0.463930`.
- Median normalized Jacobian Frobenius norm: `0.463525`.
- Mean edge distance log: `0.768022`.
- Diverged fraction: `0.0`.
- Collapsed fraction: `0.0`.
- Nearby perturbation average growth ratio: `0.661515`.
- Nearby perturbation average log growth per step: `-0.006536`.

These values indicate a contractive feedback dynamic for this configuration, not a near-critical or expansive one.

## Multi-Step Jacobian Product

Mean values by product window:

- Window 2: `product_log_gain_max=-0.735000`, `product_gain_max=0.229935`.
- Window 4: `product_log_gain_max=-0.673938`, `product_gain_max=0.067505`.

The product-Jacobian estimate is negative for both windows, consistent with contraction across multiple feedback steps.

## Multi-Step State-Space Distance

Mean lagged hidden-state distances grow with the lag window:

- Lag 1: `0.090742`
- Lag 2: `0.164728`
- Lag 4: `0.275743`
- Lag 8: `0.409220`
- Lag 16: `0.548498`
- Lag 32: `0.840670`

This means states further apart in feedback time are more separated, but the per-step perturbation and Jacobian-product measurements still show contraction toward a fixed or slowly moving attractor.

## Return Map and Poincare Figures

The fixed-projection return map uses one fixed random projection direction per sample and keeps that direction unchanged across all feedback steps. This is the preferred plot for paper-style `z_t` versus `z_{t+1}` inspection in this run.

The approximate Poincare section uses median state-norm crossings as an empirical section. It is useful as a recurrence diagnostic, but it is not a proof that the full high-dimensional trajectory has the same geometry as the paper's plotted low-dimensional system.

## Interpretation

For Pythia-70M at normal sequence length 128, the current feedback operator appears strongly contractive after burn-in. The result does not support an edge-of-chaos conclusion for this exact operator/update choice. A next experiment should either:

- increase sample count while keeping this exact config, or
- test alternative model checkpoints/operators before moving to other model families.
