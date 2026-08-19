# GPT-2 Dynamical Edge Normal-Dimension Findings

## Run

- Date: 2026-07-10
- Config: `experiments/gpt2_dynamical_edge_normal/configs/gpt2_dynamical_edge_normal.yaml`
- Plan: `experiments/gpt2_dynamical_edge_normal/plan/gpt2_dynamical_edge_normal_plan.md`
- Results: `experiments/gpt2_dynamical_edge_normal/results/`
- Model: `gpt2`, revision `main`, offline cache
- Scale: `sequence_length=128`, `num_samples=8`, `burn_in_steps=16`, `eval_steps=64`

## Main Results

- Mean normalized Jacobian Frobenius norm: `0.240475`.
- Median normalized Jacobian Frobenius norm: `0.126619`.
- Mean edge distance log: `1.692554`.
- Diverged fraction: `0.0`.
- Collapsed fraction: `0.0`.
- Final step delta mean: `0.001724`.
- Final step delta max: `0.006768`.
- Nearby perturbation average growth ratio: `0.764715`.
- Nearby perturbation average log growth per step: `-0.025544`.

## Multi-Step Jacobian Product

- Window 2: `product_log_gain_max=-0.864660`, `product_gain_max=0.515413`.
- Window 4: `product_log_gain_max=-0.665282`, `product_gain_max=0.491779`.

## Interpretation

GPT-2 settles within the configured 64 evaluation steps and is strongly below the paper's edge criterion `||J*||/sqrt(N)=1`. For this operator, GPT-2 is stable/contractive rather than near the edge of chaos.

Figures and per-figure explanations are in `results/figures/gpt2_dynamical_edge_normal__figure_manifest.md`.
