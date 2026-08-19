# Pythia Residual Alpha 0.1 Probe Plan

## Objective

Run a negative-control restoration attempt: test whether an external residual update can move the normalized Frobenius estimate toward the paper's edge criterion without actually changing the pretrained language model.

## Method

- Baseline: `pythia_dynamical_edge_normal`.
- Intervention: `x_{t+1} = 0.9 x_t + 0.1 f(x_t)`.
- Scale: `sequence_length=128`, `num_samples=4`, `burn_in_steps=16`, `eval_steps=64`.
- Product-Jacobian probes are disabled because the goal is to test the direct normalized Frobenius effect of the external residual update.

## Interpretation

If this probe approaches `||J*||/sqrt(N)=1`, it should be interpreted as an artifact of adding identity dynamics, not as evidence that the language model itself reproduces the paper's edge-of-chaos result.
