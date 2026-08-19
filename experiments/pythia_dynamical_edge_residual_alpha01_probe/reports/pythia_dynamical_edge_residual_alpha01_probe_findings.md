# Pythia Residual Alpha 0.1 Probe Findings

## Run

- Date: 2026-07-10
- Config: `experiments/pythia_dynamical_edge_residual_alpha01_probe/configs/pythia_dynamical_edge_residual_alpha01_probe.yaml`
- Plan: `experiments/pythia_dynamical_edge_residual_alpha01_probe/plan/pythia_dynamical_edge_residual_alpha01_probe_plan.md`
- Baseline: `pythia_dynamical_edge_normal`

## Result

- Mean normalized Frobenius: `0.945701`
- Median normalized Frobenius: `0.945701`
- Diverged fraction: `0.0`
- Collapsed fraction: `0.0`
- Settled fraction: `0.0`
- Max final step delta: `2.413796`
- Nearby log growth mean: `-0.035971`

## Interpretation

Adding an external residual update can move the normalized Frobenius estimate close to the paper's threshold of 1. However, this is not a faithful reproduction of the paper's result:

- the edge-like value is largely induced by the explicit identity term `0.9 x_t`;
- the trajectories are not settled under the current criterion;
- the pretrained language-model operator itself has not changed.

This probe is a useful negative control. It shows that it is possible to manufacture a near-edge scalar by changing the update rule, but that would not demonstrate that pretrained natural-language models intrinsically sit at the edge of chaos.
