# Pythia Dynamical Edge Projection64 Findings

## Goal

Add paper-style return-map plots for the LLM feedback trajectory:

```text
z_t = phi(x_t)
return map = (z_t, z_{t+1})
```

This fixes a limitation of the earlier Poincare-style diagnostic, which plotted `step_delta` against `nearby_distance` rather than adjacent projected states.

## Fixed Projection Rule

The run uses:

```text
trajectory_projection: fixed_random
projection_seed: 20260709
```

For each sample, one random projection vector is generated once:

```text
r ~ N(0, I)
r = r / ||r||
z_t = <r, x_t>
```

The same `r` is reused for every step in that sample trajectory. The projection direction is not updated inside the feedback loop.

This is important because changing the projection direction over time would make `(z_t, z_{t+1})` meaningless as a return map.

## Run Setup

Config:

```text
configs/pythia_dynamical_edge_projection64.yaml
```

Setup:

```text
model: EleutherAI/pythia-70m
sequence_length: 64
samples: 2
burn_in_steps: 8
eval_steps: 64
frobenius_eval_states: 2
frobenius_probes: 2
product_jacobian_probes: 0
```

Product-Jacobian metrics were disabled for this run so the trajectory could be inspected cheaply over more steps.

## Outputs

Raw:

```text
results/raw/pythia_dynamical_edge_projection64__EleutherAI__pythia-70m__main__seq64__dynamical_edge.jsonl
results/raw/pythia_dynamical_edge_projection64__EleutherAI__pythia-70m__main__seq64__dynamics_trajectory.jsonl
results/raw/pythia_dynamical_edge_projection64__EleutherAI__pythia-70m__main__seq64__state_distance_metrics.jsonl
```

Processed:

```text
results/processed/pythia_dynamical_edge_projection64__dynamical_edge_summary.csv
results/processed/pythia_dynamical_edge_projection64__trajectory_summary.csv
results/processed/pythia_dynamical_edge_projection64__return_map_points.csv
results/processed/pythia_dynamical_edge_projection64__pca_summary_return_map_points.csv
```

Figures:

```text
results/figures/pythia_dynamical_edge_projection64__return_map_projection.png
results/figures/pythia_dynamical_edge_projection64__return_map_pca_summary.png
results/figures/pythia_dynamical_edge_projection64__phase_projection.png
results/figures/pythia_dynamical_edge_projection64__nearby_distance_by_step.png
```

## Main Numbers

Summary:

```text
mean_normalized_frobenius = 0.4642
median_normalized_frobenius = 0.4642
diverged_fraction = 0.0
collapsed_fraction = 0.0
```

The return-map projection converges rapidly.

Selected trajectory values:

| sample | step | step_delta | nearby_distance | abs(z_{t+1} - z_t) |
|---:|---:|---:|---:|---:|
| 0 | 0 | 4.1939 | 0.000758 | 0.025454 |
| 0 | 8 | 0.5330 | 0.000115 | 0.002680 |
| 0 | 16 | 0.0782 | 0.000078 | 0.000397 |
| 0 | 32 | 0.0017 | 0.000077 | 0.000008 |
| 0 | 48 | 0.0001 | 0.000091 | 0.000000 |
| 1 | 0 | 13.4517 | 0.003264 | 0.015741 |
| 1 | 8 | 1.7526 | 0.000456 | 0.003897 |
| 1 | 16 | 0.2588 | 0.000247 | 0.000603 |
| 1 | 32 | 0.0058 | 0.000239 | 0.000008 |
| 1 | 48 | 0.0003 | 0.000282 | 0.000004 |

## Iteration-Step Guidance

For this direct Pythia-70M embedding feedback operator:

```text
16 eval steps: near fixed point, but still visibly moving
32 eval steps: effectively converged for return-map inspection
48-64 eval steps: mostly numerical-level movement
```

Recommended next small runs:

```text
burn_in_steps: 8
eval_steps: 32
```

If the goal is only to estimate the asymptotic fixed point or return map, 32 post-burn-in steps appear sufficient for this operator. If the goal is to test whether an operator variant has slower periodic/aperiodic dynamics, use 64 steps.

## Interpretation

The strict projected return map:

```text
(z_t, z_{t+1})
```

collapses close to the diagonal fixed point. Together with:

```text
normalized Frobenius < 1
nearby distance shrinking
step_delta -> 0
```

this supports the same conclusion as the previous small metrics run:

```text
direct final_hidden -> inputs_embeds feedback is strongly contracting in these small tests.
```

This return map is closer to the paper-style `(x_t, x_{t+1})` plot, with the necessary high-dimensional adaptation:

```text
x_t is high-dimensional, so z_t = phi(x_t)
```

The projection direction is fixed over the trajectory, so the return map is interpretable.
