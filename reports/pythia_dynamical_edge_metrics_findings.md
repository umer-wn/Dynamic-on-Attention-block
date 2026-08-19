# Pythia Dynamical Edge Metrics Findings

## Goal

Run small-scale validation for the paper-style feedback dynamics metrics before considering any larger experiment.

This validation follows the small-scale priority plan:

```text
plan/poincare_small_dynamics_metrics_plan.md
```

The tested operator is:

```text
x_t = inputs_embeds
f(x_t) = model(inputs_embeds=x_t).hidden_states[-1]
x_{t+1} = f(x_t)
```

The run adds three diagnostics beyond the existing normalized Jacobian Frobenius norm:

1. Poincare-style phase plots.
2. Multi-step Jacobian product gain, used as a spectral-radius proxy.
3. Multi-step state-space distances.

## Code Added Or Extended

Extended:

```text
src/dynamics.py
scripts/compute_dynamical_edge.py
tests/test_dynamics.py
```

Added:

```text
scripts/analyze_dynamical_poincare.py
configs/pythia_dynamical_edge_metrics_smoke.yaml
configs/pythia_dynamical_edge_metrics_small.yaml
```

## Unit Tests

Command:

```text
PYTHONPATH=/data1/luohaoming/model_feature \
/public/luohaoming/model_feature/.venv/bin/python \
-m unittest discover \
-s /data1/luohaoming/model_feature/tests \
-p 'test_*.py' \
-t /data1/luohaoming/model_feature
```

Result:

```text
Ran 4 tests
OK
```

The tests cover:

- identity operator normalized Frobenius near `1`;
- contracting operator normalized Frobenius below `1`;
- finite lagged state-distance and nearby-growth metrics;
- multi-step Jacobian product gain for a linear contracting operator.

## Metrics Smoke Run

Config:

```text
configs/pythia_dynamical_edge_metrics_smoke.yaml
```

Setup:

```text
model: EleutherAI/pythia-70m
sequence_length: 32
samples: 2
burn_in_steps: 4
eval_steps: 16
frobenius_eval_states: 2
frobenius_probes: 2
product_jacobian_windows: [2, 4]
product_jacobian_probes: 2
lag_distance_windows: [1, 2, 4, 8]
```

Raw outputs:

```text
results/raw/pythia_dynamical_edge_metrics_smoke__EleutherAI__pythia-70m__main__seq32__dynamical_edge.jsonl
results/raw/pythia_dynamical_edge_metrics_smoke__EleutherAI__pythia-70m__main__seq32__dynamics_trajectory.jsonl
results/raw/pythia_dynamical_edge_metrics_smoke__EleutherAI__pythia-70m__main__seq32__state_distance_metrics.jsonl
results/raw/pythia_dynamical_edge_metrics_smoke__EleutherAI__pythia-70m__main__seq32__product_jacobian_metrics.jsonl
```

Row counts:

```text
dynamical_edge: 2
trajectory: 32
state_distance_metrics: 8
product_jacobian_metrics: 4
```

Main summary:

```text
mean_normalized_frobenius = 0.4645
median_normalized_frobenius = 0.4645
diverged_fraction = 0.0
collapsed_fraction = 0.0
```

Product Jacobian metric:

```text
window 2 product_log_gain_max ~= -0.73
window 4 product_log_gain_max ~= -0.67
```

Nearby trajectory metric:

```text
nearby_log_growth_per_step < 0 for both samples
```

Interpretation:

All smoke diagnostics agree that the direct feedback operator is locally and multi-step contracting in this small run.

## Metrics Small Run

Config:

```text
configs/pythia_dynamical_edge_metrics_small.yaml
```

Setup:

```text
model: EleutherAI/pythia-70m
sequence_length: 64
samples: 4
burn_in_steps: 8
eval_steps: 32
frobenius_eval_states: 4
frobenius_probes: 4
product_jacobian_windows: [2, 4, 8]
product_jacobian_probes: 4
lag_distance_windows: [1, 2, 4, 8]
```

Raw outputs:

```text
results/raw/pythia_dynamical_edge_metrics_small__EleutherAI__pythia-70m__main__seq64__dynamical_edge.jsonl
results/raw/pythia_dynamical_edge_metrics_small__EleutherAI__pythia-70m__main__seq64__dynamics_trajectory.jsonl
results/raw/pythia_dynamical_edge_metrics_small__EleutherAI__pythia-70m__main__seq64__state_distance_metrics.jsonl
results/raw/pythia_dynamical_edge_metrics_small__EleutherAI__pythia-70m__main__seq64__product_jacobian_metrics.jsonl
```

Row counts:

```text
dynamical_edge: 4
trajectory: 128
state_distance_metrics: 16
product_jacobian_metrics: 12
```

Main summary:

```text
mean_normalized_frobenius = 0.4643
median_normalized_frobenius = 0.4643
mean_edge_distance_log = 0.7671
diverged_fraction = 0.0
collapsed_fraction = 0.0
```

Product Jacobian metric:

```text
window 2 product_log_gain_max ~= -0.73
window 4 product_log_gain_max ~= -0.67
window 8 product_log_gain_max ~= -0.54 to -0.56
```

Nearby trajectory metric:

```text
nearby_log_growth_per_step < 0 for all 4 samples
```

Lagged state distance:

```text
lag distance increases with lag window
distances remain finite
```

Poincare points:

```text
4 section crossings recorded, one per sample
```

Figures generated:

```text
results/figures/pythia_dynamical_edge_metrics_small__phase_projection.png
results/figures/pythia_dynamical_edge_metrics_small__poincare_section.png
results/figures/pythia_dynamical_edge_metrics_small__nearby_distance_by_step.png
results/figures/pythia_dynamical_edge_metrics_small__lag_distance_by_window.png
results/figures/pythia_dynamical_edge_metrics_small__product_log_gain_by_window.png
```

## Interpretation

The new diagnostics are internally consistent on both smoke and small runs.

Observed pattern:

```text
normalized Frobenius < 1
multi-step Jacobian product log gain < 0
nearby trajectory log growth < 0
no divergence
no collapse
finite lag distances
```

This supports a cautious small-scale conclusion:

```text
For direct final_hidden -> inputs_embeds feedback on Pythia-70M,
the tested embedding-space operator behaves as a contracting bounded system,
not an edge-like or expanding system.
```

This is not yet a final scientific conclusion about language models or the paper's broader claim. It is a validation that the added metrics work and agree qualitatively at small scale.

## Runtime Note

The metrics small run took about `2.5` minutes for 4 samples on one A100.

Most of the cost comes from multi-step Jacobian product JVP chains. This confirms that product metrics should remain after trajectory/distance/Poincare diagnostics in the execution priority.

## Next Step

Before any larger run, repeat the small metrics config with one changed seed or one operator variant:

```text
operator_update: residual
residual_alpha: 0.25
```

The next check should ask whether the contracting result is robust to the embedding-space feedback definition.
