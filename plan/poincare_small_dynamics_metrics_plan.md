# Poincare And Small-Scale Dynamics Metrics Plan

## Understand

This plan updates the next-stage paper-method work after the initial LLM feedback-dynamics smoke tests.

Reference paper:

```text
arXiv:1909.05176
Optimal Machine Intelligence at the Edge of Chaos
https://arxiv.org/abs/1909.05176
```

The project should stay close to the paper's core setting:

```text
x_{t+1} = f(x_t)
```

where `f` is a same-dimensional nonlinear operator and the dynamics are analyzed after repeatedly feeding the output back as the next input.

For the current LLM implementation:

```text
x_t = inputs_embeds
f(x_t) = model(inputs_embeds=x_t).hidden_states[-1]
x_{t+1} = f(x_t)
```

Current completed checks:

```text
smoke: seq_len=32, samples=2, burn_in=2, eval=2, probes=1
small: seq_len=64, samples=4, burn_in=4, eval=4, probes=2
```

These confirmed the pipeline runs, but they are not enough to characterize the phase behavior.

## Scope

This plan intentionally does not add:

- full normal-dimension execution;
- cross-model expansion.

Those should wait until the new diagnostics are validated at small scale.

The next work should focus on adding and validating four diagnostics on small runs:

1. normalized Jacobian Frobenius norm;
2. Poincare-style trajectory plots;
3. multi-step Jacobian product eigenvalue/spectral-radius metric;
4. multi-step state-space distance metric.

## Goal

Build a small-scale diagnostics suite that can distinguish:

```text
fixed-like
periodic-like
bounded aperiodic
sensitive/chaotic-like
collapsed
divergent
```

The immediate goal is not to prove the paper conclusion. It is to verify that these diagnostics are coherent with each other on short LLM embedding feedback trajectories.

## Metric 1: Normalized Jacobian Frobenius Norm

This is the already implemented paper-style metric:

```text
J_t = d f(x_t) / d x_t
edge_metric_t = ||J_t||_F / sqrt(N)
```

For post-burn-in states:

```text
edge_metric_geomean = exp(mean_t(log(edge_metric_t)))
```

Interpretation:

```text
edge_metric << 1: locally contracting
edge_metric ~= 1: near the paper's edge condition
edge_metric >> 1: locally expanding/sensitive
```

This remains the primary numerical metric because it most directly matches the paper.

## Metric 2: Poincare-Style Plots

### Reason

The paper discusses behavior near stable, pseudo-periodic, and chaotic-like regimes. A Poincare-style plot gives a visual diagnostic for whether a feedback trajectory collapses, cycles, or fills a bounded region.

In LLM embedding space, the state dimension is high, so the plot must be a projection. Treat it as a diagnostic visualization, not a proof of chaos.

### Small-Scale First Implementation

Save per-step trajectory summaries:

```text
step_index
state_norm = ||x_t||
step_delta = ||x_{t+1} - x_t||
nearby_distance = ||x_t - x'_t||
```

Use summary coordinates first:

```text
projection_1 = state_norm
projection_2 = step_delta
projection_3 = nearby_distance
```

Define an approximate section:

```text
s_t = state_norm_t - median(state_norm over post-burn-in trajectory)
section crossing: s_t <= 0 and s_{t+1} > 0
```

Plot:

```text
phase projection: state_norm vs step_delta, colored by step
Poincare section: step_delta at crossing vs nearby_distance at crossing
nearby distance curve: step -> nearby_distance
```

Expected outputs:

```text
results/processed/{experiment}__trajectory_summary.csv
results/processed/{experiment}__poincare_points.csv
results/figures/{experiment}__phase_projection.png
results/figures/{experiment}__poincare_section.png
results/figures/{experiment}__nearby_distance_by_step.png
```

## Metric 3: Multi-Step Jacobian Product Eigenvalue Metric

### Reason

For iterated dynamics, one-step Jacobian norms can miss multi-step stability. The paper's asymptotic view suggests checking behavior along a trajectory, where perturbations evolve through products of local Jacobians:

```text
P_{t,k} = J_{t+k-1} J_{t+k-2} ... J_t
```

The eigenvalues of this product, or a matrix-free proxy for its largest magnitude eigenvalue, indicate whether perturbations grow or shrink over multiple feedback steps.

### Small-Scale Practical Metric

Do not construct the full product matrix.

Use JVP chaining to apply the product to a vector:

```text
v_0 = random unit vector
v_{i+1} = J_{t+i} v_i
gain_k = ||v_k|| / ||v_0||
log_gain_per_step = log(gain_k) / k
```

For a spectral-radius-like estimate, repeat with multiple random probes:

```text
product_gain_max = max_probe(||P_{t,k} v|| / ||v||)
product_log_gain_max = log(product_gain_max) / k
```

If later needed, add power iteration on the product operator:

```text
P(v) = J_{t+k-1} ... J_t v
```

But first use random-probe gain because it is simpler and safer.

Interpretation:

```text
product_log_gain_max < 0: multi-step contraction
product_log_gain_max ~= 0: multi-step edge-like
product_log_gain_max > 0: multi-step expansion/sensitivity
```

Proposed config fields:

```yaml
dynamics:
  product_jacobian_windows: [2, 4, 8]
  product_jacobian_probes: 4
  save_product_jacobian_metrics: true
```

Expected output fields:

```text
product_window
product_gain_mean
product_gain_max
product_log_gain_mean
product_log_gain_max
```

Important note:

This metric is a matrix-free estimate of multi-step Jacobian product growth. It should be described as a spectral-radius proxy unless exact eigenvalue computation is implemented for a very small toy operator.

## Metric 4: Multi-Step State-Space Distance

### Reason

The paper's dynamics are about long-term behavior, not just one-step change. A direct state-space metric can show whether the trajectory is settling, cycling, or remaining sensitive.

Use two distance families:

### A. Lagged Self-Distance

For a trajectory after burn-in:

```text
D_lag(k, t) = ||x_{t+k} - x_t||
```

Summarize:

```text
lag_distance_mean_k
lag_distance_min_k
lag_distance_std_k
```

Interpretation:

```text
small D_lag for all k: fixed-like
small D_lag at specific k: periodic-like
large bounded D_lag: aperiodic bounded
growing D_lag with norm growth: divergent
```

### B. Nearby-Trajectory Distance

This already exists in the current pipeline:

```text
x'_0 = x_0 + epsilon * noise
D_near(t) = ||x'_t - x_t||
```

Add normalized growth:

```text
nearby_growth_ratio = D_near(final) / D_near(first)
nearby_log_growth_per_step = log(nearby_growth_ratio) / eval_steps
```

Interpretation:

```text
nearby_log_growth_per_step < 0: perturbations shrink
nearby_log_growth_per_step ~= 0: edge-like neutral perturbations
nearby_log_growth_per_step > 0: sensitive dynamics
```

Proposed config fields:

```yaml
dynamics:
  lag_distance_windows: [1, 2, 4, 8]
  save_state_distance_metrics: true
```

Expected output fields:

```text
lag_window
lag_distance_mean
lag_distance_min
lag_distance_std
nearby_growth_ratio
nearby_log_growth_per_step
```

## Priority Order

The priority should be small-scale validation first.

### Priority 1: Save Per-Step Trajectory Summaries

Reason:

Poincare plots and multi-step state distances both require per-step trajectory data. This is the shared foundation.

Implement first:

```text
compute_dynamical_edge.py:
  save_trajectory_summary: true
```

Validate on:

```text
configs/pythia_dynamical_edge_smoke.yaml
```

Success:

- trajectory CSV/JSONL exists;
- row count matches samples * eval_steps;
- state_norm, step_delta, nearby_distance are finite.

### Priority 2: Multi-Step State-Space Distance

Reason:

It is cheaper than additional Jacobian products and immediately checks whether the trajectory behaves like fixed/periodic/bounded/sensitive dynamics.

Validate on:

```text
seq_len=32
samples=2
burn_in=4
eval_steps=16
```

Success:

- lag distances are finite;
- nearby growth ratio is finite;
- labels are stable under one repeated seed.

### Priority 3: Poincare Plots

Reason:

Once trajectory summaries and lag distances exist, Poincare plotting is mostly analysis code. It helps visually audit whether labels make sense.

Validate on:

```text
seq_len=32
samples=2
burn_in=8
eval_steps=32
```

Success:

- phase projection and Poincare section figures are generated;
- crossings are recorded when present;
- no claim is made when no crossings occur.

### Priority 4: Multi-Step Jacobian Product Metric

Reason:

This is more expensive and more technically subtle than state distances, but it is closer to trajectory-wise stability and local Lyapunov behavior.

Validate on:

```text
seq_len=32
samples=1
burn_in=4
eval_steps=8
product_windows=[2, 4]
product_probes=2
```

Success:

- product gains are finite;
- product log gains qualitatively agree with nearby trajectory growth;
- runtime is acceptable.

### Priority 5: Slightly Larger Consistency Check

Only after Priorities 1-4 pass:

```text
seq_len=64
samples=4
burn_in=8
eval_steps=32
frobenius_eval_states=4
frobenius_probes=4
product_windows=[2, 4, 8]
product_probes=4
lag_distance_windows=[1, 2, 4, 8]
```

Do not move to larger runs or other models until this consistency check is understood.

## Expected Code Changes

Add or extend:

```text
src/dynamics.py
scripts/compute_dynamical_edge.py
scripts/analyze_dynamical_edge.py
scripts/analyze_dynamical_poincare.py
configs/pythia_dynamical_edge_metrics_smoke.yaml
configs/pythia_dynamical_edge_metrics_small.yaml
reports/pythia_dynamical_edge_metrics_findings.md
```

## Expected Outputs

Raw:

```text
results/raw/{experiment}__...__dynamical_edge.jsonl
results/raw/{experiment}__...__dynamics_trajectory.jsonl
```

Processed:

```text
results/processed/{experiment}__dynamical_edge_rows.csv
results/processed/{experiment}__dynamical_edge_summary.csv
results/processed/{experiment}__trajectory_summary.csv
results/processed/{experiment}__poincare_points.csv
results/processed/{experiment}__product_jacobian_metrics.csv
results/processed/{experiment}__state_distance_metrics.csv
```

Figures:

```text
results/figures/{experiment}__phase_projection.png
results/figures/{experiment}__poincare_section.png
results/figures/{experiment}__nearby_distance_by_step.png
results/figures/{experiment}__lag_distance_by_window.png
results/figures/{experiment}__product_log_gain_by_window.png
```

## Review Risks

- Poincare plots are projected diagnostics in high-dimensional embedding space.
- Product Jacobian eigenvalues should initially be reported as matrix-free gain proxies, not exact eigenvalues.
- Short trajectories can suggest but not prove fixed, periodic, or chaotic phases.
- LLM embedding feedback is a constructed continuous operator, not ordinary token generation.
- More Jacobian product probes can become expensive quickly, so validation must stay small first.

## Next Concrete Work Items

1. Extend trajectory recording in `compute_dynamical_edge.py`.
2. Implement lagged state-distance metrics.
3. Implement Poincare analysis and plotting from trajectory summaries.
4. Implement matrix-free multi-step Jacobian product gain metrics.
5. Run only smoke/small validation configs until all four diagnostics agree qualitatively.
