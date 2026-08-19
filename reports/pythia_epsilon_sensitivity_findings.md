# Pythia Perturbation-Epsilon Sensitivity Findings

## Status

Complete. Three float32 runs finished successfully on GPUs 5, 6, and 7. Each used four identical WikiText samples and the Pythia-70M residual feedback operator with `alpha=0.5`.

## Results

| epsilon | normalized Frobenius mean | nearby ratio mean | nearby log growth/step | nearby log-growth std |
|---:|---:|---:|---:|---:|
| 1e-3 | 0.729235 | 1.018912 | 0.000291 | 0.000235 |
| 1e-5 | 0.729235 | 1.013579 | 0.000208 | 0.000342 |
| 1e-7 | 0.729235 | 1.004718 | 0.000061 | 0.000718 |

The Jacobian-product results were exactly invariant across epsilon:

| window | mean product gain | mean log gain per step |
|---:|---:|---:|
| 2 | 0.535238 | -0.312522 |
| 4 | 0.294125 | -0.305939 |

No sample diverged or collapsed.

## Interpretation

The automatic-differentiation measurements pass the invariance check: normalized Frobenius and multi-step JVP products do not depend on finite perturbation epsilon.

Nearby-trajectory growth is less reliable. Its mean remains close to zero for all epsilons, but the estimate shifts systematically and its between-sample standard deviation becomes largest at `1e-7`. At this scale, float32 rounding and nonlinear finite-difference effects are comparable to the measured signal. Nearby perturbation growth should therefore remain a qualitative secondary diagnostic, not the primary criticality criterion.

This also resolves an apparent discrepancy in the earlier `alpha=0.5` run, whose nearby log growth was slightly negative: values at the order of `1e-4` per step are not stable enough in the present protocol to support a sign claim.

## Review and Method Changes

1. Retain Frobenius and JVP-product measurements as primary local/tangent diagnostics.
2. When using nearby trajectories, report an epsilon sweep and uncertainty; do not interpret only one epsilon.
3. Use `1e-3` or `1e-4` for float32 finite perturbations unless a precision study supports a smaller value.
4. Add relative step delta to the trajectory records and phase classifier.
5. Future checkpoint comparisons should use more samples and repeated probe seeds.

## Reproducibility

- Plan: `/data1/luohaoming/model_feature/plan/pythia_epsilon_sensitivity_plan.md`
- Configs: `/data1/luohaoming/model_feature/configs/pythia_epsilon_*.yaml`
- Analysis: `/data1/luohaoming/model_feature/scripts/analyze_epsilon_sensitivity.py`
- Raw data: `/home/luohaoming/model_feature_experiments/pythia_epsilon_sensitivity/eps_*/raw`
- Processed summaries: `/home/luohaoming/model_feature_experiments/pythia_epsilon_sensitivity/processed`
- Logs: `/home/luohaoming/model_feature_experiments/pythia_epsilon_sensitivity/logs`

## Next Direction

Implement and test relative convergence metrics, then run a denser residual-alpha sweep. After calibration, download selected Pythia training checkpoints and test whether training moves the native operator toward or away from the calibrated identity boundary.
