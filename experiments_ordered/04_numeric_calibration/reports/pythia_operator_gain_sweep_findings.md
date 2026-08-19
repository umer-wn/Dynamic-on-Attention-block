# Pythia Operator-Gain Calibration Findings

## Result

Four cached WikiText samples were evaluated for each controlled output gain on GPUs 5--7.

| beta | normalized Frobenius mean | nearby log growth/step | product log gain, window 2 | product log gain, window 4 | phase |
|---:|---:|---:|---:|---:|---|
| 1.0 | 0.464414 | -0.001914 | -0.736548 | -0.676210 | 4/4 bounded nonfixed |
| 2.0 | 0.504756 | -0.005521 | -0.656798 | -0.600310 | 4/4 bounded nonfixed |
| 2.5 | 0.512807 | -0.006681 | -0.641433 | -0.585570 | 4/4 bounded nonfixed |

No sample diverged or collapsed. The local and product-Jacobian measurements changed monotonically in the expected direction, but the nearby-trajectory estimate became slightly more negative and no metric approached a phase transition.

## Interpretation

The experiment is a useful negative calibration result. Multiplying the output by beta is not a clean gain control for this normalized Transformer operator. The asymptotic trajectory changes with beta, and LayerNorm-like scale regulation can suppress a direct proportional response. Therefore the absence of a crossing does not establish that the measurement pipeline cannot detect criticality.

The calibration criterion was not fully met: two tangent/Jacobian diagnostics moved monotonically, but perturbation growth did not agree and no transition occurred. Scientific claims based on the absolute threshold should remain provisional.

## Method Problems Exposed

1. The paper's criterion applies to a specified iterated operator; the LLM embedding feedback operator is constructed rather than native autoregressive generation.
2. A local normalized Frobenius norm below one does not by itself establish asymptotic stability for a non-normal, time-varying Jacobian product.
3. Nearby perturbations at epsilon `1e-5` can be affected by floating-point resolution and need an epsilon sweep.
4. Four samples are enough for debugging, not inference.
5. The current phase classifier treats small absolute step deltas as fixed-like without normalizing by state norm.
6. Poincare and low-dimensional projections are qualitative and projection-dependent.

## Next Experiment

Use `F_alpha(x)=(1-alpha)x+alpha f(x)` as a positive control. At `alpha=0`, the operator is exactly identity, so normalized Frobenius and tangent-product gain must equal one. Sweep alpha away from zero, and separately sweep perturbation epsilon. Only after these controls pass should checkpoint downloads and training-progress claims be pursued.
