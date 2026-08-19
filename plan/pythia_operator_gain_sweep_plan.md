# Pythia Operator-Gain Calibration Plan

## Question

Does the current embedding-space feedback pipeline detect a controlled transition when the operator is changed from `f(x)` to `beta * f(x)`?

This is a positive-control experiment. It does not claim that `beta != 1` is a natural language model. It tests whether normalized Frobenius, nearby-trajectory growth, Jacobian-product gain, and phase diagnostics respond coherently to a known gain intervention.

## Hypotheses

- At `beta=1.0`, reproduce the previously observed contractive regime.
- Increasing beta should generally increase local Jacobian gain, but not necessarily linearly because the asymptotic trajectory changes.
- A credible transition requires agreement among normalized Frobenius, multi-step tangent growth, nearby-trajectory growth, and boundedness/recurrence diagnostics. Frobenius crossing 1 alone is insufficient.

## Design

- Model: `EleutherAI/pythia-70m`, cached `main` revision.
- Dataset: four fixed WikiText validation samples.
- Sequence length: 64.
- Gains: `1.0`, `2.0`, `2.5`.
- Per gain: burn-in 64, evaluation 64, four Frobenius states with four probes, Jacobian-product windows 2 and 4, fixed-random trajectory projection, lag windows 1--32.
- GPUs: 5, 6, and 7, one gain per GPU.

## Decision Rules

1. Pipeline calibration passes only if at least two independent stability diagnostics change monotonically with beta.
2. Any apparent edge near normalized Frobenius 1 must also be bounded and show near-zero finite-time tangent/perturbation growth.
3. Divergence, norm collapse, floating-point saturation, or failure to settle invalidates direct comparison at that beta.
4. Results are exploratory (`n=4`) and must be replicated with more samples before scientific claims.

## Reproducibility

Commands, stdout/stderr, PIDs, GPU assignment, configs, raw JSONL, processed tables, and a findings report are retained under the project.
