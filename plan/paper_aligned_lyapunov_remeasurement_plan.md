# Paper-Aligned Lyapunov Remeasurement Plan

## Goal

Replace the incorrectly interpreted random-direction Jacobian-product summary with a maximal finite-time Lyapunov estimate and record the paper-style final separation of nearby initial states.

## Implementation

1. Add relative step delta to trajectory records.
2. Preserve the initial perturbation magnitude and record final asymptotic distance and final/initial separation.
3. Add Benettin-style tangent propagation: JVP at every asymptotic state, normalize tangent after every step, accumulate log growth, and repeat across probe seeds.
4. Keep the old product metric for backward compatibility but label it as random-direction product gain, not maximal Lyapunov.
5. Unit-test identity (`lambda=0`), contraction (`log 0.5`), and expansion (`log 2`).

## Experiments

- Pythia-70M, same four samples and sequence length 64 for controlled comparison.
- Operators: residual alpha 0, 0.5, and native alpha 1.
- Burn-in/eval: 64/128 for alpha 0.5 and 1; shorter identity control is sufficient.
- Four Lyapunov probes; Frobenius four states x four probes.
- GPUs 5, 6, 7.

## Decision Rules

- Identity must yield Lyapunov 0, final/initial separation 1, and Frobenius 1.
- Linear unit tests must match analytical exponents.
- Native-operator phase statements require agreement between Lyapunov sign, asymptotic separation, boundedness, and recurrence diagnostics.
- This debug run can correct methodology but cannot test optimal intelligence because it lacks training checkpoints and a performance curve.

## Storage

All data and logs: `/home/luohaoming/model_feature_experiments/paper_aligned_lyapunov_remeasurement`.
