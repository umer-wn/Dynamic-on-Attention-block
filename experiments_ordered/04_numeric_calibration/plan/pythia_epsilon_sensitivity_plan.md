# Pythia Perturbation-Epsilon Sensitivity Plan

## Question

Are nearby-trajectory stability estimates numerically reliable in float32, or are current values dominated by perturbation scale and rounding?

## Design

- Operator: Pythia-70M residual feedback with `alpha=0.5`.
- Four identical WikiText samples, sequence length 64, fixed seed.
- Epsilon values: `1e-3`, `1e-5`, `1e-7` on GPUs 5, 6, 7.
- Burn-in/evaluation: 64/64.
- The Jacobian/Frobenius settings remain identical, so only finite perturbation behavior should change.

## Storage

Each run writes to `/home/luohaoming/model_feature_experiments/pythia_epsilon_sensitivity/eps_<value>`. Logs are stored beside results under `logs/`.

## Decision Rules

1. Local Frobenius and JVP-product estimates should be invariant to epsilon.
2. Nearby log-growth estimates should agree across epsilon within sampling/numerical error.
3. If `1e-7` collapses to zero, oscillates, or disagrees strongly while Jacobian metrics remain stable, float32 resolution is the cause and `1e-5` results require caution.
4. No scientific criticality claim will be made from this calibration run.

## Resources

One cached Pythia-70M inference/JVP task per GPU; expected memory below 10GB per GPU and runtime under two minutes per task. No downloads or training.
