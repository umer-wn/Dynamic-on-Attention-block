# Pythia Norm-Matched Probe Findings

## Run

- Date: 2026-07-10
- Config: `experiments/pythia_dynamical_edge_norm_matched_probe/configs/pythia_dynamical_edge_norm_matched_probe.yaml`
- Plan: `experiments/pythia_dynamical_edge_norm_matched_probe/plan/pythia_dynamical_edge_norm_matched_probe_plan.md`
- Baseline: `pythia_dynamical_edge_normal`

## Result

- Mean normalized Frobenius: `0.046596`
- Median normalized Frobenius: `0.046500`
- Collapsed fraction: `0.5`
- Mean actual eval states: `32.0`
- Settled fraction among recorded trajectories: `1.0`
- Max final step delta among recorded trajectories: `0.000001`

## Interpretation

Norm matching did not recover the paper's edge-of-chaos criterion. It made the operator more degenerate: two of four samples collapsed before producing evaluation states, and the remaining samples had a much smaller normalized Frobenius norm than the direct baseline.

This argues against a simple "scale drift only" explanation for the failed reproduction. The mismatch is more structural: the language-model feedback operator is not just the wrong norm, but a different map with different geometry from the paper's explicitly input-output matched systems.
