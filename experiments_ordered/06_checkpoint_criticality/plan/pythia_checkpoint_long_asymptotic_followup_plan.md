# Pythia Checkpoint Long-Asymptotic Follow-up Plan

## Trigger

The first checkpoint comparison cannot be interpreted asymptotically: `step0`, `step1000`, and `step16000` retain large relative step deltas after burn-in 64, while `step143000` is converged. Comparing their Lyapunov values would confound training with unequal distance to the attractor.

## Goal

Repeat all four checkpoints with a matched paper-inspired long protocol and determine whether each trajectory reaches an asymptotic regime before comparing criticality.

## Protocol

- Same eight WikiText samples and sequence length 64.
- Burn-in 512, evaluation 128.
- Two Benettin Lyapunov probes per sample; four final states x four Frobenius probes.
- Perturbation epsilon `1e-3`.
- Same token-weighted performance data from the parent experiment.
- GPUs 5, 6, 7; first three checkpoints in parallel, final checkpoint after a GPU is free.

## Gates

1. Primary asymptotic gate: tail relative step delta below `1e-6` or clear stationary/recurrence evidence.
2. Checkpoints failing the gate are reported as unresolved/non-asymptotic; their Lyapunov values cannot establish paper phase.
3. Only matched, asymptotically valid rows enter the performance-criticality trend.
4. If early checkpoints still fail, add recurrence/cycle diagnostics rather than indefinitely extending fixed-point burn-in.

## Storage

`/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality/long_asymptotic`.
