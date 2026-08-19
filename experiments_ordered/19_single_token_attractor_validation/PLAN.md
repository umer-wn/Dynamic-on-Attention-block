# Experiment 19 · Single-token attractor validation — recovery plan

## Status checked on 2026-08-11

- The formal worker is no longer running. `formal_all.pid` records PID `3471244`; no matching process remains.
- `screen` and `validate` completed for their configured checkpoints. `screen_summary.csv`, `orbit_candidates.csv`, `floquet_metrics.csv`, and `precision_audit.csv` exist.
- The run stopped during the final `controls` stage, while evaluating `step61000__layer_shuffle_2`.
- The terminal exception is deterministic and local to tangent/JVP computation: `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`.

## Recovery objective

Finish missing control conditions without recomputing completed screen/validation rows, then regenerate one coherent final report with explicit precision and completion flags.

## Plan

1. **Audit completed artifacts.** Build a condition manifest from the YAML configuration and compare it to completed rows in each processed CSV. Mark every existing row immutable; record missing conditions separately.
2. **Fix the JVP dtype boundary.** For every tangent/JVP path, enforce one dtype for the state, tangent, and GPT-NeoX linear weights. Preferred implementation: load the small Pythia-70M control model in `float32` for Lyapunov/Floquet/JVP stages, and keep model output/state in `float32`. A smoke test must cover the formerly failing `step61000__layer_shuffle_2` condition before a formal resume.
3. **Resume only missing controls.** Reuse the exact token manifest, perturbation seeds, projections, and checkpoint cache. Write each completed condition atomically to a resumable part file; do not overwrite completed screen/validate results.
4. **Post-run validation.** Check expected row counts, finite values, unique `(checkpoint, condition, token)` keys, projection-independence flags, and the FP32/FP64 precision audit. Failed precision consistency must remain visible rather than being silently filtered.
5. **Report.** Produce a table separating: completed/failed/missing conditions, orbit/period evidence, Floquet multipliers, Lyapunov estimates, perturbation response, and control ablations. State that a local orbit candidate is not a validated attractor unless recurrence and precision checks both pass.

## Acceptance criteria

- No JVP dtype exceptions in the full control manifest.
- All planned controls are either complete with finite metrics or explicitly recorded as failed/skipped with a cause.
- The final report links every conclusion to a row in the processed artifacts and preserves the existing precision-audit failures.

## Dense candidate-period extension (completed 2026-08-17)

This extension remains inside Experiment 19 and uses an optimized checkpoint-screening protocol:

1. Reuse the existing candidate periods for `clones / motive / cabinet / miles` and retain checkpoints whose median period is greater than one; retain the original step10000 case as well.
2. Select the lowest-period-error nontrivial token per checkpoint. The final selection contains 21 checkpoints; step27000 is excluded because only one of four tokens has a nontrivial period.
3. Use the existing 1024-step representative trajectory and its final 512 states to initialize multiple shooting. Preserve the original Floquet implementation and strict normalized p95 closure threshold `<=1e-5`.
4. Perturb the dynamic-step-768 state along 8 reproducible random full-512D unit directions at relative scales `1e-6,1e-4,1e-2`; evolve perturbed and unperturbed states for exactly 256 steps.
5. Record all 24 endpoint gains per checkpoint. Use arithmetic mean gain `<1` / `>1` as the headline contraction/amplification label, and retain geometric mean and contraction fraction as companion statistics.
6. Keep this reduced one-token protocol separate from the original full three-bank/eight-token recovery experiment; use it to prioritize full follow-up rather than to claim a general attractor.

Acceptance status:

- 21/21 checkpoint summaries complete;
- 504/504 perturbation trials complete;
- strict closure: step22000, step44000, step48000;
- step48000 is the only strict-closure candidate with both Floquet `stable` and arithmetic mean 256-step gain below one;
- conclusions and limitations integrated into `REPORT_EN.md` and `REPORT_ZH.md`.
