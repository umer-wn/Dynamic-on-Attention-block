# Experiment 19 Complete Report (Stages 1–4)

**Generated:** 2026-08-17T13:56:14+08:00  
**Scope:** long-trajectory recurrence screening, multiple shooting, Floquet/monodromy stability, and finite-perturbation recovery. Precision audits and architecture ablations are outside the decision scope of this report.

## Executive result

- Screening covered **456 systems** (19 checkpoints × 3 initial-state banks × 8 tokens): recurrent=141, expanding=73, quasiperiodic=8, transient/unresolved=234, fixed=0.
- All eight selected checkpoints completed their long validation traces. step10000, step29000, step41000, step57000 produced shootable recurrent candidates; step2000, step9000, step13000, step61000 produced none, so their absence from the orbit table is not missing execution.
- There are **288 phase-level shooting records**, representing **96 unique checkpoint–state-bank–token systems** after choosing the lowest-residual phase. Phase-level stability: stable=51, boundary=191, unstable=46. System-level stability: stable=16, boundary=66, unstable=14.
- **288/288** shooting records converged and all residuals are at most `1e-5`, but only **0/288** records achieve perturbation recovery ≥0.9.
- Every minimal period is greater than one (`p=1`: 0), so no fixed point was detected. No candidate jointly satisfies shooting convergence, stable Floquet label, and recovery ≥0.9. Therefore, through Stage 4, **no stable attracting periodic orbit has been validated under the finite-perturbation criterion**.

## Protocol and criteria

1. Recurrence screening iterates each isolated-token state for 4096 steps and analyzes the final 2048; strict validation uses 16384 steps and the final 8192.
2. Multiple shooting optimizes all orbit nodes simultaneously. A normalized p95 closure residual ≤`1e-5` passes geometric closure.
3. Floquet stability estimates the spectral radius of the one-period Jacobian product with Arnoldi dimensions 16 and 32: stable if `rho<0.98` and relative disagreement <0.05; unstable if `rho>1.02`; otherwise boundary.
4. Recovery uses 16 random full-512D unit directions at three relative scales (`1e-6,1e-4,1e-2`) for ten periods. Recovery means final phase-invariant distance to the orbit does not exceed the initial distance; ≥0.9 is the attraction threshold.
5. Minimal period 1 denotes a fixed point; a value greater than 1 denotes a periodic orbit.

## Per-checkpoint results (phase level)

| checkpoint | records | stable | boundary | unstable | minimal periods | median residual | median rho(M) | max recovery |
|---|---|---|---|---|---|---|---|---|
| step2000 | 0 | 0 | 0 | 0 | — | — | — | — |
| step9000 | 0 | 0 | 0 | 0 | — | — | — | — |
| step10000 | 72 | 0 | 72 | 0 | p101:72 | 6.359e-07 | 0.993976 | 0.000000 |
| step13000 | 0 | 0 | 0 | 0 | — | — | — | — |
| step29000 | 72 | 0 | 72 | 0 | p100:72 | 4.422e-07 | 1.007933 | 0.000000 |
| step41000 | 72 | 12 | 29 | 31 | p51:9, p57:63 | 1.697e-07 | 1.010953 | 0.000000 |
| step57000 | 72 | 39 | 18 | 15 | p203:72 | 1.617e-07 | 0.974342 | 0.541667 |
| step61000 | 0 | 0 | 0 | 0 | — | — | — | — |

## Per-checkpoint results (deduplicated systems)

| checkpoint | systems | stable | boundary | unstable | minimal periods | median residual | median rho(M) | max recovery |
|---|---|---|---|---|---|---|---|---|
| step2000 | 0 | 0 | 0 | 0 | — | — | — | — |
| step9000 | 0 | 0 | 0 | 0 | — | — | — | — |
| step10000 | 24 | 0 | 24 | 0 | p101:24 | 6.306e-07 | 0.992634 | 0.000000 |
| step13000 | 0 | 0 | 0 | 0 | — | — | — | — |
| step29000 | 24 | 0 | 24 | 0 | p100:24 | 4.388e-07 | 1.007756 | 0.000000 |
| step41000 | 24 | 4 | 11 | 9 | p51:3, p57:21 | 1.672e-07 | 1.004639 | 0.000000 |
| step57000 | 24 | 12 | 7 | 5 | p203:24 | 1.592e-07 | 0.983231 | 0.375000 |
| step61000 | 0 | 0 | 0 | 0 | — | — | — | — |

## Numerical summary

- Shooting residual: min=1.511e-07, median=4.353e-07, p95=6.515e-07, max=6.806e-07.
- Recovery fraction: min=0.000000, median=0.000000, p95=0.333333, max=0.541667.
- Phase-level minimal-period distribution: {51: 9, 57: 63, 100: 72, 101: 72, 203: 72}.
- System-level minimal-period distribution: {51: 3, 57: 21, 100: 24, 101: 24, 203: 24}.

## Interpretation of stable / unstable / boundary

- `stable` means locally contracting under the estimated one-period linearization; it is not sufficient by itself to establish an attractor.
- `unstable` means an estimated Floquet multiplier modulus is clearly above one.
- `boundary` means the multiplier lies near the unit circle or the Krylov estimates do not satisfy the stable rule.
- Since none of the locally stable candidates passes the ≥0.9 recovery threshold, the defensible conclusion is: geometrically closed periodic candidates exist, some have locally stable labels, but no stable attractor is validated.

## Data products

- `processed/screen_summary.csv`: 456 screened systems.
- `processed/orbit_candidates.csv`: 288 phase-level shooting, Floquet, and recovery results.
- `processed/stage4_system_summary.csv`: 96 deduplicated systems generated with this report.

## Dense checkpoint extension (completed 2026-08-17)

This reduced-cost extension reuses the existing four-token period screen. A checkpoint is included when the median candidate period across `clones / motive / cabinet / miles` is greater than one; step10000 from the original Experiment 19 screen is also retained. One lowest-error nontrivial-period token represents each checkpoint. This selects 21 checkpoints and excludes step27000, whose nontrivial period occurs for only one of four tokens.

The extension is a screening protocol, not a replacement for the full 4096-step, three-state-bank, eight-token experiment. It uses an existing 1024-step trajectory, its final 512 states for orbit initialization, the original multiple-shooting/Floquet implementation, and the original strict report closure threshold of normalized p95 residual `<=1e-5`.

Perturbation response is measured from dynamic step 768 to 1024 with 8 reproducible random full-512D directions and 3 relative scales (`1e-6,1e-4,1e-2`), giving 24 trials per checkpoint and 504 rows total. For trial `i`,

\[
g_i=\frac{\|\widetilde x_{1024}^{(i)}-x_{1024}\|_2}
{\|\widetilde x_{768}^{(i)}-x_{768}\|_2}.
\]

The headline response is the arithmetic mean of the 24 gains: `<1` means average contraction and `>1` means average amplification. Geometric mean, median, extrema, mean log gain, contraction fraction, and per-scale summaries are retained as companion statistics. This same-time reference metric follows trajectory drift but is not phase-invariant and is not equivalent to the original ten-period recovery fraction.

Strict shooting closure passes at step22000, step44000, and step48000. Their joint results are:

| checkpoint | period | normalized p95 residual | Floquet | rho(M) | 256-step arithmetic / geometric gain | contracting directions | interpretation |
|---|---:|---:|---|---:|---:|---:|---|
| step22000 | 200 | 9.45e-7 | boundary | 0.9981 | 1.696 / 1.180 | 0.38 | closed, near-neutral linearization, average amplification |
| step44000 | 173 | 9.21e-7 | boundary | 0.9988 | 0.468 / 0.211 | 0.88 | closed, near-neutral linearization, average contraction |
| step48000 | 197 | 2.69e-7 | stable | 0.9676 | 0.863 / 0.348 | 0.67 | closed, locally stable, average contraction |

step45000 and step57000 meet the optimizer's looser internal convergence tolerance but fail the strict `1e-5` report threshold, so their Floquet estimates are diagnostic only. In particular, step57000 has a diagnostic stable multiplier but an arithmetic mean gain of 1.933, demonstrating why local Floquet evidence and finite-scale response must remain separate.

The reduced extension therefore identifies **step48000 as the strongest follow-up stable-periodic-orbit candidate**. It is the only representative system that jointly passes strict closure, Floquet stability, and mean 256-step contraction. Because only one token and one state source were tested, this is not yet a general validation of an attracting periodic orbit.

Additional data products:

- `processed/dense_periodic_checkpoint_selection.csv`: 21-checkpoint selection audit.
- `processed/dense_periodic_checkpoint_summary.csv`: checkpoint-level closure, Floquet, and response summary.
- `processed/dense_periodic_checkpoint_perturbation_256.csv`: all 504 direction/scale trials.
- `scripts/run_dense_periodic_extension.py`: reproducible extension runner.
