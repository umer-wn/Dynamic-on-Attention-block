# Pythia Training-Checkpoint Criticality Findings

## Status

Complete, including the long-asymptotic follow-up triggered by failure of the initial burn-in protocol.

This experiment tests whether Pythia-70M pretraining moves the constructed same-dimensional operator `inputs_embeds -> final_hidden -> inputs_embeds` toward the edge of chaos described in arXiv:1909.05176.

## Protocol

- Checkpoints: `step0`, `step1000`, `step16000`, `step143000`.
- Performance: 128 fixed WikiText validation texts, sequence length 64, 5,733 predicted tokens per checkpoint, token-weighted NLL/perplexity.
- Dynamics: eight fixed texts, sequence length 64.
- Final matched protocol: burn-in 512, evaluation 128, two Benettin Lyapunov probes, four final states x four Frobenius probes.
- All model/data caches, raw results, processed tables, and logs are under `/home/luohaoming`.

## Main Results

| checkpoint | weighted loss | weighted PPL | normalized Frobenius | maximal Lyapunov | positive-Lyapunov samples | final/initial separation | tail relative step delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| step0 | 10.9877 | 59144.99 | 0.6429 | +0.01154 | 8/8 | 12343.3 | 8.06e-2 |
| step1000 | 5.7499 | 314.15 | 0.6559 | -0.00515 | 0/8 | 863.5 | 4.22e-2 |
| step16000 | 4.7425 | 114.73 | 0.4590 | -0.01810 | 0/8 | 0.270 | 1.43e-5 |
| step143000 | 4.6974 | 109.66 | 0.4640 | -0.26613 | 0/8 | 0.182 | 4.63e-8 |

## Phase Interpretation

### step0: chaotic-attractor candidate

- All eight samples have positive finite-time maximal Lyapunov estimates.
- Nearby trajectories amplify by about 12,343x on average.
- State-norm drift between the two evaluation halves is only `4.4e-8`, so the behavior is not explained by simple norm divergence.
- No recurrence is visible for lags up to 64; normalized lag distance approaches one state norm.

This is strong exploratory evidence for a stationary chaotic-like regime of the constructed operator. Formal attractor/stationarity tests and longer evaluation remain desirable.

### step1000: near-boundary but unresolved

The mean Lyapunov is slightly negative and all sample estimates are negative, but the original perturbation has already amplified strongly and the projected/step statistics still drift. This row is consistent with a transition region but is not cleanly classified.

### step16000: slowly converging stable regime

The Lyapunov estimate is negative, final perturbations contract, and normalized lag distances are only `4e-5--4e-4`. It has not crossed the strict `1e-6` fixed-point threshold after burn-in 512, but the evidence favors slow stable convergence or a very small cycle rather than chaos.

### step143000: stable fixed-like regime

All eight samples converge, Lyapunov is strongly negative, nearby perturbations contract, and normalized recurrence distances are near numerical precision.

## Hypothesis Decisions

### H1: training improves validation performance - supported

Token-weighted loss falls from `10.99` to `4.70`, with exactly 5,733 predicted tokens at every checkpoint.

### H2: training moves the constructed LLM operator toward the edge - contradicted by this experiment

The trajectory begins on the chaotic/positive-Lyapunov side, passes near zero early, then moves deeper into the negative-Lyapunov stable regime. The final and best-performing checkpoint is the most strongly contracting checkpoint, not the closest to zero.

### H3: better performance is associated with smaller distance to the edge - not supported

Performance improves while the late-training Lyapunov magnitude grows from roughly `0.005--0.018` to `0.266`. With only four checkpoints and an unresolved transition row, a formal correlation claim is inappropriate, but the observed direction is opposite to H3 for this operator.

## Critical Method Finding: Frobenius Proxy Fails Here

At `step0`, normalized Frobenius is `0.643 < 1`, which would suggest the ordered side under the paper's high-dimensional random-Jacobian approximation. Yet all samples have positive maximal Lyapunov values and perturbations grow by four orders of magnitude.

Therefore Transformer Jacobian structure/correlation invalidates an untested equivalence between normalized Frobenius and maximal Lyapunov for this operator. Future reports must use the Benettin/JVP Lyapunov result as the primary phase diagnostic and treat Frobenius as an auxiliary average-gain statistic.

## Why the Long Follow-up Was Necessary

With burn-in 64, early checkpoints had large relative step deltas and misleadingly unstable finite-window estimates. Increasing burn-in to 512 changed the `step0` mean Lyapunov from approximately zero to clearly positive and exposed continued perturbation amplification. The short protocol is retained as evidence that asymptotic checks are essential.

## Scope and Limitations

- This is a constructed embedding-feedback operator, not native autoregressive generation.
- Eight dynamics samples and four checkpoints are exploratory.
- The paper used around 100 samples and 500 iterations in its principal numerical protocols.
- `step0/1000` require stronger attractor stationarity diagnostics.
- The experiment contradicts the paper-derived hypothesis for this operator; it does not falsify arXiv:1909.05176 generally or rule out a different, better-motivated LLM operator.

## Reproducibility

- Parent plan: `/data1/luohaoming/model_feature/plan/pythia_checkpoint_criticality_plan.md`
- Long follow-up plan: `/data1/luohaoming/model_feature/plan/pythia_checkpoint_long_asymptotic_followup_plan.md`
- Configs: `/data1/luohaoming/model_feature/configs/pythia_checkpoint*.yaml`
- Analysis: `/data1/luohaoming/model_feature/scripts/analyze_checkpoint_criticality.py` and `scripts/analyze_checkpoint_long_asymptotic.py`
- Data/logs: `/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality`
- Model cache: `/home/luohaoming/model_feature_cache/hf_cache`

## Recommended Next Experiment

Map the transition more densely with Pythia checkpoints between step0 and step16000, particularly steps 100, 500, 1000, 2000, 4000, and 8000. In parallel, design a native autoregressive-generation dynamical metric so conclusions do not depend only on the artificial embedding-feedback operator.
