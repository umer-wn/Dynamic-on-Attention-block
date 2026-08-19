# Core-Paper Alignment Audit for an LLM Version of arXiv:1909.05176

## Verdict

The current project is methodologically adjacent to the core paper, but the completed experiments do not yet constitute a test of its central optimality claim for LLMs.

The paper defines a nonlinear equal-dimensional operator, iterates it as `x[t+1]=f(x[t])`, evaluates asymptotic stability near its attractor, and argues that increasing model quality/training pushes neural networks toward the boundary between (pseudo)periodic and chaotic regimes. The boundary and performance relationship are both essential.

## Alignment Matrix

| Paper requirement | Current project | Status |
|---|---|---|
| Equal-dimensional nonlinear operator | `inputs_embeds -> final_hidden` | implemented, but constructed/non-unique |
| Repeated feedback trajectory | final hidden state fed back as next input embeddings | implemented |
| Burn-in and asymptotic evaluation | configurable burn-in/eval windows | implemented, convergence criterion needs relative normalization |
| Asymptotic Jacobian criterion | normalized Frobenius and JVP-product probes | implemented and control-calibrated |
| Periodic/chaotic boundary evidence | lag distances, perturbations, projections | incomplete; no demonstrated chaotic side or bifurcation |
| Performance/accuracy axis | mostly final checkpoints and cross-model comparisons | missing for causal test |
| Training moves model toward edge | no same-architecture checkpoint series measured | missing |
| Optimality near edge | no controlled relation between loss/capability and edge distance | missing |

## What Existing Results Establish

1. The estimator correctly recovers the exact identity control: normalized Frobenius and multi-step gain equal one.
2. The native Pythia-70M `main` embedding-feedback operator is locally contractive under the tested protocol, with normalized Frobenius around 0.46 for direct feedback.
3. A half-residual operator is also contractive, with normalized Frobenius around 0.729 and negative tangent-product log gain.
4. Automatic-differentiation metrics are invariant to finite perturbation epsilon; nearby-trajectory signs at roughly `1e-4` per step are numerically unreliable in float32.

These findings characterize a particular constructed operator and validate instrumentation. They do not establish that Pythia, LLMs generally, or native autoregressive generation are subcritical in the paper's sense.

## What Existing Results Cannot Prove or Falsify

- They cannot prove the paper's LLM analogue because no improvement/performance axis has been tested.
- They cannot falsify it because one final checkpoint can be contractive while earlier checkpoints are farther from the boundary, or because a different justified LLM operator may exhibit the relevant transition.
- Cross-model comparisons cannot substitute for checkpoints because architecture, normalization, depth, tokenizer, dimension, and training data are confounded.
- Output scaling and residual interpolation are calibration interventions, not evidence about natural model training.
- A local Frobenius value near one alone is not a demonstrated Neimark-Sacker boundary; trajectory recurrence and tangent-product behavior must agree.

## Falsifiable LLM Hypotheses

### H1: Training-to-edge hypothesis

For a fixed Pythia architecture and fixed inputs, edge distance decreases as checkpoint loss/perplexity improves:

`abs(log(normalized_frobenius))` decreases with training step, while finite-time tangent log gain approaches zero from either side.

Evidence against H1: no monotonic/rank correlation across checkpoints, or later/better checkpoints consistently move farther from the boundary with uncertainty excluding zero.

### H2: Boundary/optimality hypothesis

The best validation loss or downstream performance occurs near a transition where tangent-product growth is approximately zero and bounded trajectories change from settled/periodic-like to sensitive/chaotic-like.

Evidence against H2: performance improves monotonically without approaching a transition, or the closest-to-edge checkpoints are not among the best-performing checkpoints.

### H3: Operator-robustness hypothesis

The training trend should be qualitatively consistent across at least two justified equal-dimensional LLM operators, such as whole-model `inputs_embeds -> final_hidden` and a clearly specified residual-stream/block operator.

Evidence against H3: the sign of the training trend reverses under small, defensible operator-definition changes.

## Required Next Experiment

Measure a same-architecture Pythia-70M checkpoint series (`step0`, early, middle, late, final) on identical samples. Record validation loss/perplexity, normalized Frobenius, JVP-product log gain, relative convergence, recurrence diagnostics, and uncertainty. This is the first experiment capable of supporting or challenging the paper's central training-to-edge claim in an LLM setting.

All data must be stored under `/home/luohaoming/model_feature_experiments`; every experiment must have a plan and report in the repository.
