# Pythia Training-Checkpoint Criticality Plan

## Core Hypothesis

Test the LLM analogue of arXiv:1909.05176: as a language model becomes better during pretraining, does the paper-aligned stability metric of a same-dimensional neural operator move toward the edge of chaos?

This experiment uses the constructed operator `inputs_embeds -> final_hidden -> inputs_embeds`. It tests that operator only, not native autoregressive generation.

## Checkpoints and Performance

- Model: `EleutherAI/pythia-70m`.
- Revisions: `step0`, `step1000`, `step16000`, `step143000`.
- Performance: token-weighted validation NLL/perplexity on 128 fixed nonempty WikiText validation texts, sequence length 64.
- Dynamics: eight fixed texts, sequence length 64, burn-in 64, evaluation 128, four Lyapunov probes, four asymptotic states x four Frobenius probes.

## Primary Metrics

- Benettin maximal finite-time Lyapunov exponent.
- Normalized asymptotic Frobenius Jacobian norm.
- Final/initial nearby-trajectory separation.
- Relative step convergence and phase label.
- Token-weighted validation loss/perplexity.

## Hypotheses

- H1: training improves token-weighted validation loss across the selected revisions.
- H2: training moves the constructed operator toward the boundary: `|Lyapunov|` decreases toward zero and/or normalized Frobenius moves toward one.
- H3: better validation loss correlates with smaller distance to the edge.

H1 is a validity check. H2/H3 are the paper-derived LLM hypotheses. A monotonic movement away from zero/one as performance improves is evidence against H2 for this constructed operator, but does not falsify the paper for other possible LLM operators.

## Decision Rules

1. Do not interpret dynamics if checkpoint identity, sample indices, token counts, or loading differ.
2. Require consistent signs across most samples and report dispersion.
3. Four checkpoints and eight dynamics samples are exploratory; correlations are descriptive, not inferential.
4. If a clear trend exists, confirm with 100 samples and 500 iterations before a strong claim.

## Storage and Compute

- Model/cache data: `/home/luohaoming/model_feature_cache/hf_cache`.
- Experiment data/logs: `/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality`.
- Code/config/plan/report: `/data1/luohaoming/model_feature`.
- GPUs 5, 6, 7; expected under 10GB per task, no training.
