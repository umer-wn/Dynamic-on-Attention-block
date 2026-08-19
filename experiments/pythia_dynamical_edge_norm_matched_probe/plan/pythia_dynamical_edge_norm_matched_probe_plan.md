# Pythia Norm-Matched Restoration Probe Plan

## Objective

Test whether the failed paper-style reproduction is mainly caused by scale drift in the language-model feedback operator `inputs_embeds -> final_hidden`.

## Hypothesis

The direct operator maps input embeddings into final hidden states whose scale and normalization geometry differ from the original embedding manifold. This may make the feedback map artificially contractive. Matching the output norm to the current state norm tests whether scale preservation moves the normalized Jacobian norm closer to the paper's edge criterion `||J*||/sqrt(N)=1`.

## Method

- Model: `EleutherAI/pythia-70m`, revision `main`, offline cache.
- Baseline to compare: `pythia_dynamical_edge_normal`.
- Intervention: `operator_update=norm_matched`.
- Scale: `sequence_length=128`, `num_samples=4`, `burn_in_steps=16`, `eval_steps=64`.
- Metrics: normalized Frobenius, nearby perturbation distance, lagged state distance, product-Jacobian gain, fixed-projection return map, and approximate Poincare section.

## Interpretation

If norm matching moves the normalized Frobenius estimate toward 1 without inducing unstable trajectories, the failure may be partly due to operator scale mismatch. If it remains far below 1 or creates non-paper-like artifacts, the main mismatch is likely structural rather than only scale-related.
