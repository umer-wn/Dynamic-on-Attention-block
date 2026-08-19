# Pythia Dynamical Edge Normal-Dimension Plan

## Objective

Validate the paper-style feedback-dynamics experiment on the currently downloaded natural language model using normal sequence dimensionality rather than the earlier smoke/small settings.

## Method

- Model: `EleutherAI/pythia-70m`, revision `main`, loaded from the local Hugging Face cache in offline mode.
- State: input embedding tensor, masked to non-padding tokens.
- Operator: feed the current hidden-state tensor as `inputs_embeds`, then take the final hidden state as the next iterate.
- Iteration: burn in for 16 feedback steps, then record 64 evaluation steps.
- Return map coordinate: fixed random projection per sample, held constant for every step in that sample trajectory.
- Metrics: normalized Jacobian Frobenius norm, fixed-projection return map, approximate Poincare section, nearby trajectory distance, lagged state-space distance, and multi-step Jacobian product gain.

## Scale

This run uses `sequence_length=128`, so the active hidden-state space is near the normal tested dimension for this project (`128 * hidden_size`, reduced only by padding mask). The sample count is set to 8 to keep the first normal-dimension validation tractable while still avoiding the earlier smoke/small configuration.

## Expected Outputs

- Raw rows in `results/raw/`.
- Processed tables in `results/processed/`.
- Figures plus per-figure explanations in `results/figures/`.
- Final findings in `reports/pythia_dynamical_edge_normal_findings.md`.

## Resource Notes

This is inference-only, not training. It uses one GPU, local cached model weights, and no large downloads. The multi-step Jacobian product is intentionally limited to windows `[2, 4]` with one probe for the first normal-dimension run.
