# Pythia Embedding-Expectation Temperature Sweep Plan

## Objective

Check whether the embedding-expectation operator's distance from the paper's edge criterion is caused by the scale of the LM-head softmax distribution.

## Method

Run the same smoke protocol as `pythia_embedding_expectation_smoke`, changing only `logit_temperature`:

- `T=0.5`
- `T=2.0`

## Interpretation

If a temperature moves normalized Frobenius close to 1 while preserving bounded trajectories, that temperature can define the next normal-dimension attempt. If both remain far from 1 or become degenerate, the next scheme should train an explicit embedding-space reconstruction/denoising map rather than relying on pretrained logits alone.
