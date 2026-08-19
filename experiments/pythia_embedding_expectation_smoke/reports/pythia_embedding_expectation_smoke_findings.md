# Pythia Embedding-Expectation Smoke Findings

## Run

- Date: 2026-07-11
- Config: `experiments/pythia_embedding_expectation_smoke/configs/pythia_embedding_expectation_smoke.yaml`
- Plan: `experiments/pythia_embedding_expectation_smoke/plan/pythia_embedding_expectation_smoke_plan.md`
- Model: `EleutherAI/pythia-70m`
- Operator: `inputs_embeds -> logits -> softmax -> expected input embedding`
- Temperature: `1.0`

## Result

- Mean normalized Frobenius: `0.244026`
- Median normalized Frobenius: `0.244026`
- Diverged fraction: `0.0`
- Collapsed fraction: `0.0`
- Settled fraction: `0.0`
- Max final step delta: `1.296017`

## Interpretation

This operator is structurally better aligned with the paper than `inputs_embeds -> final_hidden` because input and output are both embedding-space tensors. However, the smoke result is still far below the paper's edge criterion and does not settle under the short smoke trajectory.

The operator is viable for experimentation, but temperature and/or a trained same-space objective are needed before it can be considered a serious LLM reproduction attempt.
