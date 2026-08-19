# Qwen Dynamical Edge Burn256 Findings

## Run

- Date: 2026-07-10
- Config: `experiments/qwen_dynamical_edge_burn256/configs/qwen_dynamical_edge_burn256.yaml`
- Plan: `experiments/qwen_dynamical_edge_burn256/plan/qwen_dynamical_edge_burn256_plan.md`
- Results: `experiments/qwen_dynamical_edge_burn256/results/`
- Model: `Qwen/Qwen2.5-0.5B`, revision `main`, offline cache
- Scale: `sequence_length=128`, `num_samples=4`, `burn_in_steps=256`, `eval_steps=128`

## Main Results

- Mean normalized Jacobian Frobenius norm: `0.470673`.
- Median normalized Jacobian Frobenius norm: `0.468591`.
- Mean edge distance log: `0.753882`.
- Settled samples: `2/4`.
- Final step delta values: approximately `0.000079`, `0.000111`, `35.415787`, `94.978081`.
- Nearby perturbation average growth ratio: `2.215181`.
- Nearby perturbation average log growth per step: `0.002936`.

## Interpretation

Increasing burn-in to 256 did not make all Qwen samples settle. Two samples converge to very small step deltas, while two remain in large-amplitude motion.

This supports keeping the Qwen row marked as not fully asymptotic in the paper-alignment matrix. The Qwen normalized Frobenius values remain below 1, but the run does not satisfy the paper's clean attractor-evaluation assumption for all samples.

Product-Jacobian probes were disabled in this check, so the main matrix should keep the `qwen_dynamical_edge_normal_long` row as the primary Qwen row because it includes the product-gain metric.
