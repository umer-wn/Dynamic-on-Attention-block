# Experiment Plans

This folder records experiment plans before and after execution.

Each plan should state:

- Why the experiment is needed.
- What hypothesis it tests.
- What data/model/config it uses.
- What outputs should be produced.
- What result would count as success, failure, or an unclear outcome.
- What should happen next.

Large model weights and datasets remain under:

```text
/public/luohaoming/model_feature/hf_cache
```

Experiment outputs remain under:

```text
results/
```

Current plans:

- `pythia_early_training_frobenius_scan_plan.md`: scans the first 100 regular Pythia checkpoints to test whether exact single-token normalized Frobenius tracks held-out loss or training step, with adaptive real-checkpoint densification around any loss reversal.
- `paper_method_refactor_plan.md`: refactor plan for paper-faithful dynamical operator experiments.
- `poincare_small_dynamics_metrics_plan.md`: next-stage small-scale plan for Poincare plots, Jacobian product metrics, and multi-step state distances.
- `generation_aligned_rolling_followup_plan.md`: streamlined engineering-validity gate comparing native growing-prefix, recency-only recomputation, and sink-preserving fixed-memory rolling before tangent experiments.
- `rolling_token_block_jacobian_followup_plan.md`: deferred per-context-token Jacobian attribution, executed only after the rolling engineering gate passes.
