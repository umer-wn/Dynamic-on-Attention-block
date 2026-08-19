# Paper 1909.05176 Alignment Matrix

> **Correction, 2026-07-12:** The historical `product_log_gain` field is a mean random-direction product gain, not the dominant Jacobian-product eigenvalue or maximal Lyapunov exponent required by paper method 2. The historical nearby-growth field is a post-burn-in distance ratio, not paper method 3's final asymptotic separation from the initial perturbation. Therefore the matrix below is retained as historical evidence but cannot establish phase by itself. A corrected Benettin-style remeasurement for Pythia-70M is reported in `reports/paper_aligned_lyapunov_remeasurement_findings.md`; the full audit is in `reports/core_paper_alignment_audit_20260712.md`.

## Scope

Goal: check whether the current natural-language-model experiments can be interpreted as a reproduction of "Optimal Machine Intelligence at the Edge of Chaos" and build a model-level result matrix for the downloaded language models.

Paper: https://arxiv.org/abs/1909.05176

## Paper Requirements Used Here

The paper defines a generic discrete operator:

```text
x_{t+1} = f(x_t)
```

The edge-of-chaos criterion is based on the asymptotic Jacobian norm:

```text
||J*|| / sqrt(N) = 1
```

where `J*` is the Jacobian of the dynamical operator evaluated near the asymptotic attractor. The paper also describes two practical complements:

- the spectral radius of a product of time-local Jacobians over a later trajectory window;
- the final separation between nearby perturbed trajectories when direct Jacobian computation is too expensive.

Poincare maps are used as a phase check, not as a replacement for the Jacobian or perturbation criterion.

## Current Fit to the Paper

Current project implementation is partially aligned:

- It constructs an iterated language-model operator by feeding hidden/input-embedding states back through the model.
- It computes normalized Frobenius Jacobian estimates after burn-in.
- It records nearby perturbation trajectories.
- It computes short-window Jacobian-product gain for the normal-dimension experiments.
- It generates fixed-projection return maps and approximate Poincare plots.

Important caveats:

- The paper's image models explicitly reshape or modify architectures so input and output dimensions match. Here, the language-model operator is defined in embedding/hidden-state space via `inputs_embeds -> final_hidden`.
- The paper's stronger experiments use many samples and long iteration counts, e.g. hundreds of images and hundreds of iterations. Current normal-dimension dynamics evidence is still a tractable validation run.
- The existing cross-model spectrum tables are layer-local static Jacobian spectra; they are useful auxiliary diagnostics but are not the same as the paper's asymptotic full-operator Jacobian norm.

Additional reflection on why the reproduction failed and which restoration probes were attempted is in:

```text
reports/failed_reproduction_reflection_1909_05176.md
```

The adjusted LLM reproduction scheme based on those results is in:

```text
reports/llm_reproduction_adjusted_plan_1909_05176.md
```

## Current Result Matrix

Machine-readable matrix:

```text
results/processed/paper_alignment_matrix.csv
```

Current values:

| model | direct dynamics? | asymptotic status | settled fraction | seq | samples | normalized Frobenius mean | product log gain mean | nearby log growth mean | paper phase by direct criterion | auxiliary spectrum layers |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| EleutherAI/pythia-70m | yes | settled | 1.00 | 128 | 8 | 0.463930 | -0.704469 | -0.006536 | stable/contractive | 6 |
| gpt2 | yes | settled | 1.00 | 128 | 8 | 0.240475 | -0.764971 | -0.025544 | stable/contractive | 12 |
| Qwen/Qwen2.5-0.5B | yes | not settled | 0.50 | 128 | 4 | 0.469756 | -0.577596 | -0.017870 | provisional stable/contractive | 24 |

Pythia-70M and GPT-2 have direct settled dynamics rows and are below the paper's edge threshold, so the current operator appears stable/contractive rather than near the edge of chaos for those two models.

Qwen has a direct row, but it is not fully asymptotically settled. The primary Qwen matrix row uses the `qwen_dynamical_edge_normal_long` experiment (`burn_in_steps=64`, `eval_steps=128`) because it includes both normalized Frobenius and product-Jacobian metrics. Only 2 of 4 samples settle under the current `step_delta <= 1e-2` criterion. A separate `qwen_dynamical_edge_burn256` convergence check (`burn_in_steps=256`, `eval_steps=128`, product probes disabled) still settles only 2 of 4 samples, so the Qwen row should remain provisional rather than treated as a clean paper-comparable attractor estimate.

## Existing Auxiliary Matrix

The auxiliary static spectrum evidence is available for all three models at sequence length 128:

| model | layers | max layer sigma max | median layer sigma max | median top-k geometric mean |
| --- | ---: | ---: | ---: | ---: |
| EleutherAI/pythia-70m | 6 | 119.715266 | 26.133215 | 6.484550 |
| gpt2 | 12 | 94.058604 | 12.538501 | 5.320149 |
| Qwen/Qwen2.5-0.5B | 24 | 512.967955 | 10.551835 | 4.023360 |

This table should not be interpreted as the paper's `||J*|| / sqrt(N)` result because it comes from block/layer-level spectra rather than the iterated full operator at the attractor.

## Next Required Experiments

To make the matrix stronger across the downloaded language models:

1. Decide whether Qwen's unsettled samples should be analyzed as long-period/pseudo-periodic trajectories rather than forced into a fixed-point convergence criterion.
2. Increase sample counts for Pythia/GPT-2 settled rows after the protocol is stable.
3. Rebuild `results/processed/paper_alignment_matrix.csv`.
4. Compare direct normalized Frobenius, nearby perturbation growth, short Jacobian-product gain, and asymptotic status across all three rows.
5. Only settled rows should be used to claim cross-model support or contradiction for the paper-style conclusion.

Recommended follow-up protocol:

- `sequence_length=128`
- `num_samples=16+` for Pythia/GPT-2
- Qwen: either increase samples at `burn_in_steps=64`, `eval_steps=128`, or add explicit cycle diagnostics for the unsettled samples
- `frobenius_eval_states=4`
- `frobenius_probes=4`
- `trajectory_projection=fixed_random`
- `lag_distance_windows=[1,2,4,8,16,32]`
- `product_jacobian_windows=[2,4]`
- `product_jacobian_probes=1`

This preserves the paper's method while making the Qwen caveat explicit.
