# Pythia Dynamical Edge Smoke Findings

## Goal

Start the project refactor toward the `arXiv:1909.05176` method using a paper-style feedback dynamical system on an already downloaded language model.

This experiment is not the old block Jacobian probe. It explicitly defines an equal-dimensional continuous operator:

```text
x_{t+1} = f(x_t)
```

For the LLM setting, `x_t` is the input embedding tensor and `f(x_t)` is the model's final hidden state tensor with the same shape:

```text
input:  inputs_embeds, shape [batch, seq_len, hidden_dim]
output: final_hidden, same shape
```

The measured paper-style metric is:

```text
normalized_frobenius = ||J_f(x_t)||_F / sqrt(N)
```

estimated with Hutchinson JVP probes after burn-in feedback steps.

## Implemented Pipeline

New code:

```text
src/dynamics.py
scripts/compute_dynamical_edge.py
scripts/analyze_dynamical_edge.py
tests/test_dynamics.py
```

New configs:

```text
configs/pythia_dynamical_edge_smoke.yaml
configs/pythia_dynamical_edge_small.yaml
```

Outputs:

```text
results/raw/pythia_dynamical_edge_smoke__EleutherAI__pythia-70m__main__seq32__dynamical_edge.jsonl
results/raw/pythia_dynamical_edge_small__EleutherAI__pythia-70m__main__seq64__dynamical_edge.jsonl
results/processed/pythia_dynamical_edge_small__dynamical_edge_rows.csv
results/processed/pythia_dynamical_edge_small__dynamical_edge_summary.csv
```

## Tests

Unit smoke:

```text
PYTHONPATH=/data1/luohaoming/model_feature \
/public/luohaoming/model_feature/.venv/bin/python \
-m unittest discover \
-s /data1/luohaoming/model_feature/tests \
-p 'test_*.py' \
-t /data1/luohaoming/model_feature
```

Result:

```text
Ran 2 tests
OK
```

The unit tests verify:

- identity operator has normalized Frobenius near `1`
- contracting operator has normalized Frobenius below `1`

This checks the core paper metric before using an LLM.

## Smoke Run

Config:

```text
configs/pythia_dynamical_edge_smoke.yaml
```

Setup:

- Model: `EleutherAI/pythia-70m`
- Dataset: `wikitext/wikitext-2-raw-v1`, validation split
- Samples: `2`
- Sequence length: `32`
- Burn-in steps: `2`
- Eval steps: `2`
- Frobenius eval states: `1`
- Hutchinson probes per state: `1`
- Operator update: `direct`
- Target: `final_hidden`

Result:

```text
wrote 2 rows
```

Both rows completed without divergence or collapse.

## Small-Batch Run

Config:

```text
configs/pythia_dynamical_edge_small.yaml
```

Setup:

- Model: `EleutherAI/pythia-70m`
- Dataset: `wikitext/wikitext-2-raw-v1`, validation split
- Samples: `4`
- Sequence length: `64`
- Burn-in steps: `4`
- Eval steps: `4`
- Frobenius eval states: `2`
- Hutchinson probes per state: `2`
- Operator update: `direct`
- Target: `final_hidden`

Summary:

| metric | value |
|---|---:|
| samples | 4 |
| mean normalized Frobenius | 0.4651 |
| median normalized Frobenius | 0.4649 |
| mean absolute log distance from 1 | 0.7654 |
| diverged fraction | 0.0 |
| collapsed fraction | 0.0 |

All four samples were labeled:

```text
bounded_nonfixed_like
```

## Interpretation

For this small Pythia-70M embedding-space feedback system, the estimated normalized Frobenius value is below the paper's nominal edge value of `1`.

Very cautious interpretation:

```text
observed small-batch metric: ||J_f||_F / sqrt(N) ~= 0.465
paper edge target:          ||J_f||_F / sqrt(N) ~= 1
```

This suggests the direct `inputs_embeds -> final_hidden -> inputs_embeds` feedback operator is locally contractive under the current small configuration.

This does not yet validate or refute the paper conclusion for language models. It only proves that the refactored pipeline now measures the paper-style feedback operator rather than the old one-step block Jacobian.

## Important Limitations

- The LLM operator is constructed in embedding space; natural language token ids are discrete and cannot be directly iterated by gradient-based Jacobian methods.
- `final_hidden` is fed back as `inputs_embeds`, which is a mathematically valid same-dimensional operator but not a standard autoregressive generation process.
- The run used only `4` samples, short sequences, `4` burn-in steps, and `2` Hutchinson probes per state.
- The phase labels are finite-time diagnostics and should not be overread as proof of true dynamical phase.
- The current result uses only Pythia-70M `main`; validating the paper-style conclusion requires checkpoint/model comparisons.

## Next Steps

1. Repeat the small-batch run with more Hutchinson probes, for example `8-16`.
2. Increase burn-in and eval steps to check whether the normalized Frobenius estimate stabilizes.
3. Run the same pipeline on cached `gpt2` and `Qwen/Qwen2.5-0.5B`.
4. Add alternative operators:
   - `residual`: `x_{t+1} = (1 - alpha) x_t + alpha f(x_t)`
   - `norm_matched`: rescale `f(x_t)` to preserve input norm
5. Compare model quality or checkpoint progress against distance to normalized Frobenius `1`.
