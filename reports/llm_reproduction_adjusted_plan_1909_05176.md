# Adjusted LLM Reproduction Plan for Paper 1909.05176

## Motivation

The current direct-feedback experiments did not reproduce the paper's edge-of-chaos result. Pythia and GPT-2 are contractive under `inputs_embeds -> final_hidden`; Qwen is only partially asymptotic. Restoration probes showed that norm matching fails and external residual mixing can create a near-edge number as an artifact.

The paper's abstract states that the edge is determined by asymptotic Jacobian norm values of the nonlinear operator and that optimality is associated with information transfer. For an LLM reproduction, the operator must therefore be a meaningful same-space map, not merely a hidden-state feedback loop.

Source: https://arxiv.org/abs/1909.05176

## New Operator Attempted

Implemented target:

```text
embedding_expectation:
inputs_embeds -> logits -> softmax(logits / T) -> expected input embedding
```

This is more faithful than `final_hidden` because it maps embedding tensors back into embedding tensors through the LM output head.

## Smoke Evidence

Pythia-70M, `sequence_length=64`, `num_samples=2`:

| operator | temperature | mean normalized Frobenius | status |
| --- | ---: | ---: | --- |
| final_hidden direct, seq128 baseline | n/a | 0.463930 | settled but contractive |
| embedding_expectation | 0.25 | 0.110450 | contractive/unsettled |
| embedding_expectation | 0.5 | 0.343072 | closest, still below edge |
| embedding_expectation | 1.0 | 0.244026 | contractive/unsettled |
| embedding_expectation | 2.0 | 0.135963 | more settled, more contractive |

Conclusion: using the pretrained LM head as a differentiable embedding-space map is viable, but it still does not recover `||J*||/sqrt(N) ~= 1`.

## Adjusted Experimental Scheme

### Phase 1: Same-space pretrained operator

Status: implemented and smoke-tested.

Next only if needed:

- run `embedding_expectation` at `T=0.5`, `sequence_length=128`, `num_samples=8`;
- add product-Jacobian windows `[2,4]`;
- compare to final-hidden direct baseline.

This phase is not expected to reproduce the paper, but it gives a better pretrained-LLM baseline.

### Phase 2: Trained embedding-space reconstruction map

This is the recommended next serious attempt.

Train a lightweight same-space map on frozen LLM representations:

```text
x_t = token embeddings + small noise
g_theta(x_t, attention_mask) -> reconstructed clean embeddings
```

Then evaluate:

```text
x_{t+1} = g_theta(x_t)
```

Metrics:

- reconstruction loss or denoising loss as the task-performance proxy;
- asymptotic normalized Frobenius `||J*||/sqrt(N)`;
- product-Jacobian gain;
- nearby trajectory separation;
- fixed-projection return maps and Poincare diagnostics.

This mirrors the paper more closely because the model is explicitly trained as a same-space nonlinear operator.

### Phase 3: Model comparison matrix

Once Phase 2 works for Pythia:

- run the same trained-map protocol for GPT-2 and Qwen;
- build a matrix with task loss, normalized Frobenius, product gain, settled fraction, and trajectory-distance metrics;
- check whether better reconstruction/denoising performance correlates with being closer to the edge.

## Stop Criteria

Do not claim reproduction unless:

- the operator is same-space by construction;
- trajectories settle or have a well-characterized periodic attractor;
- normalized Frobenius is near 1 without external identity mixing;
- task performance is measured and can be compared across model/checkpoint/setting.

If Phase 2 still stays far below 1, the most likely conclusion is that pretrained autoregressive LLM representations do not naturally instantiate the same edge-of-chaos mechanism studied in the paper without a dedicated same-space dynamical training objective.
