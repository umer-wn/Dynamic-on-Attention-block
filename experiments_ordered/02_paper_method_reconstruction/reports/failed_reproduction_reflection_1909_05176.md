# Failed Reproduction Reflection for Paper 1909.05176

## Question

Why did the natural-language-model experiments fail to reproduce the paper's edge-of-chaos result, and what improved attempts were made to recover it?

## Current Evidence

The current paper-alignment matrix is:

```text
results/processed/paper_alignment_matrix.csv
```

Direct settled rows:

- Pythia-70M: mean normalized Frobenius `0.463930`, settled fraction `1.0`.
- GPT-2: mean normalized Frobenius `0.240475`, settled fraction `1.0`.

Direct but not fully settled row:

- Qwen2.5-0.5B: primary long run mean normalized Frobenius `0.469756`, settled fraction `0.5`.
- Qwen burn256 check: still settled only `2/4` samples.

These results do not reproduce the paper's `||J*||/sqrt(N) ~= 1` criterion. Pythia and GPT-2 are clearly contractive under the current operator. Qwen is mixed: some samples settle, some remain large-step trajectories, so its row is not a clean asymptotic estimate.

## Why This Likely Happened

### 1. The language-model feedback operator is not the same object as the paper's dynamical system

The paper evaluates systems explicitly made into maps of the form `x_{t+1}=f(x_t)` with matched input and output spaces. In this project, the map is constructed by feeding `inputs_embeds` through a pretrained autoregressive language model and taking `final_hidden` as the next state.

That is a plausible research adaptation, but it is not the same trained map:

- token embeddings are a learned input representation, not the model's natural output space;
- final hidden states are contextual features, not reconstructed inputs;
- the model was not trained to make repeated hidden-state feedback meaningful;
- fixed attention masks and positions impose an additional structure absent from the paper's image-state dynamics.

### 2. The direct hidden-state map is strongly contractive

For Pythia and GPT-2, final step deltas become small and normalized Frobenius norms are well below 1. This means the current operator tends toward stable attractors rather than the paper's edge criterion.

The product-Jacobian estimates agree:

- Pythia product log gain mean: `-0.704469`
- GPT-2 product log gain mean: `-0.764971`

Negative log gain is consistent with contraction.

### 3. Static layer spectra are not the paper's asymptotic Jacobian

The project has layer-local static spectra with large singular values, especially in Qwen. Those are useful diagnostics, but they do not contradict the feedback-dynamics result. The paper's criterion is about the Jacobian of the iterated operator evaluated near the attractor, not isolated layer spectra.

### 4. Qwen may not be fixed-point-like under this operator

Qwen remains partially unsettled even with longer burn-in:

- `qwen_dynamical_edge_normal_long`: settled `2/4` samples.
- `qwen_dynamical_edge_burn256`: settled `2/4` samples.

This suggests some trajectories may be long-period, quasi-periodic, or otherwise not captured by a simple fixed-point convergence criterion. It does not currently support a clean edge-of-chaos claim because the attractor state is not reliably identified.

## Improved Restoration Attempts

### Attempt A: norm-matched output

Experiment:

```text
experiments/pythia_dynamical_edge_norm_matched_probe/
```

Change:

```text
x_{t+1} = f(x_t) * ||x_t|| / ||f(x_t)||
```

Result:

- Mean normalized Frobenius: `0.046596`
- Collapsed fraction: `0.5`

Conclusion: norm matching made the map more degenerate and did not restore the paper's edge criterion. The failure is not just a simple norm mismatch.

### Attempt B: external residual update

Experiment:

```text
experiments/pythia_dynamical_edge_residual_alpha01_probe/
```

Change:

```text
x_{t+1} = 0.9 x_t + 0.1 f(x_t)
```

Result:

- Mean normalized Frobenius: `0.945701`
- Settled fraction: `0.0`
- Max final step delta: `2.413796`

Conclusion: the scalar can be pushed close to 1 by injecting a large identity component, but this is an artifact of the update rule. It is not evidence that the pretrained language model itself reproduces the paper's result.

## Overall Conclusion

The current natural-language-model setup does not reproduce the paper's edge-of-chaos finding. The most likely reason is structural mismatch: the paper studies trained input-output dynamical systems, while this project constructs a feedback system from hidden states of pretrained autoregressive language models.

The improved attempts did not recover a faithful result:

- scale correction failed and caused collapse;
- residual mixing can manufacture a near-edge number but changes the dynamical system in a non-paper-like way.

## Best Next Attempts

1. Use a language-model objective that defines an actual same-space map, such as embedding reconstruction or denoising in embedding space, then evaluate the trained map after convergence.
2. Analyze Qwen's unsettled samples as possible cycles by adding period-return distances, not just final step deltas.
3. Compare multiple operator definitions separately: `inputs_embeds -> final_hidden`, block-level residual stream maps, and embedding-decoder-reprojected maps.
4. Avoid claiming reproduction unless the same operator has settled trajectories and normalized Frobenius near 1 without artificial identity mixing.
