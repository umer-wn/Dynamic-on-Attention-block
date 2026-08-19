# Pythia Embedding-Expectation Smoke Plan

## Objective

Try a more paper-faithful LLM dynamical operator by making the input and output spaces exactly the same embedding space.

## Operator

Previous failed operator:

```text
inputs_embeds -> final_hidden
```

New operator:

```text
inputs_embeds -> logits -> softmax(logits / T) -> expected input embedding
```

This produces a differentiable tensor with the same shape as `inputs_embeds`, using the language model's actual output head rather than treating contextual hidden states as outputs.

## Scale

- Model: `EleutherAI/pythia-70m`
- `sequence_length=64`
- `num_samples=2`
- `burn_in_steps=8`
- `eval_steps=32`
- `frobenius_eval_states=2`
- `frobenius_probes=2`

This is a smoke test. If it runs and avoids collapse/divergence, the next step is a normal-dimension run with `sequence_length=128`.

## Interpretation

If the normalized Frobenius moves closer to 1 without artificial residual mixing, this operator is a better candidate for an LLM reproduction of the paper. If it collapses or remains far from 1, the likely next step is a trained embedding-space reconstruction/denoising map rather than a pretrained LM head alone.
