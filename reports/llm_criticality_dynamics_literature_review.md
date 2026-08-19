# LLM Criticality and Dynamical-Systems Literature Map

## Scope

The project starts from Feng, Zhang, and Lai, *Optimal Machine Intelligence at the Edge of Chaos* (arXiv:1909.05176). That work iterates an equal-dimensional neural operator and relates performance to asymptotic Jacobian behavior near the boundary between periodic/pseudoperiodic and chaotic regimes. Its empirical models are computer-vision networks modified to admit feedback iteration; it does not directly establish the claim for autoregressive LLMs.

## Three Different Meanings That Must Not Be Conflated

1. **Time-feedback criticality:** repeated application `x(t+1)=f(x(t))`, the meaning closest to arXiv:1909.05176 and to this project's constructed embedding feedback loop.
2. **Depth-wise signal propagation:** treating Transformer layers as steps and studying covariance, rank collapse, partial Jacobian norms, or dynamical isometry. This informs trainability but is not the same temporal system.
3. **Autoregressive generative criticality:** dynamics or phase-like behavior across generated tokens and decoding temperature. This is native to language generation but uses a stochastic, growing-context process rather than a fixed equal-dimensional deterministic map.

## Closest Evidence

- Feng et al., [Optimal Machine Intelligence at the Edge of Chaos](https://arxiv.org/abs/1909.05176): foundational operator-feedback criterion and the direct source of the current hypothesis.
- Kedia et al., [Transformers Get Stable](https://openreview.net/forum?id=30waYPIZUA): end-to-end forward/backward signal propagation theory for language-model Transformers; directly relevant to normalization and Jacobian stability, but focused on depth/trainability.
- Noci et al., [Signal Propagation in Transformers](https://arxiv.org/abs/2206.03126): rank collapse and signal propagation at initialization; a depth-wise comparator.
- Fernando and Guitchounts, [Transformer Dynamics](https://arxiv.org/abs/2502.12131): treats residual-stream evolution across layers as a dynamical system and uses perturbations and reduced trajectories. It supports dynamical analysis of LLM activations, but not the exact feedback operator used here.
- Ugail and Howard, [Dynamical Systems Analysis Reveals Functional Regimes in LLMs](https://arxiv.org/abs/2601.11622): studies activation time series during autoregressive generation across reasoning, repetition, temperature, pruning, and noise regimes. It motivates native-generation controls.
- Ruan et al., [Generative Criticality in LLM Temperature Scaling](https://arxiv.org/abs/2606.06238): reports susceptibility peaks and order-parameter changes across decoding temperature in Qwen3. It is the closest explicit LLM criticality claim found, but concerns statistical properties of generated token embeddings rather than asymptotic Jacobians of a deterministic feedback operator.
- Alekseev, [Subcritical Signal Propagation at Initialization in Normalization-Free Transformers](https://arxiv.org/abs/2604.11890): uses averaged partial Jacobian norms and emphasizes that normalization changes criticality behavior; this directly cautions against interpreting output-scale interventions naively.
- Schoenholz et al., [Deep Information Propagation](https://arxiv.org/abs/1611.01232), and Pennington et al., [Dynamical Isometry](https://arxiv.org/abs/1711.04735): foundational depth-wise edge-of-chaos and Jacobian-spectrum theory.

## Evidence Assessment

No paper in this initial search directly proves that a pretrained causal LLM's constructed map `inputs_embeds -> final_hidden -> inputs_embeds` should sit at the arXiv:1909.05176 critical boundary. Related papers support three neighboring claims: critical signal propagation can aid trainability; residual streams have structured layer dynamics; and autoregressive generation can display temperature-dependent phase-like behavior. These are motivation, not validation of the current operator.

## Recommended Research Program

1. Complete mathematical positive controls and numerical precision checks.
2. Compare three operators on identical prompts: constructed hidden feedback, layer-depth residual dynamics, and native autoregressive generation.
3. Add training-checkpoint comparisons only after caching Pythia revisions.
4. Relate criticality measures to held-out loss or task behavior; proximity to one without a performance axis cannot test "optimal intelligence."
5. Replicate with multiple seeds, epsilon values, samples, and projection choices, and report uncertainty rather than only means.
