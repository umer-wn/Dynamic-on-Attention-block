# Core-Paper Alignment and Historical-Conclusion Audit

## Verdict

The project is directionally related to Feng, Zhang, and Lai (arXiv:1909.05176), because it constructs a same-dimensional neural operator and iterates it to an asymptotic trajectory. It is not yet a faithful LLM reproduction of the paper, and existing results cannot prove or disprove the paper's "optimal intelligence at the edge of chaos" hypothesis.

## Paper Requirements

The paper requires: a fixed same-dimensional operator; long iteration to an asymptotic attractor; a phase boundary assessed by normalized asymptotic Frobenius norm in the high-dimensional approximation or by geometric/time-averaged Lyapunov behavior in finite dimensions; independent phase validation with final separation of nearby initial states and Poincare maps; and, crucially, a relationship between stability and model performance across training epochs/models.

The paper uses 100 samples and 500 iterations for its main numerical methods, and compares stability throughout training. Its derivation assumes weakly correlated Jacobian elements for the Frobenius approximation and ergodicity when replacing time averages by sample averages.

## What Is Aligned

- `inputs_embeds -> final_hidden` is a deterministic same-dimensional map when the model, attention mask, and positions are fixed.
- Burn-in followed by asymptotic evaluation follows the paper's general construction.
- Hutchinson JVP with Rademacher input probes estimates `||J||_F^2`; division by active dimension estimates the paper's normalized Frobenius quantity.
- The identity residual control correctly returns normalized Frobenius one.

## Major Defects Found

### 1. Jacobian-product metric was mislabeled

The existing product metric averages norm growth from one or a few random directions. The paper's method 2 estimates the dominant growth/spectral radius of the Jacobian product. Without repeated renormalization and alignment to the leading tangent direction, the current metric is not a maximal Lyapunov estimate. Prior claims of long-term tangent contraction based on this field are unsupported.

### 2. Nearby-trajectory metric does not match paper method 3

The implementation reports the ratio between the first and last stored post-burn-in distances. The paper assesses final asymptotic separation of two close initial states after long iteration. The current sign can change with the storage window and ignores growth/saturation during burn-in. It is not paper-equivalent.

### 3. Phase classification is insufficient

The classifier uses an absolute step-delta threshold and a simple perturbation ratio. It does not identify cycles, estimate recurrence, or distinguish quasiperiodicity from chaos. Labels such as `bounded_nonfixed_like` are descriptive only.

### 4. The high-dimensional Frobenius equivalence is unverified for Transformers

Transformer Jacobians are structured, correlated, non-normal, and normalization-dominated. The paper's random/weak-correlation argument cannot be assumed. Local Frobenius below one is evidence about average squared singular gain, not proof that the dominant Lyapunov exponent is negative.

### 5. Core optimality hypothesis has not been tested

Only final pretrained checkpoints were measured. Artificial residual/output-scale interventions do not provide a model-performance axis. No current result relates distance to criticality to validation loss, accuracy, or downstream ability across training. Therefore current work neither proves nor falsifies optimal intelligence at criticality.

### 6. Protocol is still a debug scale

Most recent runs use four samples and 64+64 iterations, versus the paper's 100 samples and 500 iterations. Results are calibration evidence only.

## Corrections to Historical Conclusions

- Valid: the constructed Pythia/GPT-2 operators have local normalized Frobenius estimates below one under tested settings.
- Not established: their maximal Lyapunov exponents are negative.
- Not established: they occupy the paper's stable phase rather than a periodic/quasiperiodic phase.
- Not established: Qwen is closer to or farther from the edge than other models.
- Not established: LLM training moves models toward the edge of chaos.
- Not established: the paper is supported or contradicted for LLMs.

## Method Gate for Further Claims

Implement a Benettin-style tangent algorithm with per-step JVP renormalization, record final asymptotic separation from the original perturbation, add relative convergence and recurrence diagnostics, and validate all metrics on identity, contraction, expansion, and a known chaotic map before returning to LLM comparisons.
