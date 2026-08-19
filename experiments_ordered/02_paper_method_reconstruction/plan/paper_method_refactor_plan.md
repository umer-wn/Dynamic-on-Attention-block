# Paper-Method Refactor Plan

## Understand

The current project does not implement the Jacobian method used in `arXiv:1909.05176`, *Optimal Machine Intelligence at the Edge of Chaos*.

The previous experiments measured local Transformer block Jacobians:

```text
J_l = d h_{l+1} / d h_l
metric = sigma_max(J_l) or top-k singular values of J_l
```

This is useful as a layer sensitivity probe, but it is not the paper's method.

The paper treats a neural network submodule as a discrete dynamical operator:

```text
x_{t+1} = f(x_t)
```

Then it studies the asymptotic behavior after repeatedly feeding the output back as the next input. The paper's edge-of-chaos condition is based on the normalized Frobenius norm of the Jacobian of this operator near its asymptotic state:

```text
J_f^* = d f(x) / d x evaluated near an asymptotic fixed point, periodic orbit, or chaotic attractor
critical condition: (1 / N) ||J_f^*||_F^2 ~= 1
equivalently: (1 / sqrt(N)) ||J_f^*||_F ~= 1
```

For finite-dimensional periodic or chaotic trajectories, the paper uses a geometric mean over local asymptotic Jacobian norms along the trajectory, closely related to finite-time Lyapunov analysis.

Therefore, the previous layer-wise LLM experiments should not be used as direct evidence for or against the paper's hypothesis. They answer a different question.

## Can The Current Project Be Reused?

Yes, but only as infrastructure. The scientific core must be refactored.

Reusable parts:

- Model and tokenizer loading.
- Config parsing, metadata, output path conventions, JSONL writing.
- Dataset loading and tokenization utilities.
- Some JVP/VJP autograd patterns.
- Existing Lanczos utilities if we still want spectral estimates as secondary diagnostics.

Parts that must change:

- The target map must no longer be `h_l -> h_{l+1}` for one Transformer block.
- The primary map must be a same-dimensional operator `f: R^N -> R^N` that can be iterated.
- The primary metric must change from `sigma_max(J)` to normalized Frobenius norm or Lyapunov-style trajectory stability.
- The experiment must explicitly run input/output feedback loops.
- The output records must include trajectory convergence, attractor type, normalized Jacobian norm, and loop parameters.

## Main Refactor Goal

Build a paper-faithful experiment pipeline:

```text
1. define an equal-dimensional neural operator f
2. choose initial state x0
3. iterate x_{t+1} = f(x_t)
4. discard burn-in steps
5. evaluate asymptotic stability using:
   a. normalized Frobenius Jacobian norm when feasible
   b. nearby-trajectory separation when Jacobian is too expensive
   c. finite-time Lyapunov-style estimates as a bridge metric
6. compare the edge-of-chaos metric with model quality or checkpoint progress
```

## Operator Design Options

### Option A: Small Controlled MLP/CNN Replication

Use a small network whose hidden operator has the same input/output dimension.

This is the cleanest first experiment because it follows the paper most directly:

```text
dataset: FashionMNIST or CIFAR10
operator f: network trunk ending in a layer with dimension equal to input dimension
loop: x_{t+1} = f(x_t)
metric: (1 / sqrt(N)) ||J_f(x_t)||_F along asymptotic states
```

Pros:

- Most faithful to the paper.
- Small enough for explicit Frobenius estimates using Hutchinson trace estimation.
- Easy to validate phases with nearby trajectory separation.

Cons:

- Less directly connected to LLMs.
- Requires adding vision dataset/model support if not already present.

### Option B: LLM Embedding-Space Dynamical Operator

Define an equal-dimensional LLM operator over token embeddings:

```text
input:  x_t in shape [seq_len, hidden_dim]
output: final hidden states with shape [seq_len, hidden_dim]
map:    x_{t+1} = normalize_or_project(model(inputs_embeds=x_t).hidden_states[-1])
```

Pros:

- Reuses current LLM infrastructure.
- Keeps the project focused on language models.

Cons:

- The operator may diverge or collapse without normalization.
- It is less paper-faithful because LLMs are not naturally designed to feed final hidden states back as input embeddings.
- Attention masks, positional embeddings, layer norms, and residual scale may make the dynamics hard to interpret.

### Recommendation

Start with Option A as a controlled paper replication, then add Option B only after the paper-style pipeline is validated.

The current project can support both, but Option B should be treated as an extension, not the first refactor target.

## Proposed File-Level Refactor

Add new files instead of modifying old experiment scripts in place.

Suggested modules:

```text
src/dynamics.py
scripts/compute_dynamical_edge.py
scripts/analyze_dynamical_edge.py
configs/paper_mlp_fashionmnist_debug.yaml
configs/paper_llm_embedding_dynamics_debug.yaml
reports/paper_method_refactor_findings.md
```

Keep old scripts available but mark their interpretation clearly:

```text
compute_jacobian_features.py       layer sensitivity probe, not paper method
compute_jacobian_lanczos.py        layer spectrum probe, not paper method
compute_global_jacobian_lanczos.py input-output sensitivity probe, not paper method unless wrapped in an iterative operator
```

## New Experiment Definitions

### Experiment 1: Paper-Style MLP Debug

Purpose:

Validate that the refactored code reproduces the paper's workflow on a small controlled model.

Model:

```text
MLP trunk:
input_dim = 784
hidden layers = configurable
operator output_dim = 784
classifier head = separate, not part of f
activation = ReLU except final classifier
```

Loop:

```text
x_0 = flattened image
x_{t+1} = f(x_t)
burn_in_steps = 100
eval_steps = 100
```

Metrics:

```text
normalized_frobenius = ||J_f(x_t)||_F / sqrt(N)
hutchinson_trace_jtj = estimate trace(J^T J)
nearby_trajectory_distance = ||x_t - x'_t||
finite_time_log_gain = mean log(||delta_{t+1}|| / ||delta_t||)
converged_fixed_point
periodic_or_oscillatory
diverged
```

Success criteria:

- Code records an explicit feedback trajectory.
- Frobenius estimate and nearby-trajectory stability agree qualitatively.
- The debug config runs on CPU or a small GPU without training by default.

Important: training is not part of this first refactor unless explicitly approved.

### Experiment 2: Checkpoint Sweep After Debug

Purpose:

Measure whether trained checkpoints move toward the paper's edge condition.

Inputs:

```text
checkpoint list
same architecture
same dataset split
same seed list
same loop settings
```

Outputs:

```text
results/raw/{experiment}__dynamical_edge.jsonl
results/processed/{experiment}__dynamical_edge_summary.csv
results/figures/{experiment}__edge_metric_by_checkpoint.png
results/figures/{experiment}__trajectory_phase_examples.png
```

Success criteria:

- The best validation checkpoints are closer to normalized Frobenius near 1 than weak checkpoints.
- Nearby-trajectory separation supports the same phase labeling.

### Experiment 3: LLM Embedding Dynamics Debug

Purpose:

Explore whether a paper-style iterative operator can be defined for LLM hidden spaces without producing meaningless collapse/divergence.

Operator candidates:

```text
candidate A: final_hidden -> next inputs_embeds directly
candidate B: final_hidden -> layernorm -> next inputs_embeds
candidate C: residual-mixed update x_{t+1} = (1 - alpha) x_t + alpha final_hidden
candidate D: selected block-stack f instead of full model
```

Required safeguards:

```text
fixed sequence length
fixed attention mask
fixed positional behavior
norm tracking at every step
divergence threshold
collapse threshold
same random seed
no weight updates
```

Primary interpretation:

Only treat this as paper-inspired if the loop is stable enough to form fixed points, cycles, or bounded attractor-like trajectories.

## Implementation Details

### Frobenius Estimation

For large `N`, do not construct the full Jacobian.

Use Hutchinson estimation:

```text
||J||_F^2 = trace(J^T J)
estimate trace(J^T J) = mean_k ||J v_k||^2
v_k sampled as Rademacher or Gaussian unit probes
```

Implementation pattern:

```text
for each asymptotic state x_t:
    sample probes v_k
    compute jvp(f, x_t, v_k)
    accumulate ||jv||^2
normalized_frobenius = sqrt(mean_probe_norm_sq) / sqrt(N)
```

Use multiple states along the post-burn-in trajectory:

```text
local_norms = [normalized_frobenius(x_t)]
geometric_mean_norm = exp(mean(log(local_norms)))
```

### Nearby-Trajectory Stability

Run:

```text
x'_0 = x_0 + epsilon * unit_noise
x_{t+1} = f(x_t)
x'_{t+1} = f(x'_t)
distance_t = ||x_t - x'_t||
```

This is the fallback metric when Jacobian computation is too expensive. It should not replace the Frobenius estimate in the small controlled experiments.

### Phase Labels

Use conservative labels:

```text
stable_fixed_like:
  final trajectory variance low and nearby distance -> 0

periodic_or_bounded:
  bounded trajectory, nonzero variance, nearby distance -> 0 or stays small

chaotic_or_sensitive:
  bounded trajectory but nearby distance remains positive or grows

divergent:
  norm exceeds threshold or contains NaN/Inf

collapsed:
  norm or variance falls below threshold for most states
```

Do not over-claim true chaos from short finite trajectories.

## Config Requirements

Every new config should include:

```yaml
experiment_name: paper_mlp_fashionmnist_debug
seed: 1234
device: auto
dtype: float32
output_dir: results
skip_existing: true

dynamics:
  burn_in_steps: 100
  eval_steps: 100
  perturbation_epsilon: 1.0e-5
  divergence_threshold: 1.0e6
  collapse_threshold: 1.0e-8
  frobenius_probes: 16
  frobenius_eval_states: 16
  probe_distribution: rademacher

models:
  - name: mlp784_debug
    checkpoint: null

dataset:
  name: fashion_mnist
  split: test
  num_samples: 128
```

## Review Risks

- LLM embedding dynamics may be mathematically artificial and hard to interpret.
- Frobenius norm and `sigma_max` answer different questions; old results should not be re-labeled.
- A finite-time trajectory cannot prove true chaos by itself.
- Normalization choices in LLM dynamics can change the phase behavior.
- Training/checkpoint sweeps require explicit approval before running.

## Execution Order For Next Work Session

1. Implement `src/dynamics.py` with generic iteration, nearby-trajectory tracking, Hutchinson Frobenius estimation, and phase labeling.
2. Add `scripts/compute_dynamical_edge.py` for the MLP debug experiment.
3. Add `configs/paper_mlp_fashionmnist_debug.yaml`.
4. Run a tiny CPU smoke test with a randomly initialized MLP and 2 samples.
5. Validate outputs manually: trajectory length, norm traces, normalized Frobenius values, phase labels.
6. Only after smoke passes, decide whether to add a checkpoint sweep or LLM embedding dynamics.

## Final Decision

Refactor the current project rather than starting a separate project.

Reason:

The current repository already has useful experiment infrastructure, but the scientific method must be replaced. The safest path is to add a new paper-method pipeline beside the old scripts, keep old results clearly labeled as non-paper-method layer probes, and validate the new pipeline on a small controlled model before applying it to LLMs.
