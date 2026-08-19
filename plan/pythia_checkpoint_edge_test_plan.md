# Pythia Checkpoint Training-to-Edge Experiment Plan

## Core Question

Does pretraining a fixed language-model architecture move its equal-dimensional embedding-feedback dynamics toward the edge-of-chaos criterion described in arXiv:1909.05176?

## Hypotheses

- H1: validation loss decreases with checkpoint step.
- H2: `abs(log(normalized_frobenius))` decreases with checkpoint step/loss.
- H3: finite-time JVP-product log gain approaches zero as loss improves.
- H4: any apparent near-edge checkpoint also shows bounded asymptotic trajectories and recurrence/bifurcation evidence; a scalar crossing alone does not count.

## Models and Checkpoints

Fixed architecture: `EleutherAI/pythia-70m`.

Stage A debug checkpoints: `step0`, `step16000`, `step143000`.

Stage B, only if Stage A passes: `step0`, `step1000`, `step4000`, `step16000`, `step64000`, `step143000`.

## Stage A Protocol

- Four fixed WikiText validation samples, sequence length 64.
- Direct whole-model operator: `inputs_embeds -> final_hidden`.
- Burn-in/evaluation: 64/64.
- Frobenius: four terminal states, four Rademacher probes.
- Product windows: 2 and 4, one probe for debug.
- Finite perturbation epsilon: `1e-3` for secondary diagnostics.
- Compute validation loss on exactly the same tokenized samples.
- GPUs 5, 6, 7: one checkpoint per GPU.

## Analysis

Report per-checkpoint means and uncertainty, Spearman correlations with checkpoint step and validation loss, trajectory status, relative step delta, and consistency between Frobenius and tangent-product results.

## Decision Rules

- Support for training-to-edge requires a consistent trend across at least Frobenius edge distance and tangent-product log gain, together with improving loss.
- A three-checkpoint debug result is provisional; Stage B is required for a substantive claim.
- Failure to find a trend in Stage B challenges H1 for this operator/protocol, but does not falsify all possible LLM formulations.
- Operator robustness must be tested separately before generalizing to LLMs.

## Storage and Traceability

- Data/log root: `/home/luohaoming/model_feature_experiments/pythia_checkpoint_edge_test`.
- Configs, plan, analysis code, and final report remain in `/data1/luohaoming/model_feature`.
- Offline mode will be used during experiments after checkpoints are explicitly downloaded and verified.

## Resource and Download Gate

Only `main` is currently cached. Stage A requires downloading three Pythia-70M checkpoint revisions. No download or GPU experiment starts until checkpoint download is explicitly approved. Expected storage is several hundred MB to roughly 1 GB including snapshots; exact cache growth will be recorded.
