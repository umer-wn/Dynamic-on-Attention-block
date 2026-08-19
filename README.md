# LLM Edge of Chaos Experiments

This repository estimates whether open-weight Transformer language models show layer-wise hidden-state Jacobian gains near the "edge of chaos" regime.

The starting research question is:

```text
During pretraining or across model scales, do Transformer language models move
toward a layer-wise hidden-state Jacobian gain close to 1?
```

The main measured signal is:

```text
sigma_max(d h_{l+1} / d h_l)
```

where `h_l` is the hidden state entering a Transformer block. The implementation avoids constructing a full Jacobian and estimates the top singular value with power iteration using JVP/VJP autograd operations.

## Repository Layout

```text
configs/                 YAML experiment configs
scripts/                 CLI entrypoints
src/                     reusable experiment utilities
results/raw/             per-run JSONL outputs
results/processed/       aggregated CSV tables
results/figures/         generated plots
/public/luohaoming/model_feature/hf_cache/           Hugging Face cache
reports/                 short experiment reports
notebooks/               optional exploratory analysis
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv /public/luohaoming/model_feature/.venv
source /public/luohaoming/model_feature/.venv/bin/activate
pip install -r requirements.txt
```

This workspace currently uses `/public/luohaoming/model_feature/.venv` with CPU PyTorch (`torch 2.8.0+cpu`). The virtual environment can stay under public storage because it is large dependency data, while code, configs, reports, and experiment outputs stay in this workspace. For GPU runs, replace PyTorch with the build that matches the server CUDA version before running larger experiments.

## Smoke Test

```bash
python scripts/download_models.py --config configs/pythia_smoke.yaml
python scripts/compute_weight_features.py --config configs/pythia_smoke.yaml
python scripts/compute_jacobian_features.py --config configs/pythia_smoke.yaml
python scripts/compute_perplexity.py --config configs/pythia_smoke.yaml
python scripts/aggregate_results.py --config configs/pythia_smoke.yaml
```

The smoke config uses `EleutherAI/pythia-70m`, `wikitext-2-raw-v1`, sequence length 128, four samples, and the first/middle/last layers.

Expected smoke outputs:

```text
results/raw/pythia_smoke__EleutherAI__pythia-70m__main__weight_features.jsonl
results/raw/pythia_smoke__EleutherAI__pythia-70m__main__seq128__jacobian_features.jsonl
results/raw/pythia_smoke__EleutherAI__pythia-70m__main__seq128__perplexity.jsonl
results/processed/pythia_smoke__jacobian_summary.csv
results/processed/pythia_smoke__criticality_summary.csv
results/processed/pythia_smoke__perplexity_summary.csv
results/figures/pythia_smoke__layer_sigma.png
```

The smoke test has been completed in this workspace with CPU PyTorch. Results are under `results/`, and the short summary is in `reports/initial_findings.md`.

## Experiment Scripts

All scripts take a config path:

```bash
python scripts/<script>.py --config configs/pythia_smoke.yaml
```

`scripts/download_models.py` loads each configured model and tokenizer once so Hugging Face caches them.

`scripts/compute_weight_features.py` computes one row per selected matrix parameter, including spectral norm, Frobenius norm, stable rank, singular value entropy, top singular values, shape, model, checkpoint, and timestamp metadata.

`scripts/compute_jacobian_features.py` estimates block-level `sigma_max(d h_{l+1} / d h_l)` for configured layers and sequence lengths. It captures each block's real forward-call arguments from a normal model pass, then runs power iteration on the block mapping.

`scripts/compute_dynamical_edge.py` implements the paper-style feedback-dynamics probe. It defines an equal-dimensional continuous operator over LLM input embeddings, iterates `x_{t+1} = f(x_t)`, discards burn-in states, and estimates the normalized Frobenius Jacobian norm near the resulting trajectory with Hutchinson JVP probes. This is the preferred entrypoint for testing the `arXiv:1909.05176` method on cached language models.

Small paper-method smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 /public/luohaoming/model_feature/.venv/bin/python scripts/compute_dynamical_edge.py --config configs/pythia_dynamical_edge_smoke.yaml
CUDA_VISIBLE_DEVICES=0 /public/luohaoming/model_feature/.venv/bin/python scripts/compute_dynamical_edge.py --config configs/pythia_dynamical_edge_small.yaml
python scripts/analyze_dynamical_edge.py --config configs/pythia_dynamical_edge_small.yaml
```

`scripts/compute_perplexity.py` computes validation loss and perplexity on the same tokenized samples used for Jacobian analysis.

`scripts/aggregate_results.py` creates processed CSV summaries and initial figures.

## Outputs

Raw JSONL files are written to `results/raw/`.

Aggregated CSV files are written to `results/processed/`.

Figures are written to `results/figures/`.

The scripts skip existing outputs by default. Set `skip_existing: false` in a config to overwrite an experiment output.

Every row includes metadata fields such as model name, checkpoint revision, tokenizer, dataset, sequence length, dtype, device, git commit, and timestamp.

## Configs

- `configs/pythia_smoke.yaml`: minimal smoke test.
- `configs/pythia_training_dynamics.yaml`: Pythia checkpoint dynamics for `pythia-70m` and `pythia-160m`.
- `configs/qwen_compare.yaml`: small modern model comparison.

Important config fields:

```yaml
experiment_name: pythia_smoke
device: auto
dtype: float32
attn_implementation: eager
output_dir: results
cache_dir: /public/luohaoming/model_feature/hf_cache
offline: true
skip_existing: true
models:
  - name: EleutherAI/pythia-70m
    revision: main
dataset:
  name: wikitext
  config: wikitext-2-raw-v1
  split: validation
  num_samples: 4
  sequence_lengths: [128]
layers: [first, middle, last]
power_iterations: 5
jacobian_token_mode: nonpad_flattened
```

Use `layers: all` for full layer sweeps after the smoke run passes.

Set `offline: true` when the model and dataset are already cached under `/public/luohaoming/model_feature/hf_cache`; set it to `false` for first-time downloads.

## Main Experiments

After the smoke test passes, run Pythia training dynamics:

```bash
python scripts/download_models.py --config configs/pythia_training_dynamics.yaml
python scripts/compute_weight_features.py --config configs/pythia_training_dynamics.yaml
python scripts/compute_jacobian_features.py --config configs/pythia_training_dynamics.yaml
python scripts/compute_perplexity.py --config configs/pythia_training_dynamics.yaml
python scripts/aggregate_results.py --config configs/pythia_training_dynamics.yaml
```

Then compare modern small LLMs:

```bash
python scripts/download_models.py --config configs/qwen_compare.yaml
python scripts/compute_weight_features.py --config configs/qwen_compare.yaml
python scripts/compute_jacobian_features.py --config configs/qwen_compare.yaml
python scripts/compute_perplexity.py --config configs/qwen_compare.yaml
python scripts/aggregate_results.py --config configs/qwen_compare.yaml
```

## Interpreting Results

The key fields in Jacobian outputs are:

- `sigma_max`: estimated top singular value of the block hidden-state Jacobian.
- `log_sigma`: `log(sigma_max)`.
- `abs_log_sigma`: distance from the critical value in log space.
- `critical_08_12`: whether `0.8 < sigma_max < 1.2`.
- `critical_09_11`: whether `0.9 < sigma_max < 1.1`.

By default, `jacobian_token_mode: nonpad_flattened` estimates the block gain only on non-padding token positions. This avoids padding positions dominating the top singular vector in fixed-length batches.

Useful first checks:

- Does mean `abs_log_sigma` decrease across Pythia checkpoints?
- Does the fraction of critical layers increase over training?
- Are early, middle, or late layers consistently closer to 1?
- Do weight-only spectral features correlate with activation-conditioned Jacobian gains?
- Does lower perplexity correlate with lower mean `abs_log_sigma` or higher critical-layer fraction?

## Notes

- Start with CPU or a small GPU on the smoke test before running larger configs.
- Jacobian estimation can be memory intensive because it uses second-order autograd structure through a block.
- If memory is limited, reduce `sequence_lengths`, `num_samples`, selected `layers`, or `power_iterations`.
- Gated Hugging Face models are intentionally not included in the default configs.
- The first pass uses base/pretrained models rather than instruction-tuned models.
