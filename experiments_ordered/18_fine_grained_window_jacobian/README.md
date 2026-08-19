# Experiment 18: fine-grained Jacobian metrics

This experiment supplements Experiment 16 with direct Jacobian measurements at
dynamic steps `0, 64, 128, ..., 1024`. The 64-step samples fill the intervals
between the original 256-step endpoints; metric values are computed on the
actual hidden states and are not numerically interpolated.

The eight fixed tokens cover all eight WikiText-2 frequency bins. For each
checkpoint, dynamic step, and token, the experiment records only:

- spectral radius `rho(J)`;
- operator 2-norm `||J||_2`;
- Jacobian Frobenius norm `||J||_F`.

No projection trajectory or trace file is produced.

Proof-Pile-2 test loss is evaluated on one deterministic 512-document manifest
drawn from the official test splits in the corpus-size ratio 270 ArXiv, 140
OpenWebMath, and 102 AlgebraicStack. Every checkpoint uses the same first 64
Pythia-token positions. Dataset files and the tokenized manifest are stored
under `/home/luohaoming/proof_pile2`; the resulting checkpoint loss and
perplexity columns are appended to both raw and visualization-ready CSV files.

Run on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
/public/luohaoming/model_feature/.venv/bin/python \
experiments_ordered/18_fine_grained_window_jacobian/scripts/run_experiment18.py \
--device cuda:0 --stage all
```

The run is resumable. Per-checkpoint raw rows are written after every measured
token. The visualization-ready aggregate is:

`processed/jacobian_fine_grained_8tokens.csv`

Large state tensors are stored under
`/data1/luohaoming/model_feature_experiments/experiment18_fine_grained_window_jacobian`.

Evaluate and append Proof-Pile-2 loss:

```bash
CUDA_VISIBLE_DEVICES=0 \
/public/luohaoming/model_feature/.venv/bin/python \
experiments_ordered/18_fine_grained_window_jacobian/scripts/compute_proof_pile2_loss.py \
--device cuda:0
```
