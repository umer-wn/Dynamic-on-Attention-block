# Experiment 16: frequency-stratified window dynamics

This experiment follows four tokens from separated WikiText-2 frequency strata
through Pythia-70m single-token dynamic steps 0--1024.

For every 256-step endpoint it records:

- a shared, fixed four-dimensional random projection;
- cosine, Euclidean, and LM-head nearest tokens with confidence margins;
- Jacobian spectral radius, spectral abscissa, operator 2-norm, and Frobenius norm;
- the finite-time largest Lyapunov exponent in that 256-step window.

It also records each checkpoint's 0--1024 Lyapunov exponent and fixed-corpus
test loss. For selected visibly recurrent checkpoints it estimates the
full-512D return period over steps 512--1024. The Lyapunov exponent is the mean
natural-log tangent growth per dynamic step, computed with JVP propagation and
per-step tangent renormalization; the final 256-step window is additionally
summarized by mean, population/sample variance, and token min/max.

Run:

```bash
MPLCONFIGDIR=/tmp/experiment16-mpl-cache \
CUDA_VISIBLE_DEVICES=0 \
/data1/luohaoming/langurage_feature/venv/bin/python \
experiments_ordered/16_frequency_stratified_window_jacobian/scripts/run_experiment16.py \
--device cuda:0 --stage all
```

Large state tensors and resumable checkpoint parts are stored below
`/home/luohaoming/model_feature_experiments/experiment16_frequency_stratified_window_jacobian`.
Tables, plots, and the generated report stay in this experiment directory.
