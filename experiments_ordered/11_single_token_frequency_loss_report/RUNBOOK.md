# Phase 11 runbook

Remote paths:

```text
repository:
  /data1/luohaoming/model_feature
experiment code:
  /data1/luohaoming/model_feature/experiments_ordered/11_single_token_frequency_loss_report
large outputs:
  /home/luohaoming/model_feature_experiments/single_token_frequency_loss_report
```

Generate the 8-bin × 4-token frequency cohort and run the isolated-token
dynamics on eight GPUs in two rounds:

```bash
bash scripts/run_single_token_8bins.sh
```

Prepare the train-split proxy manifest:

```bash
/public/luohaoming/model_feature/.venv/bin/python \
  scripts/prepare_pile_train_manifest.py \
  --output-root /home/luohaoming/model_feature_experiments/single_token_frequency_loss_report
```

Compute loss for the same checkpoint set already available for The Pile test:

```bash
for revision in step0 step1000 step5000 step9000 step10000 step13000 step16000 \
  step17000 step21000 step25000 step29000 step33000 step37000 step41000 \
  step45000 step49000 step53000 step57000 step61000 step65000 step69000 \
  step73000 step77000 step81000 step85000 step89000 step93000 step97000 \
  step100000 step101000 step105000 step133000 step143000; do
  CUDA_VISIBLE_DEVICES=0 /public/luohaoming/model_feature/.venv/bin/python \
    /data1/luohaoming/model_feature/scripts/compute_validation_corpus_loss.py \
    --root /home/luohaoming/model_feature_experiments/single_token_frequency_loss_report \
    --source-id the_pile_train \
    --revision "$revision"
done
```

The sparse late checkpoints `step101000`, `step105000`, `step133000`, and
`step143000` have incomplete per-revision tokenizer files in the offline cache.
Because the Pythia tokenizer is checkpoint-invariant, evaluate those four with:

```bash
--tokenizer-revision step100000
```

Prepare and evaluate the fixed local hard natural-language set. This reads only
the two already-cached OpenWebMath parquet shards and runs the same 33
checkpoints, sequence length 64, 512 samples, and token-weighted loss:

```bash
bash scripts/run_local_hard_natural_language_loss.sh
```

Build plots, CSV tables, and the README report:

```bash
/public/luohaoming/model_feature/.venv/bin/python \
  scripts/build_frequency_loss_report.py \
  --output-root /home/luohaoming/model_feature_experiments/single_token_frequency_loss_report \
  --single-root /home/luohaoming/model_feature_experiments/single_token_frequency_8bins \
  --test-loss-root /home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan \
  --train-loss-root /home/luohaoming/model_feature_experiments/single_token_frequency_loss_report \
  --hard-loss-root /home/luohaoming/model_feature_experiments/single_token_frequency_loss_report
```
