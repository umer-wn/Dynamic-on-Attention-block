# Phase 12 runbook

Repository experiment:

```text
/data1/luohaoming/model_feature/experiments_ordered/12_full_text_token_projection_dynamics
```

Large outputs:

```text
/home/luohaoming/model_feature_experiments/full_text_token_projection_dynamics
```

Run all four locally cached checkpoints on GPUs 0–3:

```bash
cd /data1/luohaoming/model_feature/experiments_ordered/12_full_text_token_projection_dynamics
bash scripts/run_all.sh
```

Smoke test one checkpoint with four text-level steps:

```bash
CUDA_VISIBLE_DEVICES=0 /public/luohaoming/model_feature/.venv/bin/python \
  scripts/run_full_text_token_projection.py \
  --config configs/step1000.yaml \
  --output-dir /home/luohaoming/model_feature_experiments/full_text_token_projection_dynamics_smoke/step1000 \
  --text-steps 4
```

Rebuild only the report:

```bash
/public/luohaoming/model_feature/.venv/bin/python \
  scripts/build_full_text_token_projection_report.py \
  --root /home/luohaoming/model_feature_experiments/full_text_token_projection_dynamics \
  --repo-report-dir /data1/luohaoming/model_feature/experiments_ordered/12_full_text_token_projection_dynamics
```
