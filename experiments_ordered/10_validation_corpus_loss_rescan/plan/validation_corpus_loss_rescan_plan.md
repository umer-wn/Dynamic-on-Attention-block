# The Pile / Paloma Loss Re-evaluation for Pythia Early Frobenius Scan

状态：`planned`

## 目标

用更接近 Pythia 训练分布与更大验证集的语料重新评估 validation/test loss，并与已有 25-checkpoint single-token Frobenius/Jacobian 结果合并分析。

本实验只重算 loss，不重跑 expensive trajectory/Jacobian/controls。

## 数据源

- `the_pile_validation`: `EleutherAI/pile`, split=`validation`, 512 条非空文档
- `the_pile_test`: `EleutherAI/pile`, split=`test`, 512 条非空文档
- `paloma_representative`: `allenai/paloma` 代表性 configs/domains，每个最多 512 条

Paloma 若 gated 或服务器未授权，不中断实验；写入 `status/paloma_access_blocked.json`。

## Checkpoints

coarse:

```text
step1000, step5000, ..., step97000
```

sentinel:

```text
step0, step16000, step100000
```

## 固定参数

- `sample_count=512`
- `seed=1234`
- `sequence_length=64`
- `loss_batch_size=16`
- `dtype=float32`
- model weights `local_files_only=True`

## 输出

```text
/home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan/
  manifests/
  raw/<source_id>/<checkpoint>/loss.jsonl
  processed/
  figures/
  logs/
  status/
```

## Follow-up plan: coarse-only completion, final checkpoint, and Paloma mirror probe

Scope decision: keep the coarse-grained checkpoint experiment as the canonical result. Do not continue the dense all-100 checkpoint scan unless explicitly requested again.

Updated checkpoint target:

```text
coarse: step1000, step5000, ..., step97000
sentinel: step0, step10000, step16000, step100000, step143000
```

Execution plan:

1. Audit local HF cache for every target revision using direct `refs/snapshots` inspection.
2. If any target checkpoint lacks `config.json` plus `model.safetensors` or `pytorch_model.bin`, try mirror download with `HF_ENDPOINT=https://hf-mirror.com`.
3. Compute missing The Pile validation/test loss only for target revisions with complete local weights.
4. Re-run analysis and plotting.
5. Try Paloma through `hf-mirror.com`:
   - probe `https://hf-mirror.com/api/datasets/allenai/paloma`;
   - probe README/data access;
   - run the Paloma manifest script with `HF_ENDPOINT=https://hf-mirror.com`;
   - if data access remains gated/401, keep Paloma blocked and record the evidence.

Current audited result:

- Coarse/sentinel model weights: 30/30 complete.
- The Pile validation loss: 30/30 complete.
- The Pile test loss: 30/30 complete.
- Paloma mirror status: metadata/README can be reached, but actual dataset access remains gated and requires authentication.

New status files:

```text
status/coarse_checkpoint_loss_completion.json
status/paloma_access_blocked.json
status/final_checkpoint_step143000_complete.json
```

## Follow-up plan: coarse final segment `step100000..step143000`

Motivation: the final checkpoint check found a small rebound from `step100000` to `step143000`. To test whether overfitting begins in this late interval while staying consistent with the existing coarse experiment design, run a 4000-step loss-only scan over the final training segment.

Scope:

```text
target checkpoints:
  step100000
  step101000, step105000, step109000, ..., step141000
  step143000
```

Execution protocol:

1. Audit local HF cache for all 13 target revisions.
2. Download missing model weights for `step101000, step105000, ..., step141000` via `HF_ENDPOINT=https://hf-mirror.com`.
3. A checkpoint is complete only if `config.json` and either `model.safetensors` or `pytorch_model.bin` exist under the resolved snapshot.
4. Compute loss only; do not run Jacobian/Frobenius/trajectory.
5. Use the existing fixed The Pile manifests:
   - `the_pile_validation`
   - `the_pile_test`
6. Keep protocol unchanged:
   - `sequence_length=64`
   - `loss_batch_size=16`
   - `dtype=float32`
   - model loading `local_files_only=True` after download
7. Plot final segment validation/test loss with all sample points/checkpoints, and export CSV.
8. Interpret rebound only as loss evidence; do not infer dynamics/criticality from loss alone.

Initial audit before execution:

- target revisions: 13
- complete local weights: 2 (`step100000`, `step143000`)
- missing local weights: 11 (`step101000, step105000, ..., step141000`)
- existing loss for available endpoints: complete for The Pile validation/test

Correction note: an initial every-1000-step final segment runner was started by mistake. It has been stopped and archived; any non-target downloaded checkpoint from that attempt is excluded from the current 4000-step conclusions.

Planned outputs:

```text
status/final_segment_coarse_loss.pid
status/final_segment_coarse_loss_complete.json
status/final_segment_coarse_missing_weight_revisions.txt
raw/<source_id>/step101000,step105000,...,step141000/loss.jsonl
processed/final_segment_coarse_loss_by_source.csv
figures/final_segment_coarse_loss_curve.png
logs/final_segment_coarse_loss.log
logs/final_segment_coarse_prefetch.log
```

## Follow-up plan: Pile tail-window random manifests

Motivation: the existing Pile manifests were produced by streaming from the beginning of `val.jsonl.zst` / `test.jsonl.zst` and reservoir-sampling early valid rows. If the small Pythia model mainly saw earlier Pile shards during training, this can underestimate late held-out loss. We therefore add tail-window random sampling.

Important audit:

- No local full Pile text corpus was found.
- Large local Arrow caches that looked promising were inspected and are not Pile text; they contain protein/genomic `sequence` fields and must not be used for LLM test loss.
- Therefore the correct approach is to stream the Pile `.jsonl.zst` split and keep only a bounded tail window.

Protocol:

```text
source ids:
  the_pile_validation_tail_random
  the_pile_test_tail_random

implementation:
  stream full compressed split from hf-mirror
  keep only last tail_window valid documents in memory
  sample sample_count documents from that tail window using fixed seed
  write manifest jsonl + metadata
  do not store the full corpus

default parameters:
  sample_count = 2048
  tail_window = 50000
  seed = 4321
  min_chars = 50
```

Planned script:

```text
scripts/prepare_pile_tail_manifests.py
```

Planned outputs:

```text
manifests/the_pile_test_tail_random.jsonl
manifests/the_pile_test_tail_random.metadata.json
manifests/the_pile_validation_tail_random.jsonl
manifests/the_pile_validation_tail_random.metadata.json
status/the_pile_*_tail_random_prepare_status.json
```

After manifest creation, compute loss on the same checkpoint set as the coarse/final-segment experiment. Tail-window loss should be reported separately from the earlier front/reservoir manifest results.
