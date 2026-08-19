# Coarse 25-checkpoint full scan

状态：`full_complete`

Canonical data root:

```text
/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan
```

核心产物：

- `raw/step*/run_complete.json`：每个 checkpoint 的 full 完成标记
- `raw/step*/jacobians_controls.jsonl`：self/common-state controls
- `jacobians/step*/`：精确 `512x512` token-level Jacobian 矩阵
- `processed/checkpoint_loss.csv`：固定 128 样本 test loss
- `processed/checkpoint_frobenius*.csv`：Frobenius 聚合结果
- `figures/`：报告图像
- `logs/coarse_full_25.log`：批量执行主日志

判读备注：`step100000` test loss 没有高于 `step16000`，因此当前 coarse scan 不支持“100k 已出现 test-loss 反升/过拟合候选”。
