# 09_pythia_early_training_frobenius_scan — Pythia early training Frobenius scan

状态：`full_complete`

Pythia-70M early checkpoints 的 test loss、single-token trajectory、精确 token-level Jacobian/Frobenius 粗扫描。

## Canonical layout

- `plan/`：实验计划与后续计划
- `reports/`：阶段报告、审计报告、结论报告
- `scripts/`：该实验直接使用的脚本入口或脚本副本
- `smoke/`：smoke/pilot/profile/gate 小规模实验索引
- `full/`：正式全参实验索引
- `manifests/`：路径迁移、数据和图表 manifest

## 数据策略

大数据仍保留在 `/home/luohaoming/model_feature_experiments` 或原始实验目录；本目录只保存索引和小文件副本，不删除旧路径。

## Superseded by

N/A
