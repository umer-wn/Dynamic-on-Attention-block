# 07_single_token_frequency_dynamics — 单 token 词频动力学

状态：`full_complete`

按词频层选择 token，比较 isolated/frozen/dynamic context 的单 token dynamics 与 Jacobian。

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

09_pythia_early_training_frobenius_scan
