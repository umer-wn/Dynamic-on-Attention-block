# 早期 spectrum 实验清理记录（2026-07-13）

## 决策

早期静态 block-Jacobian、Lanczos 与跨模型 spectrum 路线已脱离当前以闭环算子、rolling next-token、Frobenius/JVP 和 Lyapunov 为核心的实验主线，因此从服务器工作目录中退役并删除。

## 删除范围

- 专属配置：`*spectrum*`、`*lanczos*`、`pythia_static_128.yaml`、`pythia_global_jacobian_debug.yaml`。
- 专属计算与分析代码：静态 Jacobian spectrum/Lanczos、power-law/seed/cross-model 分析、weight spectral feature 代码。
- 专属计划与结论报告。
- `results/raw`、`results/processed`、`results/figures` 中由这些实验生成的文件。
- 综合可视化的原 phase 01 图、sidecar 和阶段报告。

## 明确保留

- `scripts/compute_perplexity.py`：checkpoint 性能与 rolling 实验仍依赖。
- phase 02–06 的 embedding-feedback、校准、Lyapunov 与 checkpoint 数据和报告。
- rolling next-token 实验、三投影可视化、token-block Jacobian 后续计划。
- 论文综述与方法审计中必要的历史文字；不再保留或链接 spectrum 专属结果资产。

## 兼容性修改

- 综合构图脚本不再调用原 `phase1()`，后续重建只生成原 phase 02–06。
- 主可视化报告删除 phase 01 公式、图表阅读条目、阶段链接和假设台账条目。
- 实验索引将综合报告标记为“original phases 02–06”，并记录本次退役。

## 验证要求

- 服务器上不再存在专属 spectrum/Lanczos/static-128 文件。
- 当前 rolling 与 dynamics 源码不导入已删除的 `src/spectral.py`。
- 主报告的 Markdown 本地链接可解析。
- 不触碰其他既有未提交修改，不停止或删除任何 GPU 实验进程。
