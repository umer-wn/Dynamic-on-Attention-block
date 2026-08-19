# 分期实验可视化与研究审计报告计划

## 目标与交付

按六阶段整理现有实验，补采三路固定投影轨迹，生成统一图表、sidecar、manifest、中文主报告和阶段详报。

- 主报告：`reports/experiment_visualization_review.md`
- 阶段详报：`reports/experiment_phases/phase_01_*.md` 至 `phase_06_*.md`
- 全量图/派生表：`/home/luohaoming/model_feature_reports/experiment_visualization_review/`
- 精选报告图：`reports/assets/experiment_visualization/`

## 六阶段

1. 静态 Jacobian 与谱：smoke/static、random subspace、Lanczos、seed stability、跨模型谱。
2. 论文方法重构：early dynamical-edge、metrics、projection64；旧 product/nearby 指标仅作为历史诊断。
3. 跨模型与替代算子：Pythia/GPT-2/Qwen、long burn-in、norm-matched、residual、embedding expectation、温度扫描。
4. 数值与方法校准：output gain、identity/residual、epsilon sensitivity。
5. 修正 Lyapunov：论文对齐审计、Benettin 重测、Frobenius-Lyapunov 差异。
6. 训练 checkpoint：step0/1000/16000/143000、burn64/512、性能、相位、recurrence 和多投影相图。

## 多投影复测

新增配置：

```yaml
trajectory_projection_bank:
  mode: fixed_random
  count: 3
  seed: 1234
  shared_across_samples: true
```

输出 `projection_0..2` 与 `projection_0_next..2_next`。四 checkpoint 使用相同 8 个样本、seq64、burn-in512、eval256、direct operator、epsilon 1e-3；不重复 Frobenius、Lyapunov 或旧 product 计算。数据写入 `/home/luohaoming/model_feature_experiments/pythia_checkpoint_criticality/visualization_rerun`。

投影 Poincare section 使用 `z0` 向上穿越每条轨迹的中位截面，在截面绘制 `(z1,z2)`；同时绘制三维轨迹和 return maps。图名明确标为 projected/approximate。

## 图表与证据规范

- 混沌/正 Lyapunov：橙色三角；未决：绿色菱形；稳定/负 Lyapunov：蓝色圆形；历史失效指标：灰色虚线。
- Lyapunov 固定零线；Frobenius 固定参考线1；训练步 symlog；距离/relative delta 用 log 轴。
- 聚合值必须附样本点或误差条。
- 每张图写 sidecar 和 manifest：源路径、过滤条件、样本量、轴、问题、允许解释、caveat、证据状态。
- 主报告包含假设台账，并明确静态谱、构造反馈动力学和原生生成动力学的边界。

## 验收

- 三投影确定性、跨 checkpoint 一致性、字段完整性测试。
- Poincare crossing 覆盖无/单次/周期 crossing。
- 主图不把旧 `product_log_gain` 当最大 Lyapunov。
- 所有 manifest 源文件和报告图链接存在。
- 每阶段至少人工检查一张主图。
- 最终运行 dynamics 测试并确认无残留 GPU 进程。
