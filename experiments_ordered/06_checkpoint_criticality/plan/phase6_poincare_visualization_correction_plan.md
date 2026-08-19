# Phase 6 Poincaré 可视化更正计划

## 问题

原三维图只显示 `sample0`，原二维 Projected Poincaré 图却叠加全部 8 个 sample；同时每个 sample 使用各自的 `median(z0)` 截面。不同样本的固定点位置差异因此被误画成一个宽散点云，不能与 sample0 三维轨迹直接比较。

## 更正

1. 保留三维与二维的相同 checkpoint、相同 `sample0`、相同三投影方向。
2. 三维轨迹相对该 sample 的最终投影状态中心化，并标记起点、终点和真实坐标跨度。
3. Poincaré crossing 在相邻离散状态之间线性插值，不再直接取 crossing 后的离散点。
4. 主二维图只显示 `sample0`，按 crossing 顺序着色并相对最终投影状态中心化。
5. 另生成“8 sample 分别中心化”的诊断图，用于分离 sample 内收敛与 sample 间固定点差异；明确声明它不是共同截面。
6. return map 同样改为 sample0 并相对最终投影状态中心化。

## 验证

- 覆盖无 crossing、单 crossing 插值和周期 crossing 单元测试。
- 检查四个 checkpoint 的 crossing 数与原始数据一致。
- 图名和报告明确区分同一样本主图与多样本中心化诊断图。
- 不把固定点附近 float32 抖动解释为渐进周期或极限环。
