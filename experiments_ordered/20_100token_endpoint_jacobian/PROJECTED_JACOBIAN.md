# Experiment 20 补充：低维投影 Jacobian 范数

## 定义

沿用 Experiment 16/17 可视化中的四个正交投影方向。令前 `d` 个投影方向组成矩阵 `B_d ∈ R^(d×512)`，在第 1024 个 dynamic step 的状态上计算：

`J_d = B_d J(x_1024) B_d^T`

并报告：

`projected_normalized_frobenius_d{d} = ||J_d||_F / sqrt(d)`，`d=1,2,3,4`。

这与完整空间指标 `||J||_F/sqrt(512)` 使用相同的维数归一化，但含义不同：低维指标只保留“输入扰动在投影子空间中、输出响应也回到同一投影子空间”的分量。它不是完整 Jacobian 范数的近似上界，也不能替代谱半径或稳定性分析。

## 样本与覆盖范围

- token：Experiment 20 固定的 100 个 WikiText-2 频率十分位配对 token；
- dynamic step：1024；
- checkpoint：当前可视化使用的全部 59 个非零 checkpoint；
- 聚合：逐 token 计算后，按 checkpoint 给出 mean/std/median/min/max/SEM/95% CI；
- 投影基：`16_frequency_stratified_window_jacobian/processed/projection_basis.pt`。

## 输出

补充结果并入原 Experiment 20；计算期间的 `/public/luohaoming/model_feature/experiments_ordered/20_100token_endpoint_jacobian_projected` 副本保留为安全镜像。

- `processed/projected_jacobian_parts/step*.csv`：逐 token 数据；
- `processed/projected_jacobian_checkpoint_summary.csv`：低维指标汇总；
- `processed/endpoint100_with_projected_jacobian.csv`：与既有 100-token endpoint feature 合并，可直接上传到 Experiment 17 可视化的“100-token终点/收敛特征 CSV”入口。

## 完成结果

- 59 个 checkpoint × 100 个 token，共 5900 行逐 token 结果；
- 所有 `d=1,2,3,4` 数值均为有限值；
- 与一个 token 的完整 `512×512` Jacobian 直接计算结果比较，四个维度的绝对误差均小于 `4.1e-7`；
- checkpoint 间 100-token 均值范围：
  - `d=1`：0.317802–0.475139；
  - `d=2`：0.329521–0.482786；
  - `d=3`：0.308795–0.477212；
  - `d=4`：0.303726–0.472358；
- checkpoint 41000 的 `d=4` 均值为 0.342525。

`d=4` 低维指标与完整 `||J||F/sqrt(512)` 在 59 个 checkpoint 上的 Pearson 相关系数约为 -0.523。该差异不是两个公式冲突：`B_d J B_d^T` 只保留固定投影子空间内部的输入—输出响应，而完整 Frobenius 还包含投影子空间之外以及跨子空间的全部分量。因此低维 feature 应当与完整范数并排观察，不能作为完整范数的近似或替代。

## 可视化

`17_visualize/dynamic_step_projection_visualization.html` 的“100-token终点（均值）”协议中新增四个可选 feature：

- P1（1D）；
- P1–P2（2D）；
- P1–P3（3D）；
- P1–P4（4D）。

内置数据覆盖 step2000–step61000 的全部 59 个非零 checkpoint；仍可使用双 Y 轴把任一低维指标与完整 Frobenius、谱半径、Lyapunov 或 loss 放在同一图中比较。
