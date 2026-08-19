# Rolling next-token 动力学可视化补充计划

状态：完成  
日期：2026-07-13  
对应主实验：`plan/transformer_paper_validation_small_experiments_plan.md`  
对应主报告：`reports/rolling_next_token_criticality_report.md`

## 1. 目标

在不重新运行模型、不重复 JVP、保持原始 JSONL 只读的前提下，为已完成的 rolling next-token 主实验补充专用可视化与中文技术指南。重点回答：

1. rolling 算子的完整数据流、输入输出尺寸和各参数含义；
2. full-window 与 newest-token 投影分别观察什么；
3. Frobenius、innovation Jacobian、Lyapunov、nearby separation 和 hard cycle 的公式关系；
4. 每张图怎样读、能支持什么、不能支持什么；
5. 如何避免旧图中的样本混合、独立自动缩放和 sample-specific Poincaré 截面造成的误判。

## 2. 数据来源

只读输入：

- behavior summaries/trajectories：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/main_behavior/*/raw/`
- tangent summaries/trajectories：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/main_tangent/*/raw/`
- 已有 processed tables：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/main_processed/`

不下载权重，不执行 Transformer forward/JVP，不使用 GPU。

## 3. 新增图表

### 3.1 Soft trajectory diagnostics

四 checkpoint 的 32 条 behavior 轨迹，按 eval step 展示：

- nearby distance 中位数与 IQR，对数纵轴；
- relative step delta 中位数与 IQR，对数纵轴；
- soft entropy 与 top-1 probability 中位数/IQR。

所有曲线显示样本离散度，不只画 checkpoint 均值。

### 3.2 Hard-cycle distribution

展示每个 checkpoint 的 32 个样本：

- exact full-window cycle length；
- cycle start；
- 未在 512 步内检出的样本用单独标记，不伪装成长度 512 的周期。

### 3.3 Full-window 与 newest-token 三投影

使用同一个匹配 anchor `doc264@0`，四 checkpoint 分面；分别展示：

\[
z_i^{\mathrm{full}}(t)=\langle q_i,\operatorname{vec}(X_t)\rangle,
\qquad
z_i^{\mathrm{newest}}(t)=\langle r_i,x_{t,-1}\rangle.
\]

三维图明确标记起点、终点和时间方向；每个面板标注坐标跨度，防止自动缩放把微小抖动画成大线团。

### 3.4 Return maps

对同一个 anchor 分别绘制：

\[
z_0(t)\mapsto z_0(t+1).
\]

不跨 sample 混合，以免不同样本的固定点中心被误读成单样本多周期。

### 3.5 改进的 Projected Poincaré Sections

对 full/newest 各自使用一个在四 checkpoint 间共享的截面值：

\[
c=\operatorname{median}\{z_0(t):\text{matched anchor, all checkpoints}\}.
\]

只记录向上穿越，并在线段 `(t,t+1)` 上做线性插值：

\[
\alpha=\frac{c-z_0(t)}{z_0(t+1)-z_0(t)},
\]

\[
(z_1^*,z_2^*)=(1-\alpha)(z_1(t),z_2(t))+\alpha(z_1(t+1),z_2(t+1)).
\]

若没有 crossing，明确标注“该共享截面对当前轨迹不具判别力”，不补点、不改为 sample-specific median。

## 4. 输出

- 分析脚本：`scripts/build_rolling_next_token_visualization_supplement.py`
- 技术指南：`reports/rolling_next_token_visualization_guide.md`
- 精选图：`reports/assets/rolling_next_token/`
- 全分辨率图、派生 CSV 和 manifest：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/visualization_supplement/`

## 5. 说明文档固定结构

1. 研究对象和不变量；
2. rolling 算子逐步数据流与张量尺寸；
3. 参数总表；
4. 各指标公式推导；
5. 每张图的数据来源、横纵轴、样本聚合方式；
6. 实际结果解释；
7. 允许与禁止的结论；
8. 与论文 `1909.05176` 的对应关系和剩余 mismatch。

## 6. 验收

- 四 checkpoint behavior 均为 32 anchors、256 eval rows/anchor；tangent 均为 8 anchors。
- 投影图只使用一个明确 anchor，不混合 sample。
- full/newest 投影方向在 checkpoint 间一致；字段完整且无 NaN。
- Poincaré crossing 用共享截面、线性插值，并输出 crossing CSV。
- 图中标明 sample、operator、burn/eval、轴含义和是否使用 log scale。
- 人工检查无标签裁切、错误对数轴、自动缩放误导或颜色歧义。
- manifest 中所有源文件、派生 CSV 和图路径存在。
- 报告不得把 `probes=0` 写成 Frobenius/Lyapunov 数值为 0。

## 7. 执行结果

- 已生成 6 张新图、6 份派生 CSV、manifest 和 build log；
- behavior/tangent 数量与匹配 token windows 验证通过；
- 投影图只使用 `doc264@0`，未跨 sample pooling；
- shared-absolute 与 centered Poincaré 分开呈现，crossing 使用线性插值；
- full/newest 三投影面板打印三轴 span，避免独立自动缩放误导；
- 参数、公式、Jacobian 范围和逐图说明写入 `reports/rolling_next_token_visualization_guide.md`；
- 现有数据无法补算 last-token Jacobian，后续实验已写入 `plan/rolling_token_block_jacobian_followup_plan.md`。
