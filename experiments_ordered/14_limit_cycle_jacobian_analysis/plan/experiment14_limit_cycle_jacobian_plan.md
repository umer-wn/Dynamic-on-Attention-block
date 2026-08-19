# 实验 14：单 token 周期、收敛词与局部 Jacobian 谱计划

状态：`planned`，尚未运行  
日期：2026-07-27  
上游实验：`13_single_token_convergence_neighbors`

## 0. 范围与风险

本实验同时包含长时间动力学、精确 `512×512` Jacobian 和特征值。
正式规模包含 440 个 final-step Jacobian；这是一次较大的 GPU 实验。
采用 `smoke → dynamics scan → Jacobian gate` 三阶段，前一阶段通过后才
启动下一阶段。

实验继续使用实验 13 的 isolated-token operator：

\[
x_{t+1}=f_\theta(x_t)
=\operatorname{GPTNeoXBody}_\theta(x_t)[-1],
\qquad x_t\in\mathbb R^{512}.
\]

不调用 tokenizer sampling、LM softmax 或 autoregressive generation。LM head
只用于最后的 token 解释。

## 1. 研究问题

1. step5000–step77000 中哪些 checkpoint/token 真正形成固定点、周期轨道、
   准周期/长暂态或非周期运动？
2. 若存在极限环，其基本周期是多少？周期估计在完整 512 维空间中是否成立？
3. 最终状态或周期轨道在 input-embedding cosine、input-embedding Euclidean
   和 LM output-head 三种空间中分别最接近哪个 token，分离度如何？
4. 最后 10 个 dynamic steps 的局部 Jacobian 谱半径与第二特征值如何变化？
5. 逐 checkpoint 的 2D 投影，以及 step9000/step16000 的 3D 投影如何展示
   完整 512 维周期判据发现的结构？

## 2. Checkpoint 与 token 设计

### 2.1 Checkpoint 网格

采用基础网格，并保留已有的 step9000、step16000 相变附近观测点：

```text
step5000, step9000, step13000, step16000, step21000, step29000,
step37000, step45000, step53000, step61000, step69000, step77000
```

两个已观察到的相位边界原计划各在左右 `±2000 steps` 补一点：

```text
step13000 boundary: step11000, step15000
step37000 boundary: step35000, step39000
```

这 4 个加密 revision 当前均未下载。按用户要求，它们连同权重下载一起顺延到
下一实验；实验 14 不下载、不做参数插值，也不为缺失点留伪数据。因此实验 14
正式 checkpoint 数为 12，全部使用已缓存真实权重。下一实验再单独处理：

```text
step11000, step15000, step35000, step39000
```

### 2.2 Token

- 主 cohort：复用实验 13 固定 seed `20260727` 随机清单中的前 4 个 token；
- token 清单与 WikiText-2 count 复制到实验 14 manifest；
- 不按词频重新分层，不重新随机抽样，保证 checkpoint 横向匹配；
- smoke 阶段只用其中 2 个 token；主动力学阶段使用全部 4 个；
- 固定 token 为 `' repetitive'`、`' semi'`、`' evidence'`、
  `' orientations'`；本实验不再检验 token 抽样或词频效应。

## 3. 长轨迹协议

为直接观察 dynamic-step window 对呈现结果的影响，修订后的协议为：

- dtype：模型 forward 为 float32；距离与后处理转 float64；
- total steps：2048；
- 从初始 input embedding（dynamic step0）开始记录；
- recorded trace：step0–2048，共 2049 个状态；
- recurrence evaluation 仍使用 step512–2048；
- 保存完整 trace states：每个 token 为 `[2049, 512]`，而不只保存投影；
- 新轨迹写入独立的 `trace0_2048/` 数据目录，不覆盖早先的
  step512–2048和step2048–4096 states；
- 固定随机投影先 QR 正交化，并跨 checkpoint/token 共用；
- 初始暂态单独画 `[0,512]`；
- 后续 trace 按 128 次状态转移分成 12 个窗口：
  `[512,640]`、`[640,768]`、…、`[1920,2048]`，合计13张汇总图。

同时记录：

- `||x_t||_2`；
- `||x_{t+1}-x_t||_2`；
- relative step delta；
- fixed 2D/3D projections；
- tail 前半/后半的 norm、diameter、recurrence score 变化。

## 4. 周期估计：量纲与判据

### 4.1 尺度

对 evaluation tail `X={x_t}` 定义有量纲的轨道直径：

\[
D_{\max}=\max_{i,j}\|x_i-x_j\|_2.
\]

同时报告对异常点更稳健的尺度：

\[
D_{95}=Q_{0.95}\bigl(\|x_i-x_j\|_2\bigr),
\]

其中 pairwise distance 用固定 seed 抽样，避免 `O(T^2)` 存储。主归一化尺度：

\[
S=\max(D_{95},\ 8\epsilon_{32}\,\operatorname{median}\|x_t\|_2,\ 10^{-12}).
\]

`D_max`、`D95` 和数值 floor 都单独输出，避免归一化掩盖绝对量级。

### 4.2 Lag recurrence

对候选周期 `p` 定义：

\[
r_p(t)=\frac{\|x_{t+p}-x_t\|_2}{S}.
\]

对每个 `p=1..Pmax` 报告：

- median `r_p`；
- 95th percentile `r_p`；
- max `r_p`；
- absolute recurrence distance；
- tail 前半和后半的 recurrence 是否继续下降。

初始 `Pmax=256`。若 recurrence/autocorrelation 在边界附近出现候选峰，扩展
到 512。候选基本周期必须满足：

1. `p>1`；
2. median 与 P95 recurrence 同时低于预注册阈值；
3. 至少覆盖 4 个重复周期；
4. `2p、3p` 也闭合；
5. `p` 的真因子不满足同等闭合，避免把倍周期误报为基本周期；
6. 完整 512 维判据通过；不使用投影结果参与分类。

阈值不在看图后随意选择。smoke 阶段用数值 floor 校准后固定两档：

- strict：P95 normalized recurrence `≤1e-4`；
- approximate：P95 `≤1e-2` 且后半段不再系统下降。

### 4.3 分类优先级

1. 若 `D95` 不高于 float32 resolution floor：`numerical_fixed_point`；
2. 若 `p=1` recurrence 通过且 step delta 接近 floor：`fixed_point`；
3. 若 `p>1` 全维 recurrence 通过：`limit_cycle_candidate(period=p)`；
4. 若 recurrence 随时间仍下降：`long_transient`；
5. 否则为 `nonperiodic_or_unresolved`。

不会仅凭二维闭合曲线标注极限环。

## 5. 投影图

每个 128-step window 输出一张 3×4 固定 2D 投影汇总图：

- 12 个 checkpoint 各占一个子图，4 个 token 同图；
- 所有 checkpoint 共用同一组正交投影方向；
- 每个子图以对应 checkpoint/window 的 float64 centroid 为原点；
- 每个 checkpoint 子图独立自适应放大，但两个投影轴保持相同单位比例，
  避免几何形状失真；
- 子图标明 x/y 实际上下界和 projected `r95`；
- 对接近数值精度的轨迹设置 `1e-6` 最小显示半径，避免无限放大数值抖动；
- 起点为 `○`、终点为 `×`，子图标注投影半径 `r95`；
- 禁用 Matplotlib additive offset。

对 `step9000` 和 `step16000` 额外输出 3D 固定随机投影。三个方向先 QR
正交化，两个 checkpoint 使用完全相同的方向。投影只用于展示，周期分类
仍以完整 512 维 recurrence 为准。

## 6. 最终/周期 token 解释

### 6.1 分析对象

- 所有轨迹：最终状态 `x_T`；
- fixed point：尾窗中心；
- limit cycle：周期内每个 phase state，以及 phase centroid；
- 周期 token 结果同时报告逐 phase top-1 一致率，避免最终 phase 任意性。

### 6.2 Cosine input-embedding neighbor

\[
s_i=\frac{x^\top e_i}{\|x\|_2\|e_i\|_2},
\qquad d_{\cos}=1-s_i.
\]

输出：top-5、top1 similarity、cosine distance、top1−top2 cosine margin、
top-1 相对全词表的 z-score、WikiText-2 count。

这里的“可信度”是几何分离度，不是概率：

- margin 越大，最近邻排名越稳；
- z-score 越大，top-1 相对词表背景越突出；
- 仍不得称为模型 prediction。

### 6.3 Euclidean input-embedding neighbor

\[
d_i=\|x-e_i\|_2.
\]

输出：top-5、`d1`、absolute gap `d2-d1`、relative gap
`(d2-d1)/max(d1,eps)`、distance z-score、WikiText-2 count。

Euclidean 距离受 hidden/input-embedding norm mismatch 强烈影响。因此同时输出
`||x||`、`||e_i||`，并明确它与 cosine neighbor 可能不同。Euclidean margin
只是排名分离度，不是概率。

### 6.4 LM-head top-1

\[
z=W_{out}x,\qquad p=\operatorname{softmax}(z).
\]

输出：top-5 token/probability、top1 probability、top1−top2 logit margin、
probability margin、normalized entropy、WikiText-2 count。

三种 top-1 单独呈现，并增加：

- cosine neighbor 是否等于 Euclidean neighbor；
- cosine/Euclidean neighbor 是否等于 LM-head top-1；
- 周期 phase 内 top-1 switching count 与 modal token fraction。

## 7. 最后 10 步 Jacobian 谱

对每条轨迹最后 10 个 map inputs `x_{T-10}..x_{T-1}` 计算：

\[
J_t=Df_\theta(x_t)\in\mathbb R^{512\times512}.
\]

注意 `J_t` 是非对称、非正规矩阵。“最大特征值”预注册为按模排序：

\[
\rho_1=|\lambda_1(J_t)|,\qquad
\rho_2=|\lambda_2(J_t)|.
\]

同时保存 complex eigenvalue 的 real/imag/angle，不能只保存绝对值。为了避免
把 eigenvalue 与瞬时最大伸长混淆，附带报告 top-2 singular values
`sigma1/sigma2`。

核心图：

1. checkpoint × final-10-step 的 `|lambda1|` 热图；
2. 同图叠加或并排 `|lambda2|`；
3. `|lambda1|`、`|lambda2|` 的 checkpoint 中位数与 token dispersion；
4. `sigma1` 对照，注明 non-normality 差异。

### Jacobian 成本门控

完整主网格为 `11 checkpoints × 4 tokens × 10 = 440` 个精确 Jacobian。
执行顺序：

1. smoke：2 checkpoints × 2 tokens × 2 steps；
2. pilot：所有 checkpoint × 1 固定 token × 10 steps；
3. 若 eigensolver、显存和 runtime 通过，再扩到 4 token；
4. 每个 matrix 写 checksum，支持断点续跑，不重复计算。

## 8. 阶段与通过条件

### Phase A：实现与 smoke

- 2 checkpoint：一个 fixed candidate、一个 moving candidate；
- 2 token；512 dynamic steps 的缩短轨迹；
- 校验 recurrence、Jacobian shape 和 eigenvalue 排序；
- 用 finite difference/JVP spot check 校验 Jacobian。

通过条件：无 NaN/Inf；Jacobian `512×512`；JVP 相对误差在预注册容差内；
合成固定点/周期序列单元测试可恢复已知 period。

### Phase B：全 checkpoint 长轨迹与周期筛选

- 先运行 11 checkpoint × 4 token 的 batch trajectory；
- 输出分类、周期候选和规定的 2D/3D projection 图；

### Phase C：最后 10 步 Jacobian

- pilot 后再决定是否扩到全部 token；
- 聚合 top-2 eigen/singular values；原始矩阵仅在诊断需要时保存。

## 9. 预期目录与产物

```text
experiments_ordered/14_limit_cycle_jacobian_analysis/
  README.md
  RUNBOOK.md
  plan/experiment14_limit_cycle_jacobian_plan.md
  configs/
  manifests/checkpoint_grid.json
  manifests/random8_tokens.jsonl
  scripts/run_limit_cycle_scan.py
  scripts/compute_final_jacobian_spectra.py
  scripts/build_experiment14_report.py
  processed/trajectory_summary.csv
  processed/period_candidates.csv
  processed/final_token_neighbors.csv
  processed/final10_jacobian_spectrum.csv
  processed/projection_window_summary.csv
  figures/checkpoint_projection_window_step*.png
  figures/step9000_projection_3d.png
  figures/step16000_projection_3d.png
  figures/period_recurrence_heatmap.png
  figures/final10_jacobian_top2.png
```

大型原始 states/Jacobians 放实验数据根，不提交仓库；仓库只保存 config、
manifest、processed tables、报告和必要图。

## 10. 结论边界

- 二维/三维闭环不是极限环证据，完整 512 维 recurrence 才是主判据；
- float32 numerical fixed point 不等于解析意义的精确 fixed point；
- input-embedding nearest token 不等于 LM-head prediction；
- 单 token、长度 1 的 operator 结论不能直接外推到正常多 token 生成；
- 4 个随机 token、单 seed 不用于建立 token 或词频规律；它们只作为匹配的
  初始条件重复，用于检查动力学分类是否对初始 token 基本一致。
