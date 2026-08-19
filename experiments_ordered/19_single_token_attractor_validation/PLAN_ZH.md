# 实验19：单 Token 吸引子验证——方法、指标、进度与恢复计划

## 1. 实验目标与证据链

本实验检验冻结后的 Pythia-70M 单 token 动力映射

\[
x_{t+1}=F_c(x_t)
\]

是否存在可复现、数值可靠且局部稳定的固定点或周期吸引子。实验不根据二维/四维投影直接分类，而是在完整512维 hidden state 中依次建立以下证据：

```text
不同初始状态库
  → 长轨迹回归筛选
  → 多点射击求解周期轨道
  → Floquet局部稳定性
  → 有限扰动恢复
  → FP32/FP64精度一致性
  → 最终分类
```

本轮实验执行范围截止到有限扰动恢复，即第四阶段。FP32/FP64精度审计和架构controls不再作为本轮必须执行内容。只有同时通过回归、射击收敛、Floquet稳定性和有限扰动恢复，才在本轮报告中称为“具有吸引性的稳定周期轨道”；已有精度审计结果作为附加数值警告保留。

## 2. 初始状态库

每个 checkpoint 使用8个按词频分层选择的 token，并从三个状态库分别启动，因此每个 checkpoint 有 `3 × 8 = 24` 条轨迹。

| 状态库 | 初始状态 | 研究目的 |
|---|---|---|
| `native` | 当前 checkpoint 自身的8个 token embedding | 检查真实模型 embedding 出发的动力学 |
| `common_step0` | 同一批 token 在 step0 模型中的 embedding | 分离“模型权重变化”和“初始 embedding 变化” |
| `random_matched` | 512维随机高斯向量，L2归一化后匹配 step0 embedding 的典型范数 | 判断循环结构是否仅对语言 embedding 有效，还是随机状态也能进入 |

三个状态库只改变初始条件 \(x_0\)，不改变当前 checkpoint 的映射 \(F_c\)。如果来源差异较大的状态最终进入同一轨道，才构成更强的吸引域证据。

## 3. 第一阶段：回归筛选

“回归”指 recurrence，不是监督学习中的 regression。筛选阶段对每条轨迹运行4096步，分析后2048步，并检查候选 lag \(p=1,\ldots,256\)。

对每个 lag 计算完整512维距离：

\[
d_p(t)=\|x_{t+p}-x_t\|_2.
\]

归一化指标为：

\[
\hat d_p^{95}=
\frac{Q_{0.95}(\|x_{t+p}-x_t\|_2)}{S_{\mathrm{orbit}}},
\]

其中 \(S_{\mathrm{orbit}}\) 是轨迹中随机状态对距离的95%分位，并且不小于FP32数值噪声下限。采用95%分位是为了避免把偶然靠近的两个点误判成周期轨道。

### 3.1 筛选标签与阈值

- 固定点候选 `fixed_candidate`：

  \[
  \hat d_1^{95}\le10^{-5}.
  \]

- 周期回归候选 `recurrent_candidate`：

  - 最佳 lag \(p>1\)；
  - 最佳归一化P95误差 \(\le0.05\)；
  - 次优误差/最佳误差 \(\ge1.1\)，即周期峰具有足够显著性；
  - 轨迹尾部前半段和后半段得到相同最佳 lag。

- 准周期候选 `quasiperiodic_candidate`：最佳误差 \(\le0.1\)，但两个时间窗口的最佳 lag 不一致。

- 扩张候选 `expanding_candidate`：有限时间最大 Lyapunov 指数 \(>0.01\)，且最佳回归误差 \(>0.05\)。

- 其他情况：`transient_or_unresolved`。

回归筛选只产生候选，不证明存在精确周期轨道。

## 4. 第二阶段：多点射击

若筛选给出候选周期 \(p\)，取候选轨道的 \(p\) 个状态：

\[
X=(x_0,x_1,\ldots,x_{p-1}).
\]

精确周期轨道应满足：

\[
F(x_i)=x_{i+1},\qquad F(x_{p-1})=x_0.
\]

实验使用LBFGS同时优化所有轨道点，最小化：

\[
L=\frac{1}{pS_{\mathrm{orbit}}^2}
\sum_{i=0}^{p-1}\|F(x_i)-x_{i+1\bmod p}\|_2^2.
\]

主要验收指标为归一化射击残差P95：

\[
r_{\mathrm{shoot}}^{95}=Q_{0.95}
\left(\frac{\|F(x_i)-x_{i+1}\|_2}{S_{\mathrm{orbit}}}\right).
\]

优化器内部报告收敛的阈值约为：

\[
r_{\mathrm{shoot}}^{95}\le\max(\sqrt{10^{-9}},10^{-6})
\approx3.16\times10^{-5}.
\]

最终分类采用更严格阈值：

\[
r_{\mathrm{shoot}}^{95}\le10^{-5}.
\]

射击收敛说明附近能求得满足周期方程的数值解，但不代表该周期轨道稳定或具有吸引性。

## 5. 第三阶段：Floquet局部稳定性

对周期为 \(p\) 的轨道，考虑一周映射的Jacobian乘积：

\[
M=J_F(x_{p-1})\cdots J_F(x_1)J_F(x_0).
\]

其最大特征值模

\[
\rho(M)=\max_i|\lambda_i(M)|
\]

表示无穷小扰动经过一个周期后的放大比例。实验通过JVP和Arnoldi方法隐式估计，不显式构造512×512矩阵；分别使用Krylov维度16和32，并检查两次估计是否一致。

| Floquet标签 | 判据 |
|---|---|
| `stable` | 32维估计 \(\rho(M)<0.98\)，且16/32维估计相对差异 `<5%` |
| `unstable` | \(\rho(M)>1.02\) |
| `boundary` | 其余情况，包括谱半径接近1或Krylov估计不一致 |

Floquet稳定性是局部线性稳定性，不足以单独证明有限扰动会回到轨道。

## 6. 第四阶段：有限扰动恢复

对每条候选周期轨道，在轨道相位0附近施加：

- 16个随机方向；
- 相对尺度 \(10^{-6},10^{-4},10^{-2}\)；
- 共48个扰动状态；
- 演化10个完整周期。

距离使用到周期轨道所有相位的最小距离：

\[
d(x,\mathcal O)=\min_i\|x-x_i^{\mathcal O}\|_2,
\]

避免由于沿极限环发生相位平移而误判为远离轨道。若最终相位不变距离不大于初始距离，则该扰动记为恢复：

\[
R=\frac{\text{恢复扰动数}}{48}.
\]

稳定周期轨道的最终分类要求：

\[
R\ge0.9.
\]

## 7. 附加检查：FP32/FP64精度一致性（不在本轮执行范围）

对每个被审计 checkpoint，选择 `native` 状态库中归一化射击残差最小的收敛轨道。保留FP32射击残差后，将模型转为FP64，从周期轨道第一点出发演化4个完整周期，并计算：

\[
e_{64}=\max_{k=1,\ldots,4}
\frac{\|F_{64}^{kp}(x_0)-x_0\|_2}{S_{\mathrm{orbit}}}.
\]

精度一致条件为：

\[
e_{64}\le\max(10r_{\mathrm{shoot,32}}^{95},10^{-4}).
\]

如果FP32求出的周期轨道在FP64下不再闭合，则该结构可能依赖舍入误差或有限精度，只能标为数值周期候选。

分类完全使用512维状态，`classification_uses_projection=False`；二维/四维投影只用于展示。

## 8. 本轮最终分类规则（截止第四阶段）

| 最终标签 | 必要条件 |
|---|---|
| `stable_fixed_point` | 最小周期1、射击残差≤1e-5、Floquet stable、恢复率≥0.9 |
| `stable_periodic_orbit(p)` | 周期>1、射击残差≤1e-5、Floquet stable、恢复率≥0.9 |
| `unstable_periodic_orbit(p)` | 周期>1、射击残差≤1e-5、Floquet unstable |
| `numerical_cycle` | 附加精度检查明确失败；仅作警告，不要求本轮重算 |
| 候选/未解析标签 | 只通过筛选或缺少后续证据 |

本轮主结论依据前四阶段。已有 `precision_consistent=False` 必须在报告限制中披露，因此本轮使用“稳定周期轨道/具有吸引性的周期轨道”，不使用“已通过高精度验证的吸引子”这一更强表述。

### 8.1 不动点与周期轨道的区分

- 不动点满足 \(F(x^*)=x^*\)，最小周期为1；筛选时要求 `lag1_normalized_p95 ≤ 1e-5`。
- 周期轨道满足 \(F^p(x^*)=x^*\) 且最小周期 \(p>1\)。
- 射击得到候选周期后，使用 `minimal_repeated_period` 检查候选是否可约化为更短周期；若最终最小周期变成1，则归为不动点，而不是极限环。
- 本系统是离散时间神经网络迭代，严格名称应为“周期轨道”。“极限环”仅作为直观称呼；它还要求该周期轨道通过Floquet和有限扰动恢复，表现出吸引性。
- 当前288条射击解的最小周期全部大于1，周期分布为51、57、100、101、203；没有 `minimal_period=1`，筛选表中也没有 `fixed_candidate`。因此当前候选不是不动点。

## 9. 当前实验进度与结果

### 9.1 已完成

- 19个 checkpoint 的筛选完成；
- 每个 checkpoint 有24条轨迹；
- 总计 `19 × 3 × 8 = 456` 条筛选记录。

筛选结果：

| 类别 | 数量 |
|---|---:|
| transient/unresolved | 234 |
| recurrent candidate | 141 |
| expanding candidate | 73 |
| quasiperiodic candidate | 8 |

候选轨道验证：

- `orbit_candidates.csv`：288条；
- `floquet_metrics.csv`：288条；
- 当前有轨道输出的 checkpoint：10000、29000、41000、57000；
- 288条均报告多点射击收敛。

Floquet结果：

| 标签 | 数量 |
|---|---:|
| stable | 51 |
| boundary | 191 |
| unstable | 46 |

精度审计：

- step10000、29000、41000、57000各1条；
- 4条均为 `precision_consistent=False`。

### 9.2 当前可支持的结论

实验发现了大量完整512维状态上的回归候选，部分候选可以通过多点射击得到低残差周期解。但大部分Floquet结果处于边界区，且4个代表性轨道均未通过FP32/FP64一致性检验。因此当前只能结论为：

> 检测到数值周期候选，但尚未验证出数值精度可靠的稳定吸引子。

### 9.3 本轮截止第四阶段的结果

- 288条候选均达到射击残差阈值，说明存在低残差周期解；按三相位重复去重后对应96个 `(checkpoint, state_bank, token)` 系统。
- 去重后的Floquet标签为：16个stable、66个boundary、14个unstable。
- 有限扰动恢复率在全部288条记录中的范围为0到0.5417，中位数为0；没有轨道达到 `recovery_fraction ≥ 0.9`。
- 因此截止第四阶段，当前没有满足“稳定且具有有限扰动吸引性”的周期轨道。现有结果应称为周期解或周期候选，而不是已验证的极限环吸引子。

### 9.4 稠密 checkpoint 简化扩展（已完成）

#### 9.4.1 目的与取舍

为覆盖当前已有周期特征的更多 checkpoint，本扩展复用实验25的周期筛选结果，不再对每个模型运行完整4096步、3个状态库和8个token。它是用于定位重点 checkpoint 的低成本协议，不能取代第1–9节的完整实验。

#### 9.4.2 checkpoint 与 token 选择

1. 对每个 checkpoint 汇总固定展示 token `clones / motive / cabinet / miles` 的候选周期。
2. 仅保留候选周期中位数大于1的 checkpoint；另保留原实验已有的step10000。
3. 在该 checkpoint 的非平凡周期 token 中，选择候选周期误差最小的一个作为代表。
4. 共选择21个 checkpoint：`10000、15000、19000、22000、29000、30000、32000、36000、38000、40000、41000、42000、43000、44000、45000、47000、48000、50000、53000、57000、60000`。
5. step27000因只有1/4 token给出非平凡周期、checkpoint中位数仍为1而排除。step32000和43000存在token间周期分歧，结果必须标注为token依赖。

#### 9.4.3 周期闭合与稳定性

- 使用已有1024步代表轨迹的后512步，按候选周期构造相位质心。
- 沿用LBFGS多点射击、最小周期检查和归一化残差定义。
- 优化器内部收敛阈值约为 `3.16e-5`，最终报告仍只把 `归一化P95残差≤1e-5` 记为严格通过。
- 只有射击收敛后才计算Krylov维度16/32的Floquet乘子；联合结论只采用严格闭合候选的Floquet标签。

#### 9.4.4 修改后的256步扰动协议

对每个代表 token 在动态步768的完整512维状态施加扰动：

- 8个可复现随机单位方向；每个方向同时包含512个维度的分量；
- 3个相对尺度：`1e-6、1e-4、1e-2`；
- 共24个扰动；
- 扰动与未扰动参考轨迹均继续运行256步。

单次响应增益定义为

\[
g_i=\frac{\|\widetilde x_{1024}^{(i)}-x_{1024}\|_2}
{\|\widetilde x_{768}^{(i)}-x_{768}\|_2}.
\]

主统计量是24个 `g_i` 的算术平均：小于1表示平均缩小，大于1表示平均放大。同时保存几何平均、中位数、最小/最大值、平均log增益、收缩方向占比和逐尺度统计。该指标跟随未扰动参考轨迹处理漂移，但不做周期相位最小化，因此不得与原来的相位不变恢复率混为同一指标。

#### 9.4.5 完成状态与验收

- 21/21个checkpoint完成；
- 21条checkpoint汇总记录；
- `21 × 8 × 3 = 504`条逐扰动记录；
- 严格射击通过：step22000、44000、48000；
- step48000为唯一同时满足严格闭合、Floquet stable和256步算术平均增益小于1的候选；
- 结论已整合进 `REPORT_ZH.md`，并明确它是单代表token的简化协议结果。

输出文件：

- `processed/dense_periodic_checkpoint_selection.csv`
- `processed/dense_periodic_checkpoint_summary.csv`
- `processed/dense_periodic_checkpoint_perturbation_256.csv`
- `raw/dense_extension_orbits/`
- `scripts/run_dense_periodic_extension.py`

### 9.5 不再纳入本轮执行的内容

架构controls和新的FP32/FP64精度审计不再纳入本轮实验范围。此前controls中的有限时间Lyapunov/JVP路径曾出现：

```text
RuntimeError: expected mat1 and mat2 to have the same dtype,
but got: float != c10::Half
```

尚需系统完成的控制包括：

- 仅保留MLP residual；
- 仅保留V/O residual；
- 去内部LayerNorm；
- 去最终LayerNorm；
- 去residual；
- layer shuffle；
- 随机初始化模型。

在这些controls完成前，不能判断循环候选主要由残差结构、LayerNorm、层顺序还是训练权重产生。

## 10. 本轮收尾计划（截止第四阶段）

1. 核对前四阶段的完整condition manifest和结果主键，保留现有成功结果。
2. 按三相位重复去重，使用每个 `(checkpoint, state_bank, token)` 中射击残差最小的解作为主结果。
3. 分别汇总最小周期、射击残差、Floquet谱半径/Krylov一致性和恢复率。
4. 明确区分 `fixed point`、`periodic orbit`、`stable periodic orbit` 和 `attracting periodic orbit`。
5. 更新中英文最终报告；精度失败和未完成controls只列为限制，不继续执行第五阶段或controls。

## 11. 验收标准

- 前四阶段所有纳入报告的输出均为有限值，主键无重复；
- Floquet 16/32维估计差异显式保留；
- 每个最终结论都可追溯到processed表格的具体行；
- 最小周期1与最小周期大于1明确分开汇总；
- 只有同时满足回归、射击、Floquet stable和恢复率≥0.9的轨道才标为本轮“具有吸引性的稳定周期轨道”；
- 不继续执行FP32/FP64精度阶段和architecture controls。
