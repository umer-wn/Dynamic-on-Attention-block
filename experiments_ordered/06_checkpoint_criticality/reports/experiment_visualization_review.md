# 分期实验可视化与研究审计报告

## 审计结论

本项目当前采用可校准、可复核的 LLM embedding-feedback 动力学实验框架。当前最可靠的发现不是“LLM 普遍位于混沌边缘”，而是：**相位结论强烈依赖所构造的闭环算子；Frobenius 平均增益不能替代最大 Lyapunov 指数；对当前 direct operator，Pythia-70M 从 step0 的混沌候选转向训练末期的稳定定点。**

这与核心论文 arXiv:1909.05176 的对应关系是“方法论的 LLM 版本”：将同维反馈算子迭代、稳定点/吸引子、扰动增长和 Lyapunov 判据迁移到语言模型表示空间。它不是对原论文具体网络、训练规则或理论结论的逐项复现，也不能据此宣称原生 autoregressive LLM 动力学已被证明处于临界状态。

从文本、mask、embedding、反馈循环、JVP 到 raw/processed/figure 的逐字段数据流，见独立文档：[LLM 临界性实验数据流技术详解](experiment_dataflow_technical_guide.md)。

## 先澄清：报告中的 step 到底是什么

本报告同时出现两种完全不同的 step，必须分开理解。

### 1. training step / checkpoint

`step0`、`step1000`、`step16000`、`step143000` 是 Pythia 官方预训练过程保存的权重 checkpoint。它们表示优化器更新大约进行了 0、1000、16000、143000 次。**本项目没有重新训练四个模型，也没有在动力学实验中更新任何模型权重。**实验只是从本地缓存加载同一个 Pythia-70M 架构在四个训练时刻的冻结权重：

$$
\theta_0,\;\theta_{1000},\;\theta_{16000},\;\theta_{143000}.
$$

因此，横轴“training step”回答的是“预训练权重随训练进度怎样变化”，不是一次轨迹内部的循环次数。

### 2. dynamics step / eval step

对某个固定 checkpoint $\theta_s$ 和一个 WikiText 样本，先把 token 转成初始 embedding 张量 $x_0$，再冻结权重并循环执行同一个算子：

$$
x_{t+1}=F_{\theta_s}(x_t),\qquad t=0,1,2,\ldots
$$

这里的 $t$ 才是轨迹图中的 `eval step`、`step_index` 或 return map 中的时间。每一次循环都把上一步得到的连续 hidden-state 张量作为下一次 Transformer 的 `inputs_embeds`；没有采样 token、没有反向传播、没有 optimizer step。它是一套人为构造的、用于研究的闭环动力系统，不等同于通常的 autoregressive 文本生成。

`burn-in=512` 表示先循环 512 次但不纳入主统计，用来尽量消除初始文本 embedding 的暂态；`eval=128` 或三投影复测中的 `eval=256` 表示随后保存和分析的轨迹长度。

## 实际计算的 LLM 闭环算子

设冻结的 causal Transformer 为 $M_\theta$，输入连续张量 $x\in\mathbb{R}^{L\times d_h}$，attention mask 为 $m$。实现中先将 padding 位置清零，再调用：

```text
model(inputs_embeds=x, attention_mask=m, output_hidden_states=True, use_cache=False)
```

### 原始映射 $G_\theta$

主实验使用 `target: final_hidden`：

$$
G_\theta(x)=H_L(x),
$$

其中 $H_L$ 是最后一层 hidden state，形状仍为 $L\times d_h$，所以可以反复送回同一个模型。

替代实验使用 `target: embedding_expectation`：先由 logits 得到温度 $T$ 下的 token 概率，再对输入 embedding 矩阵 $E$ 求期望：

$$
p_t=\operatorname{softmax}(\operatorname{logits}_t/T),\qquad
G_{\theta,T}(x)_t=p_tE.
$$

它同样保持 $L\times d_h$，但算子含义与 `final_hidden` 不同，因此两者的临界性不能直接混为一个模型固有属性。

### update mode

代码先应用输出倍率 $\beta$：$\widetilde G_\theta(x)=\beta G_\theta(x)$，再按 `operator_update` 生成 $F$：

| mode | 实际公式 | 含义 |
|---|---|---|
| `direct` | $F(x)=\widetilde G_\theta(x)$ | 主实验；最后层 hidden 直接回灌 |
| `residual` | $F(x)=(1-\alpha)x+\alpha\widetilde G_\theta(x)$ | 在恒等映射和模型映射之间插值 |
| `norm_matched` | $F(x)=\widetilde G_\theta(x)\|x\|/(\|\widetilde G_\theta(x)\|+\varepsilon_n)$ | 强制下一状态范数与当前状态相同 |
| `residual_norm_matched` | 先 residual mixing，再把范数匹配到 $\|x\|$ | 同时控制方向更新和范数 |

`residual_alpha=0` 时严格得到 $F(x)=x$，因此解析真值是 Frobenius=1、Lyapunov=0、扰动分离比=1；这就是阶段五的恒等锚点。`residual_alpha=1` 才恢复 direct raw map（当 $\beta=1$）。

## 状态、扰动和 Jacobian 指标的定义

以下范数都是对非 padding 的 active dimensions 计算。若序列有 $L_a$ 个有效 token、hidden size 为 $d_h$，则有效维数 $D=L_a d_h$。

### 轨迹收敛量

$$
\text{state\_norm}_t=\|x_t\|_2,
$$

$$
\text{step\_delta}_t=\|x_t-x_{t-1}\|_2,
$$

$$
r_t=\text{relative\_step\_delta}_t=
\frac{\|x_t-x_{t-1}\|_2}{\max(\|x_t\|_2,10^{-12})}.
$$

实现用最后 5 个 $r_t$ 的平均值与 `convergence_tol=1e-6` 比较。低于阈值只能说明“fixed-like”，即数值上接近定点；它本身不证明轨道的 Lyapunov 稳定性，因此报告还同时检查扰动和 Lyapunov。

### nearby trajectory

从相同 $x_0$ 构造一个随机单位方向 $u$：

$$
x'_0=x_0+\epsilon u,\qquad \|u\|_2=1,
$$

随后不重归一化地同时迭代 $x_t$ 与 $x'_t$：

$$
d_t=\|x'_t-x_t\|_2.
$$

有限窗口辅助量为：

$$
R_d=d_T/d_0,\qquad
g_d=\frac{1}{T}\log(d_T/d_0).
$$

它直观但不是最大 Lyapunov：距离可能进入非线性区、饱和、下溢，也可能在 float32 中被舍入误差主导。本项目据 epsilon sensitivity 选择 `epsilon=1e-3`，并把 nearby 图降为辅助证据。

### normalized Frobenius / RMS Jacobian gain

令当前状态的闭环 Jacobian 为：

$$
J_t=\frac{\partial F_\theta(x_t)}{\partial x_t}.
$$

对 Rademacher probe $v_i\in\{-1,+1\}^D$ 用 JVP 计算 $J_tv_i$。因为 $\mathbb E\|J_tv\|_2^2=\|J_t\|_F^2$，实现估计：

$$
\rho_t=sqrt{\frac{1}{D K}\sum_{i=1}^{K}\|J_tv_i\|_2^2}
\approx\frac{\|J_t\|_F}{\sqrt D}.
$$

$\rho_t=1$ 表示奇异值的均方根约为 1，不表示最大奇异值为 1，也不保证长期 Lyapunov 为 0。跨状态主汇总采用几何平均：

$$
\rho_{\mathrm{geo}}=\exp\left(\frac1T\sum_t\log\max(\rho_t,10^{-12})\right).
$$

阶段六 step0 的 $\rho_{\mathrm{geo}}<1$ 但最大 Lyapunov 为正，正是“平均方向收缩但特殊切向方向长期扩张”的实例。

### 最大有限时间 Lyapunov（Benettin/JVP）

初始化单位切向量 $v_0$，随后沿实际轨道逐步执行：

$$
w_t=J_tv_t,
$$

$$
a_t=\|w_t\|_2,
$$

$$
v_{t+1}=w_t/a_t.
$$

最终估计：

$$
\widehat\lambda_{\max}=\frac1T\sum_{t=0}^{T-1}\log a_t.
$$

代码对多个随机 probe 分别计算，再报告 mean/std/max。每步重归一化避免了 nearby distance 的饱和问题，也让切向量逐渐对齐主扩张方向。这里仍是有限时间、有限样本估计：$\widehat\lambda>0$ 是混沌候选证据，$\widehat\lambda<0$ 支持局部收缩，接近零时必须报告不确定性而不能强行二分。

### recurrence、return map 和投影 Poincaré

lag-$k$ recurrence distance 为：

$$
D_k=\frac{1}{T-k}\sum_{t=0}^{T-k-1}\|x_{t+k}-x_t\|_2,
$$

热图使用 $D_k/\overline{\|x_t\|_2}$。小值表示相隔 $k$ 步的状态接近，但既可能是周期回归，也可能只是定点收敛，必须结合 relative delta 判断。

三个固定随机单位投影记为 $q_0,q_1,q_2$：

$$
z_i(t)=\langle x_t,q_i\rangle.
$$

return map 画 $z_0(t)$ 对 $z_0(t+1)$。Projected Poincaré Section 先取每条轨迹 $z_0$ 的中位数 $c$，只保留向上穿越：

$$
z_0(t-1)\le c<z_0(t),
$$

然后画 $(z_1(t),z_2(t))$。这只是三维投影上的近似截面，不是完整高维流形上的严格 Poincaré map。

### language-model loss 和 perplexity

checkpoint performance 使用正常 teacher-forced causal language modeling，只作为“模型训练到什么程度”的外部性能坐标，不参与反馈循环。若有效 next-token 数为 $N$：

$$
\mathcal L_{\mathrm{token}}=-\frac1N\sum_{i=1}^{N}\log p_\theta(w_i\mid w_{<i}),
\qquad
\mathrm{PPL}=\exp(\mathcal L_{\mathrm{token}}).
$$

报告按预测 token 数加权，而不是先算每个样本 loss 再做不加权平均。

## checkpoint 主实验的完整参数

| 参数 | 当前值 | 技术含义 |
|---|---:|---|
| model | `EleutherAI/pythia-70m` | 同一架构，不同官方训练 checkpoint |
| checkpoints | 0/1000/16000/143000 | 预训练优化步；本项目不训练新权重 |
| dataset | WikiText-2 validation | 用真实文本只生成初始 $x_0$ 和 attention mask |
| samples | 8 | 四个 checkpoint 使用同一批样本 |
| sequence length | 64 | 每个初始状态最多 64 token |
| target | `final_hidden` | $G_\theta(x)=H_L(x)$ |
| operator update | `direct` | $F(x)=G_\theta(x)$，$\alpha$ 不参与主实验 |
| token mode | `nonpad_flattened` | 只在非 padding token×hidden dimensions 上计算范数/JVP |
| dtype | float32 | 解释了极小 nearby distance 的数值边界 |
| random seed | 1234 | 数据抽取、探针和投影的可复现起点 |
| burn-in | 512 | 不纳入主统计的预迭代 |
| long eval | 128 | Frobenius/Lyapunov checkpoint 主统计窗口 |
| visualization eval | 256 | 三投影轻量复测轨迹窗口 |
| epsilon | `1e-3` | nearby 初始扰动范数 |
| convergence tol | `1e-6` | 最后 5 个 relative delta 均值的 fixed-like 门槛 |
| Frobenius states/probes | 4/4 | 从 4 个轨道状态、每状态 4 个 Rademacher JVP 估计 |
| Lyapunov probes | 2 | 两条独立随机切向量的 Benettin 估计 |
| lag windows | 1,2,4,8,16,32,64 | recurrence 检查的时间间隔 |
| projection bank | 3, seed 1234 | checkpoint/sample 间共享的三个固定随机方向 |

三投影复测将 Frobenius/Lyapunov probes 设为 0，是为了只补充轨迹几何、不重复昂贵且已有的主指标；因此投影图应与 long-asymptotic 的 Lyapunov 表联合阅读。

## 20 张动力学图的阅读指南

| 阶段/图 | 横纵轴怎样看 | 主要回答 | 不应怎样解释 |
|---|---|---|---|
| P2 early diagnostics | state norm—step delta；时间—nearby distance | 早期轨迹为何促成方法重构 | historical nearby 不是最大 Lyapunov |
| P2 early return map | $z_t$—$z_{t+1}$，颜色为时间 | 单投影轨迹是否趋点/成带 | 单投影形状不能确认高维吸引子 |
| P3 cross-model Frobenius | 模型—平均 RMS Jacobian gain，黑点为样本 | 同一构造算子是否跨模型一致 | 模型样本数不同，不能只比柱高得出普遍规律 |
| P3 operator comparison | update mode—Frobenius，黑点为样本 | 算子定义对结果影响多大 | residual≈1 是人工控制，不是模型天然临界 |
| P3 temperature sweep | 温度—Frobenius/旧 nearby/settled fraction | embedding expectation 是否受温度控制 | 中间图是 historical finite-distance 指标 |
| P4 output gain | $\beta$—Frobenius | 简单输出缩放是否提供干净控制轴 | 非线性与归一化使关系不必线性 |
| P4 epsilon sensitivity | epsilon—Frobenius/nearby log growth | AD/JVP 与有限差分谁更稳 | 误差条跨零时不能按均值符号定相位 |
| P5 residual calibration | $\alpha$—Lyapunov/Frobenius/separation | 实现能否恢复恒等和收缩真值 | 只校准测量机制，不证明主算子唯一合理 |
| P5 Frobenius–Lyapunov | RMS gain—最大 Lyapunov | 控制组内两个指标是否同向 | 控制组相关不代表一般等价 |
| P6 performance | training step—loss/PPL，symlog 横轴 | 官方 checkpoint 的语言建模性能如何改善 | 与 dynamics step 无关 |
| P6 Lyapunov samples | training step—样本级 $\lambda$，零线为相位参考 | 扩张方向随训练如何变化 | step1000 接近零，应视为未决 |
| P6 Frobenius–Lyapunov | 四个 checkpoint 的两指标散点 | proxy 是否出现直接反例 | step0 反证当前算子中的等价，不反证所有理论 |
| P6 short-vs-long | checkpoint—尾部 relative delta，对数轴 | burn-in 不足会怎样误判暂态 | 非固定不自动等于混沌 |
| P6 relative-step trajectories | eval step—$r_t$，对数轴 | 哪些轨迹趋向定点 | 平台也可能是非固定有界轨道 |
| P6 recurrence heatmap | lag—checkpoint，颜色为归一化 $D_k$ 的 log10 | 有无短周期/近回归 | 小距离也可能仅由定点收敛产生 |
| P6 nearby trajectories | eval step—$d_t$，对数轴 | 有限扰动扩张还是收缩 | 会饱和/舍入，只作辅助证据 |
| P6 three-projection trajectories | sample0 的 $(\Delta z_0,\Delta z_1,\Delta z_2)$，相对最终状态 | 同一轨迹如何接近最终投影状态 | 坐标跨度必须结合标题数量级，不能把自动放大的抖动当大轨道 |
| P6 return maps | sample0 的 $\Delta z_0(t)$—$\Delta z_0(t+1)$ | 同一轨迹是弥散还是趋向固定返回点 | 一维 return map 依赖投影方向 |
| P6 Projected Poincaré | sample0 向上穿越的插值 $(\Delta z_1,\Delta z_2)$ | 同一轨迹的截面点是否收缩 | step143000 多 crossing 是定点附近数值抖动，不是周期 |
| P6 per-sample centered Poincaré | 8 个 sample 分别减去自己的最终投影状态 | 排除不同固定点位置后，各轨迹内部是否收缩 | 8 个 sample 使用不同 median-z0 截面，不是共同 Poincaré map |

## 当前动力学证据链

1. [论文方法重构与早期动力学](experiment_phases/phase_02_paper_refactor.md)：建立闭环算子和早期轨迹诊断，同时更正旧 product metric 的解释。
2. [跨模型与替代算子](experiment_phases/phase_03_operator_exploration.md)：证明 operator choice 会显著改变所谓“临界状态”。
3. [数值与方法校准](experiment_phases/phase_04_calibration.md)：定位 float32 nearby-distance 边界，并检验显式控制量。
4. [修正后的 Lyapunov 重测](experiment_phases/phase_05_lyapunov_correction.md)：以恒等映射和收缩映射校准 Benettin Lyapunov。
5. [训练 checkpoint 临界性](experiment_phases/phase_06_checkpoint_criticality.md)：在统一协议下比较 step0、1000、16000、143000。

## checkpoint 主结果

| checkpoint | loss | PPL | Frobenius | 最大 Lyapunov | 扰动末/初比 | 相位审计标签 |
|---|---:|---:|---:|---:|---:|---|
| step0 | 10.9877 | 59144.99 | 0.6429 | +0.01154 | 12343 | 混沌候选 |
| step1000 | 5.7499 | 314.15 | 0.6559 | -0.00515 | 863.5 | 临界附近未决 |
| step16000 | 4.7425 | 114.73 | 0.4590 | -0.0181 | 0.270 | 慢收敛稳定候选 |
| step143000 | 4.6974 | 109.66 | 0.4640 | -0.2661 | 0.182 | 稳定定点 |

![checkpoint 样本级 Lyapunov](assets/experiment_visualization/phase_06__checkpoint-lyapunov-samples.png)

step0 是关键反例：其 Frobenius 小于 1，但最大 Lyapunov 为正且扰动显著扩张。因此，平均随机方向增益小于 1 不足以判定稳定相位。

![Frobenius 与 Lyapunov](assets/experiment_visualization/phase_06__checkpoint-frobenius-vs-lyapunov.png)

## 稳定点和 Poincaré 图如何判断

稳定点判断使用三类互补量：相邻迭代的 relative step delta 是否持续趋近数值零；扰动是否收缩；最大 Lyapunov 是否为负。仅凭“轨迹看起来聚成一点”或短 burn-in 后的单次距离不能判定稳定点。step143000 同时满足强负 Lyapunov、扰动收缩和极小 relative delta，因此标为稳定定点；step16000 的 Lyapunov 为负但仍有慢收敛尾部，因此只标为稳定候选。

本轮补采三个跨 checkpoint、跨样本共享的固定随机投影。审计发现原三维图只画 sample0，而原二维图混合 8 个 sample，并且各 sample 使用不同的 median-z0 截面；后期样本间不同固定点的位置差异因此掩盖了单轨道收敛。现已将主三维、return map 和 Projected Poincaré 统一为 sample0，相对最终投影状态中心化，并对 crossing 做线性插值。step16000 显示阻尼式收缩；step143000 的单轨道跨度仅约 `1.72e-5`，截面内波动约 `2.56e-6`，属于固定点附近数值抖动，不是周期轨道证据。另图将 8 个 sample 分别中心化，只作 sample 内收敛诊断。完整证据见 [Phase 6 三投影与 Projected Poincaré 差异审计](phase6_poincare_projection_audit.md)。

![Projected Poincaré Sections](assets/experiment_visualization/phase_06__checkpoint-projected-poincare.png)

![Per-sample centered Projected Poincaré diagnostics](assets/experiment_visualization/phase_06__checkpoint-poincare-per-sample-centered.png)

## 假设台账

| 假设 | 当前状态 | 支持图 | 反证图 | 限制 |
|---|---|---|---|---|
| direct embedding-feedback 可作为论文方法的 LLM 操作化版本 | 有条件支持 | phase02 operator flow；phase05 controls | phase03 operator dependence | 是人为构造闭环，不是原生生成过程 |
| Frobenius≈1 等价于零 Lyapunov | 已证伪（当前算子） | phase05 控制组内近似一致 | phase06 step0 反例 | 不外推到所有动力系统 |
| 训练使当前算子趋向混沌边缘 | 当前证伪 | step1000 接近零 | phase06 后期强负 Lyapunov | 只有四个 checkpoint、单一模型族 |
| step0 为混沌 | 候选支持 | 正 Lyapunov、扩张、非固定 | 尚缺更长窗口/更多 seed | 需平稳性和收敛误差审计 |
| step143000 为稳定定点 | 支持 | 负 Lyapunov、收缩、relative delta | 无 | 限当前样本和 direct operator |
| 原生 LLM 普遍处于临界状态 | 未决 | 无 | 当前构造算子后期偏稳定 | 尚未定义唯一“原生同维闭环” |

## 重大缺陷与更正

- 旧 `product_log_gain` 是沿随机方向增益乘积，不是 Benettin 最大 Lyapunov；只作为灰色 historical 诊断保留。
- 旧 nearby-ratio 受有限差分、饱和与 float32 舍入影响，不再承担主相位判定。
- 早期以 Frobenius 或单一投影形状直接命名“临界”的表述已收紧。
- 固定点收敛和混沌不是同一个问题：非固定轨道不能仅因“不收敛到点”就被叫作混沌。
- Poincaré 图是三维随机投影上的近似截面，不声称恢复完整高维吸引子拓扑。

## 下一步

最有价值的下一轮不是继续增加装饰性图，而是做预注册的相位判别：增加 checkpoint 密度和样本/投影 seed；为 Lyapunov 报告窗口收敛与置信区间；比较至少两种有理论动机的同维闭环算子；增加 surrogate controls 与维度/归一化消融。若 direct operator 的相位转变在这些检验下保持，才能讨论它是否对应训练中的可复现动力学转变。

完整图、派生 CSV、sidecar 和 manifest 位于 `/home/luohaoming/model_feature_reports/experiment_visualization_review/`；精选图位于 `reports/assets/experiment_visualization/`。
