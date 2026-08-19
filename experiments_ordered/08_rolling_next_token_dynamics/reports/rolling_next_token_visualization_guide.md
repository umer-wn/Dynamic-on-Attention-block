# Rolling next-token 动力学：可视化、参数与公式技术指南

状态：完成  
日期：2026-07-13  
主实验计划：`plan/transformer_paper_validation_small_experiments_plan.md`  
可视化计划：`plan/rolling_next_token_visualization_supplement_plan.md`  
主报告：`reports/rolling_next_token_criticality_report.md`  
token-block 后续：`plan/rolling_token_block_jacobian_followup_plan.md`

## 1. 实验循环的对象

这组实验没有训练新模型。`step0/1000/16000/143000` 是 Pythia-70m 的训练 checkpoint；图中的 dynamics/eval step 是冻结同一 checkpoint 后再次执行 rolling 算子，不是 optimizer step。

从 WikiText validation 的长文档截取 64 个 token：

\[
I_0\in\mathbb N^{1\times64},
\qquad
X_0=E[I_0]\in\mathbb R^{1\times64\times512}.
\]

每个 soft dynamics step 为：

\[
h_t=\operatorname{GPTNeox}_\theta(X_t)_{-1},
\]

\[
\ell_t=W_{\rm out}h_t,
\qquad
p_t=\operatorname{softmax}(\ell_t/T),
\]

\[
e_{t+1}=p_tE\in\mathbb R^{1\times512},
\]

\[
X_{t+1}=F_\theta(X_t)
=[X_t[:,1:,:],e_{t+1}]
\in\mathbb R^{1\times64\times512}.
\]

```text
token IDs [1,64]
  -> embedding lookup
X_t [1,64,512]
  -> causal Transformer，读取最后位置
h_t [1,512]
  -> LM head -> logits [1,V] -> softmax(T)
p_t [1,V]
  -> p_t @ embedding table
e_new [1,512]
  -> 与 X_t[:,1:,:] 拼接
X_(t+1) [1,64,512]
```

### 1.1 自回归、mask 与 padding

- 每次只使用最后位置预测下一 token，是 causal next-token operator。
- Transformer 内部并行计算 64 个位置，但 causal mask 阻止看到未来。
- 每个样本单独运行且固定 64 token，没有 padding；attention mask 全为 1。
- 每次 position IDs 重置为 `0..63`，使 \(F_\theta\) 成为自治映射；它不是绝对位置持续增长的原生无限生成。

### 1.2 soft 与 hard 路径

soft 路径使用：

\[
e_{t+1}=\sum_{v=1}^Vp_t(v)E_v.
\]

它通常不是任何真实 token embedding，但保持可微，可计算 JVP、Frobenius 和 Lyapunov。

hard 路径使用：

\[
y_{t+1}=\arg\max_vp_t(v),
\qquad
I_{t+1}=[I_t[:,1:],y_{t+1}].
\]

它对应 greedy generation，用于完整 token-window 精确周期检测；`argmax` 不可微，因此不计算 Jacobian。

## 2. 核心参数

| 参数 | Behavior layer | Tangent layer | 含义 |
|---|---:|---:|---|
| checkpoints | 4 | 4 | 冻结权重 |
| documents | 8 | 8 | 跨 checkpoint 匹配 |
| anchors/document | 4 | 1 | behavior 32 samples；tangent 8 samples/checkpoint |
| window \(L\) | 64 | 64 | rolling token 数 |
| hidden \(H\) | 512 | 512 | embedding 维度 |
| state dimension \(D=LH\) | 32768 | 32768 | full state 维度 |
| temperature | 1.0 | 1.0 | softmax 温度 |
| position mode | reset | reset | 每轮 `0..63` |
| dtype | float32 | float32 | nearby 小距离受精度限制 |
| burn-in | 512 | 256 | 不进入正式统计的预迭代 |
| eval | 256 | 128 | behavior 画轨迹；tangent 算 Lyapunov |
| hard steps | 512 | skipped | hard 完整窗口周期检测 |
| epsilon | \(10^{-3}\) | \(10^{-3}\) | nearby 初始扰动范数 |
| Frobenius states | skipped | 最后连续 4 states | 当前局部抽样的限制 |
| Frobenius probes | 0/skipped | 8/state | 0 表示没算，不是范数为 0 |
| Lyapunov probes | 0/skipped | 2 | Benettin 随机切向量数 |
| projection bank | 3 full + 3 newest | 同定义 | seed 1234，共享方向 |

## 3. 当前 Jacobian 的范围

当前同时报告两个范围，但没有计算最后输入 token block。

### 3.1 Full seq→seq Jacobian

\[
J_F(X_t)
=\frac{\partial\operatorname{vec}(X_{t+1})}
{\partial\operatorname{vec}(X_t)}
\in\mathbb R^{32768\times32768}.
\]

对应 `total_local/total_geomean/shift_fraction_of_total_squared`。它接近 1，主要来自 63 个 token 的机械复制。

### 3.2 新 token 输出对完整 sequence 输入

\[
J_{\rm new}
=\frac{\partial e_{t+1}}
{\partial\operatorname{vec}(X_t)}
\in\mathbb R^{512\times32768}.
\]

对应 `innovation_geomean` 与 `innovation_output_geomean`；二者是同一矩阵，只是归一化分母不同。

### 3.3 尚未计算的 last-token block

\[
J_{\rm last}
=\frac{\partial e_{t+1}}{\partial x_{t,L}}
\in\mathbb R^{512\times512}.
\]

`projection_newest_*` 只是状态内积，不是 Jacobian。现有 JSONL 没有完整连续状态或分 token JVP，不能后处理反推；新实验见 `plan/rolling_token_block_jacobian_followup_plan.md`。

## 4. 为什么 seq Frobenius 天然接近 1

将新 token Jacobian 按输入 token 分块：

\[
J_{\rm new}=[J_1,J_2,\ldots,J_L],
\qquad
J_\ell=\frac{\partial e_{t+1}}{\partial x_\ell}
\in\mathbb R^{H\times H}.
\]

rolling Jacobian 为：

\[
J_F=
\begin{bmatrix}
0&I&0&\cdots&0\\
0&0&I&\cdots&0\\
\vdots&&&\ddots&\vdots\\
0&\cdots&0&0&I\\
J_1&J_2&\cdots&J_{L-1}&J_L
\end{bmatrix}.
\]

因此：

\[
\|J_F\|_F^2=(L-1)H+\|J_{\rm new}\|_F^2.
\]

full normalization 与 shift-only 基线为：

\[
\rho_{\rm total}=\frac{\|J_F\|_F}{\sqrt{LH}},
\qquad
\rho_{\rm shift}=\sqrt{\frac{L-1}{L}}.
\]

当 \(L=64\)：

\[
\rho_{\rm shift}=\sqrt{63/64}=0.9921567416.
\]

即使新 token 完全不依赖输入，full seq normalized Frobenius 也约为 0.992。

innovation 两种归一化为：

\[
\rho_{\rm innovation,total}
=\frac{\|J_{\rm new}\|_F}{\sqrt{LH}},
\]

\[
\rho_{\rm innovation,output}
=\frac{\|J_{\rm new}\|_F}{\sqrt H}
=\sqrt L\,\rho_{\rm innovation,total}.
\]

它们满足：

\[
\rho_{\rm total}^2
=\frac{L-1}{L}
+\rho_{\rm innovation,total}^2
=\frac{L-1}{L}
+\frac{\rho_{\rm innovation,output}^2}{L}.
\]

## 5. Frobenius、Lyapunov 与 nearby

对 Rademacher probe \(v\)，有：

\[
\mathbb E\|Jv\|_2^2
=\operatorname{tr}(J^\top J)
=\|J\|_F^2.
\]

所以：

\[
\|J\|_F^2
\approx\frac1K\sum_{k=1}^K\|Jv_k\|_2^2.
\]

当前 tangent Frobenius 使用最后连续 4 states × 8 probes，是轨道末端局部抽样，不是完整吸引子时间平均。周期/混沌候选应改用均匀状态抽样。

Benettin Lyapunov 每步执行：

\[
u_{t+1}=J_t\widehat v_t,
\quad
a_t=\|u_{t+1}\|_2,
\quad
\widehat v_{t+1}=u_{t+1}/a_t,
\]

\[
\widehat\lambda_{\max}
=\frac1T\sum_{t=0}^{T-1}\log a_t.
\]

Frobenius 是单步奇异值平方平均；Lyapunov还包含跨 step 的方向对齐与 Jacobian 连乘，二者不能互换。

Nearby 路径定义：

\[
X'_0=X_0+\epsilon\eta/\|\eta\|,
\qquad
d_t=\|X'_t-X_t\|_2.
\]

它不重归一化，所以扩张时会饱和、收缩时会碰到 float32 数值地板，只是 Lyapunov 的有限扰动辅助证据。

Relative step delta 为：

\[
r_t=\frac{\|X_{t+1}-X_t\|_2}{\|X_{t+1}\|_2}.
\]

\(r_t\to0\) 支持固定点；非零但规律可能是周期；持续大且不规则支持活跃非固定轨道。

分布指标：

\[
H(p_t)=-\sum_vp_t(v)\log p_t(v),
\qquad
p_{\max}(t)=\max_vp_t(v).
\]

高 entropy/低 top1 表示 soft expectation 混合许多词表 embedding；低 entropy/高 top1 表示 soft 路径接近 hard argmax。

## 6. 图一：checkpoint 总览

![checkpoint overview](assets/rolling_next_token/main_training_dynamics_overview.png)

左上显示 8 个 tangent anchors 的样本级 Lyapunov 和 checkpoint 均值；零线是相位参考。右上是 innovation-output Frobenius。左下是 32 个 behavior anchors 在 512 hard steps 内发现完整窗口周期的比例。右下比较 soft entropy 与 top1。

| checkpoint | Lyapunov mean ± std | 正 Lyapunov | innovation-output | hard cycle |
|---|---:|---:|---:|---:|
| step0 | -0.02555 ± 0.00052 | 0/8 | 0.3083 | 0/32 |
| step1000 | 0.01781 ± 0.05548 | 6/8 | 0.3490 | 32/32 |
| step16000 | 0.03071 ± 0.08042 | 4/8 | 0.2830 | 32/32 |
| step143000 | 0.06684 ± 0.09927 | 5/8 | 0.2785 | 32/32 |

训练使相位从 step0 的一致收缩转成正负混合，但没有单调停在 \(\lambda=0\)。均值不能替代样本点。

## 7. 图二：Jacobian 分解

![tangent decomposition](assets/rolling_next_token/main_tangent_decomposition.png)

左图把 total normalized Frobenius 与 Lyapunov 对照。虚线是 shift-only 0.9921567；所有 total 点都紧贴该基线，即使 Lyapunov 从负变正。右图比较 innovation-output 与 Lyapunov，它去除了 shift identity，但仍不能代替长期方向乘积。

四 checkpoint 的 shift 平方贡献为 99.67%–99.85%，因此禁止把 `total≈1` 当作临界证据。

## 8. 图三：soft 轨迹诊断

![soft diagnostics](assets/rolling_next_token/rolling_soft_trajectory_diagnostics.png)

每条实线是 32 anchors 的中位数，阴影是 IQR；横轴是 burn-in 512 后的 256 eval steps。Nearby 和 relative delta 使用对数纵轴。

| checkpoint | nearby：eval0→255 | relative delta：eval0→255 | entropy：eval0→255 | top1：eval0→255 |
|---|---:|---:|---:|---:|
| step0 | 2.42e-8→1.46e-8 | 1.03e-4→6.26e-6 | 10.625→10.625 | 0.000247→0.000246 |
| step1000 | 1.42e-4→7.81e-4 | 0.855→0.868 | 5.353→5.366 | 0.182→0.191 |
| step16000 | 0.936→0.060 | 4.89e-5→9.43e-7 | 8.642→8.642 | 0.0307→0.0307 |
| step143000 | 3.68→3.77 | 1.058→1.092 | 2.790→0.333 | 0.296→0.970 |

解释：

- `step0` 很快进入近均匀词表的 soft 平均态并收缩；这不是语言能力良好的稳定点。
- `step1000` nearby 仍较小，但轨道本身持续移动，说明“附近轨道彼此接近”和“收敛到固定点”不是同一件事。
- `step16000` 中位轨道趋近固定点，但 IQR 很宽，说明 anchor 异质性强；这也解释 tangent 样本仍有正 Lyapunov。
- `step143000` 分布逐渐极尖锐、relative delta 约为 1、nearby 维持有限大值，支持活跃且输入依赖的非固定轨道。

Behavior 与 tangent 的 burn-in、样本覆盖以及 nearby/renormalized 指标不同，不能要求中位曲线与 Lyapunov 均值逐点相等。

## 9. 图四：hard exact cycles

![hard cycles](assets/rolling_next_token/rolling_hard_cycle_distributions.png)

左图只绘制实际检测到的周期长度，未检测样本不伪装成 512；右图是首次完整窗口重复前的 transient 长度。

| checkpoint | detected | cycle median [IQR] | cycle-start median |
|---|---:|---:|---:|
| step0 | 0/32 | 未检测 | 未检测 |
| step1000 | 32/32 | 4 [3, 7.25] | 70 |
| step16000 | 32/32 | 11.5 [6, 27.5] | 69 |
| step143000 | 32/32 | 13 [7.75, 22.25] | 82 |

这里检测的是 64-token 完整窗口精确重复，不只是相邻 token 重复。一旦窗口重复，reset-position、greedy argmax 的确定性保证后续永久循环。

hard 周期不能直接决定 soft Lyapunov，因为 soft expectation 与 hard argmax 是两个不同算子。

## 10. 图五：full-window 与 newest-token 三投影

![rolling projections](assets/rolling_next_token/rolling_projection_trajectories_full_vs_newest.png)

这张图只使用一个跨 checkpoint 完全匹配的 anchor：`doc264@0`，没有混合 sample。绿色圆点是 eval 起点，橙色叉是终点，viridis 颜色随时间增加。

Full-window 投影：

\[
z_i^{\rm full}(t)
=\langle q_i,\operatorname{vec}(X_t)\rangle,
\qquad q_i\in\mathbb R^{32768}.
\]

它同时观察 64 个 embedding，混入新 token、63 个历史 token 和窗口左移。

Newest-token 投影：

\[
z_i^{\rm newest}(t)
=\langle r_i,e_{t+1}\rangle,
\qquad r_i\in\mathbb R^{512}.
\]

它更直接观察新生成 embedding，但仍只是状态投影，不是 token Jacobian。

每个面板标题打印三轴实际 span，必须先看数值范围再看线条形状：

- `step0` full/newest span 都约 \(10^{-6}\)，只是极小收敛轨迹；自动缩放会把它画得很大。
- `step16000` full 某轴 span 达 \(1.59\times10^{-2}\)，但 newest 最大约 \(2.58\times10^{-4}\)，说明新 embedding 已接近稳定，而完整窗口仍在清除/移动历史。
- `step143000` full/newest 都有 \(10^{-2}\) 量级跨度，支持该 anchor 的持续活跃轨道。

## 11. 图六：Return maps

![rolling return maps](assets/rolling_next_token/rolling_projection_return_maps_full_vs_newest.png)

每个点为：

\[
(z_0(t),z_0(t+1)).
\]

虚线是 \(z_0(t+1)=z_0(t)\)。仍然只用 `doc264@0`。

- 收缩到对角线一个点：固定点候选；
- 两个交替簇：period-2 候选；
- \(k\) 个按时间循环的簇：period-\(k\) 候选；
- 连续多分支：复杂周期、准周期、混沌或长暂态候选。

step1000 full/newest 都出现两个主要簇，centered crossing 约 128 次/256 steps，符合 period-2-like 线索；但必须用高维 recurrence 确认。step16000 newest 向一个位置收缩；step143000 多分支且跨度大，与其扩张候选一致。

## 12. 图七：共享绝对截面 Projected Poincaré

![shared Poincare](assets/rolling_next_token/rolling_projected_poincare_shared_absolute.png)

对 full/newest 分别在四 checkpoint 的同一个 matched anchor 上定义一个共享截面：

\[
c=\operatorname{median}\{z_0(t):\text{all four checkpoints}\}.
\]

检测向上穿越：

\[
z_0(t)\le c<z_0(t+1).
\]

在线段上插值：

\[
\alpha=\frac{c-z_0(t)}{z_0(t+1)-z_0(t)},
\]

\[
z_1^*=z_1(t)+\alpha[z_1(t+1)-z_1(t)],
\]

\[
z_2^*=z_2(t)+\alpha[z_2(t+1)-z_2(t)].
\]

| scope | step0 | step1000 | step16000 | step143000 |
|---|---:|---:|---:|---:|
| full shared absolute | 1 | 128 | 0 | 0 |
| newest shared absolute | 0 | 0 | 0 | 6 |

多数面板没有 crossing，不表示没有动力学，只表示轨迹位于该共享截面的同一侧。它说明不同 checkpoint 的投影中心偏移很大，不能为了画出点云给每个 checkpoint 偷换截面位置。

## 13. 图八：中心化诊断 Poincaré

![centered Poincare](assets/rolling_next_token/rolling_projected_poincare_centered.png)

为观察各轨迹中心附近的局部几何，另画诊断版本：

\[
\widetilde z_i(t)
=z_i(t)-\operatorname{median}_tz_i(t),
\qquad
\widetilde z_0=0.
\]

| scope | step0 | step1000 | step16000 | step143000 |
|---|---:|---:|---:|---:|
| full centered | 1 | 128 | 22 | 85 |
| newest centered | 0 | 128 | 2 | 85 |

它更容易显示局部循环，但牺牲了绝对位置：

- shared-absolute 图回答“同一个投影平面是否被穿越”；
- centered 图回答“各自中心附近有什么重复几何”；
- centered crossing 多不自动等于高维周期，更不自动等于混沌。

step16000 full 有 22 次 crossing、newest 仅 2 次，再次说明完整窗口仍可能因历史平移穿越截面，而新 embedding 已基本稳定。

## 14. 图与指标的联合阅读顺序

1. 先看样本级 Lyapunov 的符号和离散度；
2. 用 nearby distance 验证有限扰动方向，但注意饱和；
3. 用 relative delta 区分固定点与活跃稳定轨道；
4. 用 hard exact cycle 判断 greedy token-window 周期；
5. 比较 full/newest 投影，分离历史窗口运动和 innovation；
6. 用 return map 提出周期候选；
7. 最后看 shared/centered Poincaré 截面几何。

| 组合现象 | 更合理解释 | 仍需排除 |
|---|---|---|
| Lyapunov正、nearby扩张、投影有界复杂 | soft 混沌候选 | 长暂态、有限时间偏差 |
| Lyapunov负、relative delta→0 | 稳定固定点候选 | float32 数值地板 |
| Lyapunov负、delta非零、return map有限簇 | 稳定周期候选 | 投影重叠、burn-in不足 |
| full活跃、newest收缩 | 新 token 稳定但窗口历史仍移动 | shift/lag 结构 |
| centered点云、shared无 crossing | 局部几何存在但绝对中心不同 | 截面依赖 |

三投影不能独立证明混沌，因为 \(32768\to3\) 是非单射压缩；Frobenius不能独立证明临界，因为它是平均单步增益；hard 周期也不能直接决定 soft operator 的 Lyapunov 相位。

## 15. 与核心论文的对应关系

[原论文 `Optimal Machine Intelligence at the Edge of Chaos`](https://arxiv.org/abs/1909.05176)研究同维非线性算子的渐近吸引子、Jacobian norm 和稳定/周期/混沌相位。

rolling next-token 相对旧 direct-hidden feedback 更贴近 causal LM 任务，但仍有三个 mismatch：

1. soft expectation embedding 不是实际 token；
2. reset-position rolling operator 不是绝对位置增长的原生生成；
3. full seq Jacobian 含确定性 shift identity，导致 normalized Frobenius 天然接近 1。

因此合理对应是：

\[
\text{训练进度}
\leftrightarrow
\lambda_{\max}
\leftrightarrow
J_{\rm new}
\leftrightarrow
\text{soft/hard asymptotic behavior},
\]

而不是：

\[
\rho_{\rm rolling,total}\approx1
\Rightarrow\text{临界}.
\]

## 16. 能支持与不能支持的结论

能支持：

- total rolling Frobenius 接近 1 主要由 seq shift 产生；
- 训练 checkpoint 改变 soft stability 和 hard cycle 分布；
- step0 的 soft expectation 是一致收缩的近均匀平均态；
- 训练后存在明显 anchor 异质性；
- full-window 与 newest-token 投影观察不同层次的运动；
- 对 `doc264@0`，step16000 newest 比 full 更接近静止，step143000 两者都保持较大跨度。

不能证明：

- Pythia 单调到达并停留在 edge of chaos；
- total Frobenius≈1 表示临界；
- 三投影点云就是高维混沌吸引子；
- centered crossing 数就是周期长度；
- hard argmax 周期等于 soft expectation 周期；
- `projection_newest` 等于 last-token Jacobian；
- innovation 总范数能给出最后输入 token 的贡献。

## 17. 文件与复核入口

全分辨率图、CSV、manifest 和 build log：

`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/visualization_supplement/`

派生 CSV：

- `processed/soft_trajectory_quantiles.csv`
- `processed/hard_cycle_anchor_metrics.csv`
- `processed/selected_anchor_projection_rows.csv`
- `processed/selected_anchor_projection_spans.csv`
- `processed/projected_poincare_shared_absolute.csv`
- `processed/projected_poincare_centered.csv`

代码：`scripts/build_rolling_next_token_visualization_supplement.py`  
精选图：`reports/assets/rolling_next_token/`  
token-block 后续：`plan/rolling_token_block_jacobian_followup_plan.md`
