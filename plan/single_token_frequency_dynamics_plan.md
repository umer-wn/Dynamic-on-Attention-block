# 三组单 Token 词频分层动力学与方形 Jacobian 实验计划

状态：v1.3 四 checkpoint Pilot 完成；Main 暂缓，优先处理 nearby numerical floor 与早期 checkpoint 加密
日期：2026-07-13  
优先级：高于 rolling token-block follow-up  
数据根：`/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/`

## 0. 实验目的

构造三组不含 rolling shift、目标 token 输入输出严格同维的 Transformer 连续动力系统，比较无上下文、冻结上下文和动态上下文。G1/G2 的完整状态为单 token：

$$
x_t\in\mathbb R^H,
\qquad
x_{t+1}=F_\theta(x_t)\in\mathbb R^H.
$$

G3 的完整状态是 $X_t\in\mathbb R^{L\times H}$，但观测和求导目标始终是最后一个 token 的 $H$ 维 block。

对 Pythia-70M，hidden/featuresize 为 $H=512$，因此每一步的 Jacobian 是真正的方阵：

$$
J_t=\frac{\partial F_\theta(x_t)}{\partial x_t}
\in\mathbb R^{512\times512}.
$$

三组定义为：

1. `G1 isolated_token`：只输入并循环目标 token；
2. `G2 frozen_context`：输入前文与目标 token，但前 $i-1$ 个位置每步恢复为原始 embedding，只有目标 token 参与动态更新；
3. `G3 dynamic_context`：前文与目标 token 都用上一轮 hidden state 回灌，整个上下文参与动态更新，但 Jacobian 始终只取目标 token 对目标 token 的 $H\times H$ block。

本实验同时回答：

1. 去掉 rolling-window 的 shift identity 后，normalized Frobenius 是否仍接近 1；
2. 单 token 初始条件是否随语料 token 词频表现出不同的收敛、Jacobian 和 Lyapunov 行为；
3. 训练 checkpoint 是否改变词频与动力学之间的关系；
4. Hutchinson/JVP 估计与可直接计算的完整 $512\times512$ token-block Jacobian 是否一致；
5. 同一个 token 在无上下文、冻结上下文和动态上下文中是否呈现不同的条件稳定性。

## 1. 与前面实验的区别

### 1.1 不是 rolling-window

不存在：

$$
[x_1,\ldots,x_L]\mapsto[x_2,\ldots,x_L,e_{\mathrm{new}}].
$$

因此没有 $(L-1)H$ 维的确定性 shift identity，也不存在
`sqrt((L-1)/L)≈0.9922` 的结构性 Frobenius 基线。

### 1.2 不是 rolling token-block Jacobian

rolling token-block 实验计算的是：

$$
\frac{\partial e_{\mathrm{new}}}{\partial x_\ell}
\in\mathbb R^{H\times H},
$$

但其算子本身包含 sliding window 和 next-token innovation。本实验不做 shift：G1/G2 的完整自治算子是：

$$
F_\theta:\mathbb R^H\rightarrow\mathbb R^H.
$$

G3 虽然更新完整上下文，但只计算 last-output/last-input 的 $H\times H$ Jacobian block，并明确降级为条件稳定性指标。

### 1.3 数学对齐程度分组不同

G1 和 G2 都是严格的 $\mathbb R^H\rightarrow\mathbb R^H$ 自治映射，可重复迭代并在渐近轨迹上计算方形 Jacobian，因此在数学形式上比 rolling-window 更接近 arXiv:1909.05176。

G3 的完整自治状态实际是 $X_t\in\mathbb R^{L\times H}$；本实验只测目标位置的条件 Jacobian block。因此 G3 更接近“带动态上下文的 token 条件稳定性”，不能把 token block 的 normalized Frobenius 当作完整 $LH$ 维系统的论文指标。

但它不执行 next-token prediction，也没有上下文生成；token 只决定初始状态 $x_0$。所以它不能被称为“原生自回归 LLM 动力学”。

## 2. 三组算子定义

### 2.1 初始状态

对 tokenizer token id $w$，从当前 checkpoint 的输入 embedding 表取得：

$$
x_0=E_\theta[w]\in\mathbb R^H.
$$

同一批 token id 在四个 checkpoint 间保持一致，但 $E_\theta[w]$ 会随 checkpoint 改变。

### 2.2 G1：isolated-token feedback

每一步构造：

```text
inputs_embeds: [1, 1, H]
attention_mask: [1, 1]，值为1
position_ids: [1, 1]，值为0
use_cache: false
model weights: frozen
```

取 base Transformer 的最终 hidden state：

$$
F_\theta(x)
=M_\theta(\text{inputs\_embeds}=x[None,None,:])_{0,0,:}.
$$

输出形状为 `[H]`，直接作为下一步输入。

### 2.3 G2：frozen-context single-token feedback

从真实语料中取目标 token $w_i$ 及其前文：

$$
C_0=[E_\theta(w_{i-L+1}),\ldots,E_\theta(w_{i-1})]
\in\mathbb R^{(L-1)\times H},
$$

$$
x_0=E_\theta(w_i).
$$

每一步输入：

$$
X_t=[C_0,x_t],
$$

但只取目标位置输出并更新：

$$
x_{t+1}=F^{(2)}_{\theta,C_0}(x_t)
=M_\theta([C_0,x_t])_{L-1}.
$$

前文 $C_0$ 每一步都重新填入相同的原始 token embeddings，不使用上一轮 prefix hidden states。因此对给定上下文，G2 仍是自治的 $\mathbb R^H\rightarrow\mathbb R^H$ 映射。

这里“维持原样填充”指填入原前文 embedding，不是填 PAD token。所有位置都有效，attention mask 全为 1。

### 2.4 G3：dynamic-context feedback

初始状态仍为真实上下文 embedding：

$$
X_0=[C_0,x_0]\in\mathbb R^{L\times H}.
$$

但每一步所有位置都参与回灌：

$$
X_{t+1}=F^{(3)}_\theta(X_t)=M_\theta(X_t)\in\mathbb R^{L\times H}.
$$

目标 token 轨迹为最后位置：

$$
x_t=X_t[L-1].
$$

本计划把用户所说“前文参与填充”解释为“前文也参与每步 dynamic feedback”；即 prefix 使用上一轮对应位置的 hidden state，而不是恢复为原始 embedding。

### 2.5 三组共同约束

LM head、softmax、argmax、token sampling 和 embedding expectation 均不参与三组主算子。统一使用 final hidden output。

```text
attention_mask: 全1，无padding
position_ids: 0...L-1，每个dynamic step重置
use_cache: false
model weights: frozen
dtype: float32，autocast关闭
```

G1 的 $L=1$；G2/G3 的主 context length 为 $L=32$，即 31 个前文 token 加 1 个目标 token。

### 2.6 mask、padding、batch 和位置

- 没有 padding；G2/G3 只选择同一文档中具有足够前文的目标 occurrence。
- attention mask 在所有有效位置恒为 1。
- 主 Jacobian 计算采用 `batch_size=1`，避免 batch 维混入导数定义。
- position id 在每一步固定为 `0...L-1`，保证算子不显式依赖动力学时间 $t$。
- 不使用 KV cache，因为每一步是完整而独立的同一个自治映射。

### 2.7 必须明确的 attention 差异

只有 G1 的序列长度为 1，每个 attention head 的 softmax 只有一个元素：

$$
\operatorname{softmax}([a])=[1].
$$

因此 G1 的 attention 权重不再表达 token-token 选择，Q/K 对权重变化的路径基本退化；主要有效路径来自 V/O 投影、MLP、LayerNorm 和 residual。G2/G3 保留目标 token 对前文的 causal attention，可用来判断 G1 结果是否主要来自 seq1 退化。

Pilot 中对 G1 增加 `position_id∈{0,31,63}` 的小型一致性检查。若三者结果完全一致，这是 seq1 attention/position 退化的预期 sanity check；若不一致，需要定位模型的位置编码路径。

## 3. Token 词频统计与分层

### 3.1 统计对象

这里统计的是 tokenizer 切分后的 **token id occurrence frequency**，不是自然语言“词”的词频。

默认语料：本地已有的 `WikiText-2 train`。使用与 Pythia 相同的 tokenizer，逐文档执行：

```text
add_special_tokens=false
padding=false
truncation=false
```

然后累计：

- `token_count`：总出现次数；
- `document_frequency`：出现过该 token 的文档数；
- `corpus_token_fraction`；
- `frequency_rank`；
- `log10_count=log10(token_count+1)`。

输出：

```text
/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/
  frequency_audit/token_frequency_table.csv
  frequency_audit/token_selection_manifest.csv
  frequency_audit/frequency_summary.json
```

### 3.2 频率是代理量

WikiText-2 频率不是 Pythia 在 The Pile 预训练时的真实 token 频率。本轮只能解释为：

> token 在统一外部语料上的经验常见程度，与单 token 动力学指标之间的关联。

不能写成“Pythia 训练频率导致了某种动力学”。若后续获得 The Pile token counts，再以相同 token manifest 替换频率列复测。

### 3.3 频率分层

frequency audit 同时输出全词表排名和 contextual-eligible 排名。三组配对实验要求 token 至少有两个带63-token前文的真实 occurrence，因此实际抽样在 `context_eligible_count>=2` 的 lexical token 中按 frequency rank 预注册四层：

| 层 | rank 范围 | 含义 |
|---|---:|---|
| very_high | top 1% | 极高频 token type |
| high | 1%–10% | 高频 |
| medium | 10%–50% | 中频 |
| rare | bottom 50% | 低频/稀有 |

实际 count 上下界在频率审计后写入 manifest，不根据动力学结果调整。

全词表排名仍保留在表中，用于说明 context eligibility 对 rare token 的选择偏差；报告中的“rare”严格指 contextual-eligible cohort 内的相对低频，而不是全词表 singleton。

### 3.4 Token 过滤与混杂控制

主统计 cohort 只保留可打印、包含至少一个字母/数字/CJK 字符的 lexical token；保留前导空格并单独标记。排除：

- tokenizer special token；
- 空字符串和纯控制字符；
- 无法稳定 decode 的 byte fallback；
- 语料中未出现的 token。

每个 token 额外记录：

- decoded string；
- 字符长度；
- 是否有前导空格；
- lexical/punctuation/whitespace/byte 类别；
- 初始 embedding norm；
- embedding 与各频率层中心的距离。

主 cohort 在四层各固定抽取 16 个 token，共 64 个 token，seed=`1234`。另构造每层 8 个 token 的 embedding-norm matched 子集，用于检查频率效应是否只是初始向量范数差异。

若某层 lexical token 不足，先报告不足，再扩大 rank 边界；不得静默用标点或 byte token 补齐。

### 3.5 上下文 occurrence 抽样

频率层按 token type 定义，但 G2/G3 必须从语料中选择该 token 的具体 occurrence。对每个入选 token id：

- 只保留同一文档内至少有 63 个前文 token 的 occurrence；
- 使用 seed=`1234` 固定选择 3 个不同 context occurrences；
- 同一 context id 在 G2/G3 和四个 checkpoint 间完全一致；
- 主分析使用 `L=32`，即截取最近 31 个前文 token；
- context-length ablation 使用 `L∈{8,64}`，只在每个频率层预注册的 4 个 token 上执行；
- 记录 prefix token ids、decoded text、目标 token 在文档中的位置和文档 id；
- 不允许跨文档拼接前文。

若 rare token 没有足够的长前文 occurrence，该 token 在上下文组中标记为 unavailable，并从所有上下文组的配对分析中共同剔除；不能为 G2/G3 分别换不同 token。

## 4. Jacobian 与 normalized Frobenius

### 4.1 完整 Jacobian

对三组统一定义目标 token block：

$$
(J^{(g)}_t)_{ij}
=\frac{\partial x^{(g)}_{t+1,i}}{\partial x^{(g)}_{t,j}},
\qquad J^{(g)}_t\in\mathbb R^{H\times H}.
$$

- G1：$J^{(1)}_t=\partial F^{(1)}(x_t)/\partial x_t$；
- G2：$J^{(2)}_t=\partial F^{(2)}(x_t;C_0)/\partial x_t$，冻结前文不求导；
- G3：

$$
J^{(3)}_t
=\frac{\partial F^{(3)}_\theta(X_t)[L-1]}
{\partial X_t[L-1]},
$$

即完整 $LH\times LH$ Jacobian 的 last-output/last-input block。前文状态在当前导数调用中作为给定值，但在下一 dynamics step 会继续更新。

实现优先使用 `torch.func.jacrev`，以 `chunk_size=16/32` 分块，避免一次 materialize 512 条反向图造成峰值显存过高。Pilot 用 basis-JVP 构造 Jacobian 列进行交叉验证：

$$
J_te_j=\text{column}_j(J_t).
$$

每个保存的 matrix 必须记录：group、checkpoint、token id、context id/length、trajectory step、shape、dtype、checksum 和 operator version。

### 4.2 精确 normalized Frobenius

因为输入输出维数都为 $H$：

$$
\rho_t^{\mathrm{exact}}
=\frac{\|J_t\|_F}{\sqrt H}
=\sqrt{\frac1H\sum_{i=1}^{H}\sum_{j=1}^{H}J_{t,ij}^2}.
$$

这里阈值 1 的含义与论文同维归一化一致：奇异值平方均值为 1。它仍不等于最大奇异值为 1，也不能单独证明 Lyapunov 为 0。

### 4.3 Hutchinson/JVP 估计

对 Rademacher probe $v_k\in\{-1,+1\}^H$：

$$
\widehat\rho_t
=\sqrt{\frac{1}{HK}\sum_{k=1}^{K}\|J_tv_k\|_2^2}.
$$

完整 Jacobian 状态上同时计算 exact 与 Hutchinson，用于估计误差：

$$
\mathrm{relative\ error}
=\frac{|\widehat\rho_t-\rho_t^{\mathrm{exact}}|}
{\rho_t^{\mathrm{exact}}+10^{-12}}.
$$

Pilot 比较 `K∈{8,16,32,64}`，Main 使用达到误差门控的最小 K，默认从 K=32 开始。

### 4.4 奇异值和主方向

对完整 $512\times512$ Jacobian 计算：

- 全部 singular values；
- `sigma_max`、median/mean/RMS singular value；
- stable/effective rank；
- `sigma_max / RMS` 各向异性；
- 最大奇异向量与轨迹更新方向的夹角。

这使“Frobenius 接近 1”可以与“少数方向强扩张”区分。

### 4.5 G3 的解释边界

G3 的 $J^{(3)}_t$ 不是完整序列 Jacobian，所以：

- $\rho^{(3)}=\|J^{(3)}_t\|_F/\sqrt H$ 是目标 token 条件 RMS gain；
- 它不能替代 $\|J_{\mathrm{full}}\|_F/\sqrt{LH}$；
- 它不能排除 prefix 子系统存在更强扩张方向；
- 沿 $J^{(3)}_t$ 连乘得到的是 target-restricted/conditional Lyapunov。

由于 causal mask，最后位置输入理论上不应影响更早位置的同一步输出：

$$
\frac{\partial F^{(3)}(X_t)[0:L-1]}
{\partial X_t[L-1]}=0.
$$

Pilot 必须显式抽查这一 cross-gradient 是否为数值零。若不为零，说明 causal mask、位置索引或实现有错误，实验停止。

## 5. 轨迹与 Lyapunov 协议

### 5.1 迭代长度

每个 `group × checkpoint × token × context` 运行 768 个 dynamics steps：

- `t=0...511`：transient window；
- `t=512...767`：asymptotic evaluation window。

这不是在隐藏前 512 步：完整轨迹都会保存和画图。512 只是把“初始 token embedding 的暂态”和“后期动力学”分开汇总，并对应论文中约 500 次循环的量级。

### 5.2 每步保存

- 目标 token 完整 state vector `[H]`；
- state norm；
- absolute/relative step delta；
- 三个固定随机投影；
- nearby trajectory distance；
- 是否出现 NaN/Inf；
- 与初始 token embedding、最终状态的距离。

G3 额外每步保存 prefix norm、prefix relative delta 和三个固定随机投影；完整 `[L,H]` 状态只在 `t=0,512,767` 保存，避免 Main 轨迹数据膨胀到几十 GiB。

完整 state 以 checkpoint/token 分片保存，避免一个超大文件。

### 5.3 Jacobian 采样状态

Main 对每个 group/token 的 canonical context 至少保存三个完整目标-token Jacobian：

```text
t=0     初始 token embedding
t=512   asymptotic window 起点
t=767   最终状态
```

G1 没有 context id；G2/G3 对每个 token 预注册一个 canonical context 保存完整矩阵。另两个 context replicate 只计算 Hutchinson/JVP summaries，避免重复保存数 GiB 完整矩阵。所有 context 在 `t=512...767` 均匀选择 8 个状态计算 Hutchinson Frobenius。Pilot 可增加更多 exact states，以检查 Jacobian 是否已稳定。

模型权重虽然固定，但 $J_t=J_F(x_t)$ 依赖当前非线性状态 $x_t$，所以不同 dynamics step 的 Jacobian 通常不相同。

### 5.4 Benettin 最大 Lyapunov

在 256-step evaluation window 计算：

$$
w_t=J_tv_t,
\qquad
a_t=\|w_t\|_2,
\qquad
v_{t+1}=w_t/a_t,
$$

$$
\widehat\lambda_{\max}
=\frac1{256}\sum_{t=512}^{767}\log a_t.
$$

每个 token/context 使用 2 个确定性 seed 的初始切向 probe。G1/G2 报告单-token最大有限时间 Lyapunov；G3 报告 target-conditional Lyapunov，不称为完整系统最大 Lyapunov。最终相位解释优先级：

```text
Lyapunov + trajectory convergence
> exact normalized Frobenius / singular spectrum
> nearby finite-distance separation
> 投影几何
```

### 5.5 收敛判据（唯一主判据）

是否达到稳定收敛只由以下三项联合判断：

1. `tail_relative_step_delta`：最后32步

$$
r_t=\frac{\|x_{t+1}-x_t\|_2}{\max(\|x_{t+1}\|_2,10^{-12})};
$$

2. `nearby_distance`：从目标token方向加入 $\epsilon=10^{-3}$ 单位扰动，比较 evaluation window 起点和终点，并报告

$$
g_d=\frac1T\log\frac{d_T}{\max(d_0,10^{-12})};
$$

3. Benettin `target Lyapunov`：G1/G2 为单token最大有限时间指数，G3 为 target-conditional 指数。

预注册标签：

| 标签 | relative step delta | nearby | Lyapunov |
|---|---|---|---|
| stable fixed-point candidate | tail mean `<1e-6` | $g_d<0$ | 两probe均 `<0` |
| stable non-fixed candidate | tail mean `>=1e-6` | $g_d<0$ | 两probe均 `<0` |
| expanding/chaotic candidate | tail mean `>=1e-6` | $g_d>0$ | 两probe均 `>0` |
| unresolved/conflicting | 其他组合 | 其他组合 | 包含0或probe异号 |

若 nearby distance 已落到 float32 resolution floor，nearby 标为 `numerical_floor`，整体标签降为 unresolved，不能仅靠 relative delta 与 Lyapunov 补成“已收敛”。本轮将可复核阈值预注册为

$$
d_{\mathrm{floor}}=8\,\epsilon_{\mathrm{fp32}}\max(\|x_t\|_2,1),
$$

并把每条轨迹实际使用的 `nearby_resolution_floor` 写入 summary。系数8是保守的多次舍入余量；后续可通过 float64/epsilon ablation 审计，但不得在看到标签后反向调阈值。Jacobian Frobenius、奇异值、recurrence和Poincaré均不参与这个主标签。

### 5.6 2D/3D轨迹与Poincaré绘图目标

对目标token状态使用跨group/checkpoint/context共享的4个固定随机单位投影：

$$
z_k(t)=\langle q_k,x_t\rangle,\qquad k=0,1,2,3.
$$

必须生成：

1. 2D轨迹：$(z_0,z_1)$，颜色编码 dynamics step；
2. 3D轨迹：$(z_0,z_1,z_2)$，标记起点和最终状态；
3. 2D Projected Poincaré：以 $z_0$ 从截面下方向上穿越，画 $(z_1,z_2)$；
4. 3D Projected Poincaré：同一crossing画 $(z_1,z_2,z_3)$。

截面取每条 evaluation trajectory 自己的 `median(z0)`，crossing在相邻离散状态间线性插值。主图一次只展示一个 `token × context × group × checkpoint`，不同轨迹不得在绝对坐标上混画。跨token/context聚合时必须先减去各自最终投影状态，并明确写成 centered diagnostic，不称为共同Poincaré section。

Poincaré只回答“穿越截面时的投影几何是否成点/成簇/成带”，不参与收敛判定：

- 无crossing是合法结果；
- 固定点可能最终不再穿越；
- 固定点附近大量微尺度crossing可由float32抖动产生；
- 有限个Poincaré点只有结合nearby/relative delta/Lyapunov后才能讨论渐进周期候选。

## 6. 对照与数值校准

必须先通过以下控制：

1. Identity：$F(x)=x$，应有 $J=I$、$\rho=1$、Lyapunov=0。
2. Contraction：$F(x)=\alpha x$，应有 $J=\alpha I$、$\rho=|\alpha|$、Lyapunov=$\log|\alpha|$。
3. Exact-vs-JVP：完整 Jacobian 乘随机向量应与 autograd JVP 一致。
4. Finite difference：epsilon=`1e-3` 的中心差分方向导数与 JVP 在 float32 容差内一致。
5. Determinism：同 token/checkpoint/seed 重跑轨迹、Jacobians 和 probes 一致。
6. Position sanity：seq1 下 position 0/31/63 的结果若不同，必须解释位置编码路径。
7. Pre/post final LayerNorm diagnostic：Pilot 比较最后 block 输出与最终 LayerNorm 后输出，判断归一化层是否主导收缩；主结论仍以预注册的 post-final-hidden 算子为准。
8. Group equivalence：当 G2/G3 的 context length 人为设为1时，第一步输出和 target Jacobian 应与 G1 一致。
9. Frozen-prefix：G2 每一步 prefix tensor 与 $C_0$ bitwise 一致。
10. Causal cross-gradient：G3 的 `d prefix_output / d target_input` 应为0或数值容差内的0。

## 7. 实验假设

| 假设 | 支持条件 | 反证/未决条件 |
|---|---|---|
| H1 去掉 shift 后 normalized Frobenius 仍接近1 | 多数 token 的 asymptotic exact $\rho$ 在预注册容差内靠近1，且 Lyapunov也接近0 | $\rho$ 系统偏离1；或 $\rho≈1$ 但 Lyapunov显著非零 |
| H2 token 频率与局部动力学有关 | `log10_count` 对 tail $\rho$/Lyapunov 的效应在 checkpoint、embedding norm 和 token属性控制后仍稳定 | 效应接近0、方向跨seed/checkpoint不稳定 |
| H3 训练改变频率—动力学关系 | frequency×checkpoint interaction 稳定且置信区间不跨0 | 四 checkpoint 斜率无系统变化 |
| H4 完整 Jacobian 能校准 JVP | Hutchinson 与 exact Frobenius 相对误差达到门控，JVP与矩阵乘积一致 | 估计偏差大或数值不稳定 |
| H5 高频/低频 token 进入不同吸引子 | sample内轨迹、fixed point/period、Lyapunov 和 Jacobian均出现可复现分层 | 仅最终固定点位置不同，局部稳定性相同 |
| H6 上下文改变单token局部稳定性 | 同 token/context 配对下 G1/G2/G3 的 tail $\rho$、条件Lyapunov或收敛状态有稳定差异 | group effect 小且置信区间跨0，或只由context length/embedding norm解释 |
| H7 动态前文不同于冻结前文 | G3 相对 G2 出现可复现差异，且prefix delta与target指标相关 | G2/G3在误差内相同，或G3差异仅来自暂态未消除 |

“接近 1”不预先写死为结论。Pilot 后只根据 identity/contraction 校准误差确定数值容差，不能根据模型结果调阈值。

## 8. Pilot

### 8.1 配置

- model：`EleutherAI/pythia-70m`；
- checkpoints：`step0`、`step143000`；
- frequency strata：4；
- tokens：每层 4，共 16；
- groups：G1/G2/G3；
- contexts：G2/G3 每 token 固定 2 个 occurrence；
- context length：主 `L=32`，另在每层1个 token 上检查 `L=8/64`；
- trajectory：768 steps；
- exact Jacobian：`t=0,512,767`；
- Hutchinson K：8/16/32/64；
- Lyapunov probes：2；
- dtype：float32，关闭 autocast；
- GPU：step0 的 G1/G2/G3→GPU5，step143000 的 G1/G2/G3→GPU6；
- GPU7 留作 context-length、position、LayerNorm diagnostic 和失败重试；
- 不下载新模型权重。

数据：

```text
/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/pilot/
```

### 8.2 Pilot 门控

进入 Main 前必须满足：

- 所有完整 Jacobian shape 严格为 `[512,512]`；
- 三组都只能生成目标 token `[512,512]` Jacobian，禁止误存 full-sequence Jacobian；
- identity/contraction 解析误差通过；
- matrix-vector、JVP 和 finite difference 一致；
- 至少 K=32 时 median Frobenius relative error ≤5%；
- 无 NaN/Inf，完整轨迹可确定性复现；
- 单个 checkpoint 的预计 Main 运行时间和峰值显存可接受；
- 词频四层各有足够合格 lexical token。
- G2 prefix 全程不变，G3 prefix 确实更新；
- G3 causal cross-gradient 通过零值检查；
- 同 token 的 group/context 配对 manifest 完整。

若 `jacrev` 显存过高，改为 chunked VJP/basis-JVP；不得为了通过门控减少 Jacobian 维度或重新引入 sequence flattening。

## 9. Main

- checkpoints：`step0`、`step1000`、`step16000`、`step143000`；
- 固定 token cohort：四个频率层各 16，共 64；
- 四 checkpoint 使用完全相同的 token ids 和频率标签；
- groups：G1 isolated、G2 frozen context、G3 dynamic context；
- G2/G3 每 token 3 个固定 context occurrences，主长度 `L=32`；
- 每个频率层4个token增加 `L=8/64` ablation；
- trajectory：每 token 768 steps；
- exact Jacobian：G1和G2/G3 canonical context各保存 `t=0,512,767`；
- 其余 context：不保存完整矩阵，但计算8个tail Hutchinson状态与conditional Lyapunov；
- Lyapunov：2 probes × 256 steps；
- GPU5/6/7 并行三个 checkpoint，第四个接续最先空闲 GPU；
- 数据只写 `/home/luohaoming`，仓库只保留 config、代码、计划、报告和压缩图。

预计 canonical-context 完整 Jacobian 原始矩阵约：

$$
4\times64\times3\ \text{groups}\times3\ \text{states}
\times512\times512\times4\ \text{bytes}
\approx 2.25\ \text{GiB}.
$$

目标-token轨迹、G3稀疏full-state快照和context replicates预计再占1–3 GiB；运行前检查目标目录至少保留 12 GiB 空间，用于矩阵、states、临时文件、日志与processed数据。

## 10. 统计分析

主要分析单位是 token，而不是 trajectory step。先对同一 token 的 tail states 聚合，再比较频率，避免把 256 个时间点当作 256 个独立样本。

主模型：

$$
y_{w,c,g}=\beta_0+\beta_1\log_{10}(\mathrm{count}_w+1)
+\beta_2\mathrm{checkpoint}+\beta_3\mathrm{group}
+\beta_4(\mathrm{frequency}\times\mathrm{group})
+\beta_5(\mathrm{checkpoint}\times\mathrm{group})
+\gamma^T C_{w,c}+u_w+u_c+\epsilon_{w,c,g},
$$

其中 $C_{w,c}$ 至少包含 initial embedding norm、字符长度、前导空格、token 类别、context length和prefix norm；$u_w/u_c$ 表示 token 与 context occurrence 的配对效应。

同时报告：

- 每 checkpoint 的 Spearman correlation；
- 四频率层的全部 token 点、median 和 bootstrap 95% CI；
- frequency label permutation null；
- raw cohort 与 embedding-norm matched cohort；
- 多指标比较的 FDR 校正；
- token-level effect size，不只报告 p-value。
- 同 token、同 context 的 G2-vs-G3 配对差值及bootstrap CI；
- context间方差，避免把一个特定句子的结果误写成token固有属性。

## 11. 输出图表

1. token count/rank 分布及四个频率层边界；
2. 选中 token 的 decoded string、count、embedding norm 审计图；
3. `log frequency → initial/tail exact normalized Frobenius`；
4. `log frequency → 最大 Lyapunov`；
5. checkpoint × frequency 的相位/收敛矩阵；
6. 各频率层代表 token 的 relative-step 和 nearby-distance 轨迹；
7. exact Frobenius 与 Hutchinson 估计 parity plot；
8. 初始 Jacobian 与 tail Jacobian 对照；
9. singular spectrum 和 `sigma_max/RMS` 各向异性；
10. raw 与 embedding-norm matched frequency effect；
11. position id 与 pre/post LayerNorm sanity 图。
12. 同 token 的 G1/G2/G3 exact $\rho$ 配对图；
13. frozen-vs-dynamic context 的 conditional Lyapunov 差值；
14. G3 prefix relative delta 与 target-token稳定性关系；
15. context length 8/32/64 ablation。

每张图必须附 sidecar/manifest，记录源文件、token ids、频率层、checkpoint、状态 step、坐标含义、允许解释和 caveat。

## 12. 能证明与不能证明

本实验可以支持或反驳：

- 在无 sliding shift 的方形单 token 算子中，normalized Frobenius 是否接近1；
- 不同经验词频 token 作为初始条件时，是否进入不同稳定性区域或吸引子；
- 训练 checkpoint 是否改变这种关联；
- 旧 JVP Frobenius 实现是否被完整 Jacobian 校准。
- 同一token在无上下文、冻结前文和动态前文中的目标block稳定性是否不同；
- 动态前文是否改变目标token的条件Lyapunov和吸引行为。

本实验不能单独证明：

- Pythia 原生自回归生成位于混沌边缘；
- token 词频因果决定动力学；
- WikiText-2 词频等于 Pythia 预训练词频；
- seq1 结果能推广到多 token attention；
- $\rho≈1$ 单独等价于 Lyapunov≈0 或任务性能最优。
- G3目标token block等价于完整序列系统的normalized Frobenius或最大Lyapunov；
- 上下文效应可以从有限的WikiText context直接推广到任意语境。

## 13. 计划文件与留痕

计划实现文件：

```text
src/single_token_dynamics.py
scripts/compute_token_frequency.py
scripts/compute_single_token_dynamics.py
scripts/analyze_single_token_frequency_dynamics.py
configs/pythia_single_token_frequency_*.yaml
tests/test_single_token_dynamics.py
reports/single_token_frequency_dynamics_report.md
```

目录结构：

```text
/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/
  frequency_audit/
  pilot/
  main/
  processed/
  figures/
  manifests/
  logs/
```

## 14. 2026-07-13 执行记录

用户确认绘图目标和收敛判据后，本计划由“仅规划”进入 Pilot 执行：

- 收敛标签只使用 `relative_step_delta + nearby_distance + Benettin Lyapunov`；
- 2D/3D 轨迹和 2D/3D Projected Poincaré 只描述投影几何，不参与收敛标签；
- frequency audit 已在本地缓存的 WikiText-2 train 上运行，得到 2,419,745 个 token、29,320 个上下文合格 lexical token type 和 16 个分层样本；
- GPU5 的 step0 smoke 已完成 12 条轨迹和 36 个 exact Jacobian；所有矩阵严格为 `[512,512]`，G3 causal cross-gradient 最大值为0，四投影字段齐全；
- step0 与 step143000 Pilot 已分别提交 GPU5/GPU6；输出仅写入 `/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/pilot/`；
- 执行审计发现当前 Pilot 保存全部 target state 和逐步 prefix norm/delta，但尚未另存 G3 的三个完整 `[32,512]` snapshot；本轮目标级判据不受影响，Main 前补齐；
- Main 仍未获自动放行，必须等待 Pilot 报告和数值审计。

本节是执行留痕，不把正在运行的 Pilot 预写成完成结果。

### 14.1 Pilot 完成补记

2026-07-13：8/8 shard 完成，得到160条轨迹、122,880个时间点、288个 exact `[512,512]` Jacobian 和49张图；完整性审计通过。112/160 条 nearby trajectory 进入预注册 float32 floor，故 Main 不自动放行。下一轮优先级调整为：

1. float64 state-loop 或至少 float64 nearby subtraction 的可行性 smoke；
2. epsilon `1e-2/1e-3/1e-4` 配对校准，验证收缩/扩张符号是否稳健；
3. 补存 G3 `t=0/512/767` 完整 `[32,512]` snapshot；
4. exact-vs-Hutchinson/JVP 正式 parity 表；
5. 上述门控通过后再决定是否扩大到 Main 64-token cohort。

### 14.2 四 checkpoint 扩展补记

2026-07-13：按同一协议追加 `step1000` 与 `step16000`。四 checkpoint 合计16/16 shard、320条轨迹、245,760个时间点、576个 exact `[512,512]` Jacobian，审计通过。训练变化不是单调趋稳：isolated Lyapunov 在 step0到step1000之间跨越零附近；dynamic context 在step16000最强收缩，随后到step143000回升。nearby numerical floor 增至232/320，因此 Main 继续暂停。下一轮把 `step0→step1000` 的 checkpoint 加密与 float64/epsilon 校准列为最高优先级。
