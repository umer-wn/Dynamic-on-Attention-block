# 从论文公式到 LLM 应用场景：rolling、增长序列与 Jacobian 层级

状态：方法推导完成，追加实验尚未执行  
日期：2026-07-14

核心来源：Feng, Zhang, Lai, *Optimal Machine Intelligence at the Edge of Chaos*, arXiv:1909.05176（正文与附录 S1–S2）：https://arxiv.org/abs/1909.05176

## 1. 结论（经工程审计修订）

现有 `reset-position + soft expected embedding + window=64` rolling main 在数学上是合法的固定维自治映射；但在工程上，它只是 **recency-only、full-recompute 的自定义截断算子**，不是 Pythia 原生生成，也没有实现 Attention Sink 或模型原生 sliding-window attention。因此旧实验的 Lyapunov、周期和轨迹结论仍然成立，但其对象必须写成“该 rolling 算子”，不能直接外推为 Pythia 正常生成行为。

下一轮先做最小的工程有效性门控，而不是同时扩展四类昂贵实验：

1. `N`：Pythia 原生增长前缀 + `DynamicCache`，作为上下文上限内的正常生成基线；
2. `R`：复用现有 recent-only、reset-position、full-recompute rolling；
3. `S`：保留最初4个真实 token、滚动最近60个 token 的 sink-preserving full-recompute 基线；
4. 先比较 matched CE/PPL、hard generation、重复/周期和稀疏 attention mass；
5. 只有 R 或 S 得到工程行为支持后，再做 full-state Lyapunov 和 token-block Jacobian。

Jacobian 不能只选 token-level 或 seq-level 之一。正确分工是：

| 问题 | Jacobian/指标 | 地位 |
|---|---|---|
| rolling 动力系统是否收缩、临界或扩张 | full-state square Jacobian 的 Benettin Lyapunov | 稳定相位主判据 |
| 新生成 token 对全部上下文有多敏感 | new-token innovation Jacobian，shape `H × (L·H)` | 应用敏感性主指标 |
| 哪个输入 token 贡献敏感性 | 每个 `H × H` token block | 归因指标 |
| 最后一个输入 token 自身贡献 | `H × H` last-token block | 子指标，不能代表全部 attention context |
| full-state normalized Frobenius | `LH × LH` | shift 结构审计，不作为临界主证据 |

因此，“token-level 更符合当前模型”对 Frobenius 和任务解释是正确的；但若把 token-level Jacobian 单独当作 rolling 系统的 Lyapunov，则会漏掉扰动随窗口移动、经历史 attention 再反馈到未来 token 的路径。是否值得计算这些导数，必须先由 N/R/S 行为门控决定。

## 2. 原论文实际循环的对象

论文从固定维离散动力系统出发：

\[
x_{t+1}=F_\theta(x_t),\qquad x_t\in\mathbb R^N,
\]

\[
J_t=\frac{\partial F_\theta(x_t)}{\partial x_t}\in\mathbb R^{N\times N}.
\]

关键条件是输入与反馈输出维度相同、每次迭代使用同一个算子、状态维度固定，并在渐近轨道上判断 Jacobian norm、nearby separation 或 maximal Lyapunov。

论文的视觉模型没有把最终类别 logits 回灌。作者特意使用了一个与输入 image 像素维数相同的中间/上采样输出作为动力算子输出。因而“单位是 image”更准确地说是：**一整张固定维 image vector 是状态**，不是逐像素输出，也不完全等于部署时的分类标签输出。

高维随机近似下，论文使用渐近 normalized Jacobian norm 接近1作为 edge 条件；有限维和周期轨道上则要看沿轨道的几何平均，等价地关注 maximal Lyapunov 是否接近0。该理论要求 square state-to-state map。

## 3. LLM 中可能的状态定义

### 3.1 单 token 状态

\[
h_{t+1}=f_\theta(h_t;c),\qquad h_t\in\mathbb R^H.
\]

若上下文 `c` 冻结，这是 `H→H` square autonomous map，最接近论文的代数形式，也最容易计算 exact token Jacobian。但它反复改写同一 token state，不执行真实的 append-next-token generation，因此是局部条件动力学，不是部署生成过程。

### 3.2 不断增长的完整前缀

真实生成写成：

\[
w_{1:n+t+1}=w_{1:n+t}\oplus w_{n+t+1}.
\]

若把完整前缀作为状态，状态维度随 `t` 增长：

\[
N_t=(n+t)H.
\]

于是 `J_t` 每步尺寸不同，不再是同一个 `R^N→R^N` 算子，论文的 fixed point、Poincaré、normalized Frobenius threshold 不能直接使用。把前缀 padding 到最大长度也不能自动解决：过去 token 的 identity retention 和 inactive slots 会主导 Frobenius，产生与 rolling shift 相似的假阳性。

### 3.3 固定记忆 rolling window

令内部动力状态是最后 `L` 个连续 token states：

\[
X_t=[x_{t,1},\ldots,x_{t,L}]\in\mathbb R^{L\times H}.
\]

模型从该状态产生下一个 token embedding：

\[
e_{t+1}=g_\theta(X_t)\in\mathbb R^H,
\]

并更新：

\[
X_{t+1}=F_\theta(X_t)
=[x_{t,2},\ldots,x_{t,L},e_{t+1}].
\]

外部生成文本仍然逐 token 增长：

\[
Y_{t+1}=Y_t\oplus \operatorname{decode}(e_{t+1}),
\]

但只要模型未来仅依赖最后 `L` 个 token，增长的 transcript `Y_t` 是观测记录，不必全部进入 Jacobian 状态。这样同时满足逐 token 输出、固定维状态、可重复迭代。

因此 rolling window 是“应用语义”和“论文 fixed-dimensional map”之间最合理的桥梁，但它代表的是**有限记忆生成器**。若原模型实际使用全部历史或 KV cache 中的间接历史，rolling 是受控近似，必须做 window-length/native-cache 对照。

## 4. soft、hard 与 teacher-forced 是三个不同问题

### 4.1 可微 soft rolling

\[
p_t=\operatorname{softmax}(\ell_t/T),\qquad
e_{t+1}=p_tE.
\]

它可微，允许 JVP/Frobenius/Lyapunov，但 `p_tE` 通常不是词表中任何真实 token embedding。多步后可能进入 soft off-manifold 状态。因此它是 tangent relaxation，不是实际解码。

### 4.2 hard native generation

\[
w_{t+1}=\arg\max_v\ell_{t,v}
\]

或按固定 sampling protocol 采样。这是真实 token 输出，前缀可以真正增长并使用 native position/KV cache；但 argmax/sampling 不可微，不能直接给出论文式 Jacobian。它用于验证 soft tangent phase 是否具有生成行为意义。

### 4.3 teacher-forced 长序列

每步输入真实 next token，适合计算 CE/PPL 和真实文本轨迹上的局部 predictive Jacobian。但它是受外部语料驱动的非自治系统：

\[
X_{t+1}=F_\theta(X_t,w^{\rm true}_{t+1}).
\]

不能把 teacher-forced 的局部增益直接标为 autonomous edge of chaos。

## 5. position 使“应用对齐”与“论文对齐”不能完全重合

现有 rolling main 每一步把位置重置为 `0..L-1`。这保证 `X_{t+1}=F_θ(X_t)` 是固定 autonomous map，但不是原生长文本生成。

native absolute position 使用：

\[
X_{t+1}=F_{\theta,p_t}(X_t),\qquad p_{t+1}=p_t+1,
\]

属于非自治 cocycle。可以计算条件/纤维 Lyapunov：

\[
\lambda_X
=\lim_{T\to\infty}\frac1T
\log\left\|
J_{T-1}(p_{T-1})\cdots J_0(p_0)v_0
\right\|,
\]

但它不再是论文同一个固定 `F` 的直接复现。

形式上可以把 position phase 加入扩展状态；然而 `p→p+1` 或 RoPE phase rotation 自带中性方向，扩展系统会人为产生零 Lyapunov。若后续执行 native-position 控制，应把结果称为 `generation-aligned conditional Lyapunov`，不能用该中性方向宣称 edge of chaos；该控制已从下一轮第一优先级后置。

## 6. rolling Jacobian 的完整分解

定义每个输入 token 对新 token 的 block：

\[
J_\ell
=\frac{\partial g_\theta(X)}{\partial x_\ell}
\in\mathbb R^{H\times H},
\qquad \ell=1,\ldots,L.
\]

new-token innovation Jacobian 是：

\[
J_{\rm new}
=[J_1,J_2,\ldots,J_L]
\in\mathbb R^{H\times LH}.
\]

完整 rolling state Jacobian 是 block companion matrix：

\[
J_F=
\begin{bmatrix}
0&I&0&\cdots&0\\
0&0&I&\cdots&0\\
\vdots&&&\ddots&\vdots\\
0&0&0&\cdots&I\\
J_1&J_2&J_3&\cdots&J_L
\end{bmatrix}
\in\mathbb R^{LH\times LH}.
\]

因此：

\[
\|J_F\|_F^2=(L-1)H+\sum_{\ell=1}^{L}\|J_\ell\|_F^2,
\]

\[
\rho_{\rm seq}^2
=\frac{\|J_F\|_F^2}{LH}
=\frac{L-1}{L}
+\frac1{LH}\sum_{\ell=1}^{L}\|J_\ell\|_F^2.
\]

当 `L=64` 时，即使 `J_new=0`：

\[
\rho_{\rm seq}=\sqrt{63/64}\approx0.99216.
\]

这解释了现有实验的 total Frobenius≈1为什么不是临界证据。

但 full-state Benettin Lyapunov 仍然必要。shift-only 虽然单步 Frobenius 接近1，却在 `L` 步后把所有扰动移出窗口；只有 `J_1..J_L` 把扰动重新注入新 token，才可能产生长期扩张。因此长期 Jacobian 连乘能够区分机械 shift 和真正反馈，而单步 full Frobenius 不能。

## 7. token-level 应该具体指什么

推荐同时报告三种归一化：

\[
\rho_{\ell}=\frac{\|J_\ell\|_F}{\sqrt H},
\]

\[
\rho_{\rm new,out}
=\frac{\|J_{\rm new}\|_F}{\sqrt H}
=\sqrt{\sum_\ell\rho_\ell^2},
\]

\[
\rho_{\rm new,input}
=\frac{\|J_{\rm new}\|_F}{\sqrt{LH}}
=\frac{\rho_{\rm new,out}}{\sqrt L}.
\]

- `rho_l`：第 `l` 个上下文 token 的局部贡献；
- `rho_new,out`：每个新-token output feature 接收的总上下文 RMS gain；
- `rho_new,input`：按全部输入自由度平均的 gain。

这些矩形 Jacobian 没有论文中普适的“等于1即临界”阈值。尤其不能只计算：

\[
J_L=\frac{\partial e_{t+1}}{\partial x_{t,L}}
\]

并把它称为“整个 next-token Jacobian”。Transformer 的 attention 允许所有历史位置贡献到新 token；最后 block 只是 profile 的一个位置。

## 8. 对现有结果的重新定位

现有 rolling main 已经回答：

- reset-position soft rolling 的 full-state Lyapunov 随训练从 step0 全负变为训练后正负混合；
- total Frobenius 0.9929–0.9938，平方贡献99.67%–99.85%来自 shift identity；
- hard reset-position argmax 在三个训练后 checkpoint 的32/32 anchors进入有限窗口周期；
- 这些结果说明相位变化，但没有证明最佳性能 checkpoint 自组织到零 Lyapunov。

工程审计后的限定是：上述32/32周期可能同时受到 greedy argmax、64-token历史截断、位置重置、未保留原始 sink 和 soft/off-manifold 状态的影响。它不是“Pythia 原生增长前缀生成必然进入周期”的证据。

尚未回答：

- 原生增长前缀和 sink-preserving fixed-memory 是否复现 R 的退化/周期；
- 同一 validation token 上的 CE/PPL 是否与 `|lambda|` 相关；
- `J_new` 主要来自最后 token 还是分布在历史上下文；
- soft phase 是否与真实增长前缀的 hard behavior一致。

## 9. 推荐追加实验（精简）

正式协议见 `plan/generation_aligned_rolling_followup_plan.md`。执行顺序只有两级：

1. 行为门控：在相同长文本和 target 上比较 N/R/S 的 CE/PPL、hard generation、重复/周期及 attention mass；
2. 切向 pilot：只有行为门控支持固定记忆近似后，才比较 R/S 的 full-state Lyapunov 和 token-block attribution。

原计划中的 native/absolute-position soft cocycle、窗口长度扫描和 differentiable KV-cache JVP 全部后置。原因不是这些问题无价值，而是它们不能先回答“旧 rolling 是否对应合理工程用法”这一基础问题。

最终结论仍必须分栏为 `custom autonomous evidence`、`fixed-memory engineering control`、`native hard generation behavior` 和 `teacher-forced task performance`，不能合并成一个“LLM edge-of-chaos”数字。

## 10. Rolling、Attention Sink 与当前实现核查

### 10.1 当前代码到底做了什么

对 `src/rolling_dynamics.py` 的逐路径核查结果如下：

| 路径 | 输入状态 | position | cache | 窗口更新 |
|---|---|---|---|---|
| soft rolling | `[1,64,H]` continuous embeddings | 每步 `0..63` | `use_cache=False` | 丢最早 state，追加 soft embedding expectation |
| hard rolling | 64个离散 token IDs | 每步 `0..63` | `use_cache=False` | 丢最早 token，追加 greedy argmax token |

attention mask 全为1；模型内部 causal mask 仍限制每个位置只能看左侧。每步都重新计算窗口内全部 Q/K/V，并没有从上一步滚动 KV cache。`src/model_utils.py` 的分析加载路径也关闭了 cache。

缓存中的 Pythia-70m 配置为 GPT-NeoX dense causal attention，`max_position_embeddings=2048`、RoPE、`use_cache=true`；但没有 `sliding_window` 或 sink 配置。因此：

- Pythia 在2048以内的原生使用是增长前缀 + 普通动态 KV cache；
- `SlidingWindowCache` 不能因为 Transformers 提供了类就自动变成 Pythia 原生能力；它面向配置声明支持 sliding-window attention 的模型；
- 当前环境 Transformers 4.57.6 的 `SinkCache` 是迁往外部 custom-generation repository 的 `NotImplementedError` stub，项目中也没有替代实现；
- 所以上一轮实验**没有实现 attention sink**。

### 10.2 Attention Sink 是否是 rolling 能正常运行的必要条件

需要区分两种“滑窗”：

1. **naive rolling KV cache**：直接逐出旧 KV 而不重算。StreamingLLM 论文发现，初始 token 往往承担 attention sink；把它们逐出会造成 perplexity 崩溃。保留少量初始 sink token，并正确处理 cache-relative position/RoPE，是高效长期流式生成的关键。
2. **sliding window with full recomputation**：每一步用当前窗口重新计算全部 K/V。StreamingLLM 将它作为慢但质量稳定的基线；它不依赖持久 KV，因此不是“不加 SinkCache 就无法执行”。一种与该观察相容的机制解释是：重算后，当前窗口的起始 token 可重新承担局部 sink 作用；这应视为推断，而不是本项目已经测得的 attention 事实。

当前 R 在计算结构上接近第2种，而不是失败的 naive KV eviction；但它又额外使用 soft embedding expectation 和每步位置重置，所以不能直接继承论文对 hard-token recomputation baseline 的质量结论。Attention Sink 解释提示我们增加 S 对照，却不能事后证明 R 已经工程对齐。

### 10.3 对上一轮 rolling 结论的影响

| 旧结论 | 审计后状态 |
|---|---|
| total normalized Frobenius≈1主要来自 shift identity | 保持有效，且与 cache/sink 无关 |
| R 的 soft Benettin Lyapunov 随 checkpoint 攻变 | 保持有效，仅属于 R 自治算子 |
| 训练后 hard R 在32/32 anchors检出周期 | 保持可复现，但不能外推为 native generation 周期 |
| rolling 是正常/应用对齐的 Pythia 滑窗用法 | 不成立；需要 N/R/S 行为门控 |
| 加 attention sink 会自动修复 soft dynamics | 不成立；sink 不解决 soft expectation 离开离散 token manifold 的问题 |

最关键的可证伪条件是：若 R 周期化或 PPL 明显变差，而 S/N 不发生，则上一轮“应用场景相位变化”的解释应撤回；数值结果本身不删除，只降级为 custom-operator evidence。

### 10.4 依据

- 原始 edge-of-chaos 方法：[Feng et al., 2019](https://arxiv.org/abs/1909.05176)。
- Attention Sink、naive window KV cache 失败及 recomputation baseline：[StreamingLLM / Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)。
- Dynamic/SlidingWindow/Sink cache 的官方接口语义：[Hugging Face KV cache 文档](https://huggingface.co/docs/transformers/v4.48.2/kv_cache)。
