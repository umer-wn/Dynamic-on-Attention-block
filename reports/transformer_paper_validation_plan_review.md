# Transformer 论文验证主实验计划反思与修订结论

## 1. 总体判断

滑动窗口 next-token operator 的方向值得做，因为它比 `final_hidden → inputs_embeds` 更接近语言模型的实际任务：每一步只预测一个新 token，并保持固定维度以形成可迭代闭环。

但原始计划若直接执行，存在明显假阳性和预算失控风险。最严重的问题不是 temperature 或 sample 数，而是 rolling-window 的确定性 shift 会天然把 normalized Frobenius 推到1附近：

$$
\rho_{\mathrm{shift}}=\sqrt{\frac{L-1}{L}}.
$$

当 $L=64$ 时：

$$
\rho_{\mathrm{shift}}\approx0.9922.
$$

因此，这个实验即使得到“归一化 Jacobian≈1”，也不能直接声称还原论文。只有 Lyapunov、innovation Jacobian 和 hard-token behavior 同时支持，才有论文验证价值。

评审后的结论是：

> 主方向合理，但必须按 v1.1 分级协议先做小 pilot；原来的全组合 main 不应直接启动。

## 2. 主要问题清单

| 优先级 | 原计划问题 | 风险 | 修订 |
|---|---|---|---|
| P0 | shift identity 天然给出 Frobenius≈1 | 结构性假阳性 | shift-only、innovation decomposition、Lyapunov为强制对照 |
| P0 | soft expected embedding 与 hard generation 不同 | 可微结果可能无生成行为意义 | soft主JVP、hard主行为、teacher-forced控制并列 |
| P0 | 4×32×512 再叠加JVP/温度 | 已不属于小型实验 | 行为层32 anchors无JVP；切向层8 anchors；温度只做小子集 |
| P1 | 假设WikiText有8条≥1024 token单行样本 | 数据筛选可能不可执行 | 先重建document并生成manifest |
| P1 | 4 anchors/文档被当成32独立样本 | 误差条过窄 | document-level bootstrap/mixed effects |
| P1 | reset position被描述成原生生成 | operator语义夸大 | 明确 autonomous reset-position；absolute-position仅作非自治控制 |
| P1 | innovation Jacobian归一化未定义 | 矩形映射阈值混乱 | 同时报 total contribution 与 per-output RMS；后者无论文阈值1 |
| P1 | Poincaré投影完整窗口 | shift部分可能支配几何 | 同时投影full window和newest token；每anchor分面 |
| P2 | 只预设“正Lyapunov向0下降” | 结论导向性过强 | 改为高性能附近 $|\lambda|$ 是否最小，不预设方向 |
| P2 | full-block训练任务过难且易collapse | 低优先实验可能无效 | 2k pilot、固定target codebook、mean predictor与CE baseline |

## 3. 最重要的不合理点：接近1可能由窗口结构保证

rolling operator：

$$
F(X)=[x_2,\ldots,x_L,g_\theta(X)].
$$

前 $L-1$ 个输出只是把输入位置复制到左侧，因此 Jacobian 的大部分是 identity blocks。完整 Frobenius 分解：

$$
\rho_{\mathrm{total}}^2
=\frac{L-1}{L}
+\frac{\|J_{\mathrm{new}}\|_F^2}{LH}.
$$

这意味着：

- `total Frobenius≈1` 几乎是设计结果；
- 窗口越长，shift baseline越接近1；
- 用更长序列反而会让假阳性更强；
- 不同 checkpoint 的 total Frobenius 差异可能非常小。

所以主图不能只画 total Frobenius。必须画：

```text
total rho
shift-only rho
rho_total^2 - (L-1)/L
innovation per-output gain
Lyapunov
```

真正有区分度的量是：

$$
\Delta_{\mathrm{innovation}}
=\rho_{\mathrm{total}}^2-\frac{L-1}{L}.
$$

## 4. soft operator 的合理性与局限

soft operator：

$$
e_t^+=\operatorname{softmax}(\ell_t/T)E.
$$

优点：

- 连续可微；
- 可计算JVP；
- 只使用最后位置next-token logits；
- 比full-hidden回灌更贴近语言模型任务。

问题：

- expected embedding通常不是真实token embedding；
- 多步后窗口全部由soft embeddings组成，仍产生distribution shift；
- 大温度可能收敛到词表平均embedding；
- 小温度可能softmax饱和，梯度趋零；
- soft相位不一定对应argmax/sampling行为。

因此 hard argmax 不是装饰性图，而是 soft tangent result 的行为效度检查。若 soft 显示临界、hard 却立即进入单token重复，则不能把 soft 结果解释为 LLM 生成临界性。

## 5. 长序列 sample 设计的修正

原想法“选择更长序列逐token观察”有两种不同含义。

### Autonomous rollout

只有初始 $L$ 个真实token进入状态。生成 $L$ 步后，真实token全部移出窗口。更长原文不会继续驱动系统。

因此长文档的价值是提供多个语义相关但位置不同的初始anchor，而不是让一条真实长序列持续进入自治系统。

### Teacher-forced observation

每步追加真实next token，可以沿长文逐token观察，但它是外部驱动系统：

$$
X_{t+1}=F(X_t,w^{\mathrm{true}}_{t+1}),
$$

不再是论文的单一自治算子 $F(X_t)$。

两者必须分开输出，不能把 teacher-forced stability 作为 autonomous edge-of-chaos 证据。

## 6. position id 的两难

### Reset position

每个step都用 `0...L-1`，保持单一自治算子。但token移位后位置编码改变，不等同于真实长文本生成。

### Absolute position

位置持续增长，更接近生成，但算子显式依赖时间：

$$
X_{t+1}=F_t(X_t).
$$

论文的固定点、周期轨道和Poincaré理论不能原样套用。

因此主实验选择 reset position 是合理折衷，但结论名称必须限定为：

> autonomous reset-position rolling operator

不能写成“原生 autoregressive LLM dynamics”。

## 7. 预算反思

原 main 单温度就有：

$$
4\times32\times512=65536
$$

条 state transitions。

考虑 reference/nearby、soft/hard/teacher-forced、Lyapunov probes 和 Frobenius probes 后，实际模型调用远大于该数；三温度会再次乘3。原计划与“小型实验”不符。

评审后分级：

### A1 行为层

```text
4 checkpoints
8 documents × 4 anchors
512 steps
soft trajectory + hard argmax
no JVP
```

### A2 切向层

```text
4 checkpoints
8 documents × 1 anchor
burn256 + eval128
T=1
Frobenius 4 states × 8 probes
Lyapunov 2 probes
```

### A3 敏感性

```text
step0 + step143000
4 anchors
T=0.7/1.0/1.3
```

先由pilot实测每forward/JVP耗时，再决定main，不提前承诺全部组合。

## 8. 统计反思

同一文档的4个anchor共享主题、词汇和长程上下文，不能视为4个独立样本。若直接以32 anchors计算标准误，会夸大显著性。

主统计单位应为document：

$$
\bar y_s=\frac1{A_s}\sum_a y_{s,a},
$$

然后用8个 $\bar y_s$ 计算 checkpoint mean/std 或 bootstrap CI。anchor点可以显示，但置信区间必须按document resample。

四个checkpoint也不足以建立可靠的“性能–临界距离”函数关系，只能做有方向的案例研究。若后续要声称训练轨迹，应增加更多缓存checkpoint；在没有更多权重前不得对四点相关系数做强解释。

## 9. Poincaré 与周期判断反思

rolling window天然携带 $L$ 步延迟结构。full-window随机投影可能主要看到旧token依次移位，而不是新token innovation。

必须同时输出：

- full-window projection；
- newest-token projection；
- token ID cycle/repetition；
- 每document/anchor独立Poincaré；
- 合并图中明确sample/anchor编码。

visible clusters不能直接解释为周期长度。hard argmax 若进入离散token周期，应通过精确token-window重复检测确认，而不是仅凭投影点数。

## 10. 对实验B的简要反思

低优先级 full-embedding predictor 方向可以保留，但当前主任务 `next-block embedding prediction` 仍有设计风险：

- decoder-only causal输出的早期位置看不到完整输入block，却要预测远未来位置；
- fixed target embedding 的尺度/whitening会直接改变Jacobian数值；
- continuous regression可能学习mean embedding；
- WikiText-2对10M–20M模型可能不足以稳定建立结论；
- 若目标含identity或shift成分，会再次人为得到norm≈1。

因此B启动前需要单独计划，比较：

1. causal shifted-next-embedding；
2. full-context encoder/seq2seq next-block；
3. standard CE baseline；
4. mean predictor/collapse baseline。

在实验A完成前不建议投入B的20k训练。

## 11. 评审后推荐的最小pilot

```text
checkpoints: step0, step143000
documents: 2
anchors/document: 2
window: 64
state burn-in: 128
eval: 128
temperature: 1.0
position: reset
soft + hard + shift-only
Frobenius: 2 states × 4 probes
Lyapunov: 1 probe
```

pilot只回答：

1. 算子尺寸和JVP是否正确；
2. total Frobenius是否被shift baseline完全解释；
3. innovation gain是否可稳定估计；
4. soft轨迹是否快速塌缩；
5. hard轨迹是否立即重复；
6. step0与step143000是否存在足以继续main的差异；
7. 单任务GPU时间和显存是多少。

如果pilot没有显示任何 innovation/Lyapunov 差异，就不应扩大样本和温度扫描。

## 12. 最终建议

主实验A可以执行pilot，但需要降低对“恢复原论文结论”的预期。更准确的研究问题应改为：

> 当 Transformer 被构造成 next-token 对齐的固定窗口自治算子时，去除shift identity后，其 innovation tangent dynamics 是否随预训练性能接近零 Lyapunov？

这个问题可证伪，也不会因为 total Frobenius 天然接近1而得到预设答案。

当前合理执行顺序：

1. dataset/document preflight；
2. rolling/shift解析单测；
3. 两checkpoint、四anchor pilot；
4. 写pilot报告和GPU预算；
5. 决定是否进入A1行为层；
6. 只有A1有差异才进入A2切向层；
7. A完成后再重新评审B。
