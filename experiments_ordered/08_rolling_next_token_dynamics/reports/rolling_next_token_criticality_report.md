# Rolling next-token Transformer 动力学实验报告

状态：**主实验完成（低优先级 full-embedding training 仍未启动）**  
最近更新：2026-07-13  
计划：`plan/transformer_paper_validation_small_experiments_plan.md`  
代码：`src/rolling_dynamics.py`、`scripts/compute_rolling_next_token_dynamics.py`  
数据根目录：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality`

参数、公式和逐图说明：`reports/rolling_next_token_visualization_guide.md`  
token-block Jacobian 后续：`plan/rolling_token_block_jacobian_followup_plan.md`

## 1. 本轮要检验什么

原论文把一个定长、高维状态反复送入同维映射，并考察训练是否把系统推到稳定与混沌的边缘。本实验不重新训练 Pythia，而是把已训练的 causal LM 变成一个可反复迭代、仍然执行 next-token prediction 的定长算子：

\[
X_t=[e_{t-L+1},\ldots,e_t]\in\mathbb R^{L\times H},
\]

\[
p_t=\operatorname{softmax}(\operatorname{LMHead}(f_\theta(X_t)_{-1})/T),
\qquad
\bar e_{t+1}=p_t^\top E,
\]

\[
F_\theta(X_t)=[e_{t-L+2},\ldots,e_t,\bar e_{t+1}].
\]

这里没有训练新模型；`step0`、`step1000`、`step16000`、`step143000` 是 Pythia 训练过程保存的四个权重 checkpoint。动力学中的一个 `step` 是对同一冻结模型执行一次“窗口左移 + 预测并追加一个新 embedding”，不是一次 optimizer update。

主假设不是“rolling window 必然复现原论文结论”，而是：随着语言模型训练，rolling next-token 算子的长期稳定性、局部创新敏感性和 hard token 轨道会发生可重复的相变，并可能经过 \(\lambda_{\max}\approx0\) 的区域。

## 2. 数据流与关键约定

- 数据：WikiText-2 raw validation；先按标题重建文章，再从明确的 `document_index` 和 `anchor_offset` 取连续 64 token。
- 尺寸：token IDs 为 `[B,L]=[1,64]`；输入 embedding 状态为 `[B,L,H]=[1,64,512]`；最后位置 hidden 为 `[1,512]`；logits/probability 为 `[1,V]`；期望 embedding 为 `[1,512]`；输出状态仍为 `[1,64,512]`。
- causal attention：模型在每一步并行处理整个窗口，但 attention mask 是下三角 causal mask；只读取最后位置来预测下一 token。它是自回归 next-token operator，不是 full-sequence embedding reconstruction。
- padding/mask：每个样本独立运行且长度固定为 64，因此无 padding；显式 `attention_mask=1`。不存在 padding token 参与 Jacobian 的问题。
- position：每一步位置均重置为 `0..63`，从而得到自治映射 \(X_{t+1}=F(X_t)\)。这是 rolling-window 主协议的定义，也意味着它不是对无限绝对位置的原始生成过程。
- soft 路径使用词表分布的期望 embedding，保证对输入可微；hard 路径使用 `argmax` token，只分析离散轨道和精确周期，不计算 JVP。
- 所有权重冻结；实验只做前向、JVP 和轨道迭代。

### 2.1 工程语义审计更正（2026-07-14）

代码复核确认：soft 与 hard rolling 都在每一步对当前64-token窗口做全量重算，`position_ids=0..63`、`use_cache=False`。项目没有使用 Pythia 的 `DynamicCache`，也没有实现 Attention Sink 或模型原生 sliding-window attention。Pythia-70m 本身是 dense causal attention + RoPE，配置中没有 `sliding_window`。

因此本报告中的 rolling 应完整读作：**recency-only、reset-position、full-recompute autonomous operator**。它是可复现的研究算子和固定维动力系统，但不是 Pythia 的默认工程生成路径。其总 Frobenius、Lyapunov、nearby distance、投影和 hard cycle 均不因这一更正而失效；改变的是可外推范围。

Attention Sink 主要处理高效 rolling KV cache 逐出初始 sink token 后的性能崩溃。当前实验并未滚动 KV cache，而是全窗口重算，所以不是典型的 naive KV-eviction failure；但它也没有保留原始 sink，且 soft 路径会离开离散 token manifold。下一轮必须用 native growing-prefix (`N`) 与 sink-preserving fixed-memory (`S`) 对照，才能判断当前 rolling (`R`) 的周期和相位是否具有应用意义。

## 3. 为什么必须分解 Jacobian

rolling 算子的 Jacobian 有固定块结构：

\[
J_F=
\begin{bmatrix}
0&I&0&\cdots&0\\
&&\ddots&&\\
\frac{\partial\bar e_{t+1}}{\partial e_{t-L+1}}&\cdots&
\frac{\partial\bar e_{t+1}}{\partial e_t}
\end{bmatrix}.
\]

前 \(L-1\) 个 token 的平移块是精确恒等映射，所以即使新 token 完全不依赖输入，按全状态维度 \(LH\) 归一化也有

\[
\rho_{\rm shift}
=\frac{\|J_{\rm shift}\|_F}{\sqrt{LH}}
=\sqrt{\frac{L-1}{L}}
=0.9921567\quad(L=64).
\]

因此“总归一化 Frobenius 接近 1”几乎是窗口设计的代数结果，不是临界性的证据。报告必须同时给出：

1. 总量 \(\rho_{\rm total}=\|J_F\|_F/\sqrt{LH}\)；
2. shift-only 解析基线 \(\rho_{\rm shift}\)；
3. innovation-total \(\|J_{new}\|_F/\sqrt{LH}\)；
4. innovation-output \(\|J_{new}\|_F/\sqrt H\)；
5. shift 对 \(\rho_{\rm total}^2\) 的贡献比例。

JVP 使用 Hutchinson 恒等式估计 Frobenius：对 Rademacher 探针 \(v_k\)，计算

\[
Jv_k=\left.\frac{d}{d\epsilon}F(X+\epsilon v_k)\right|_{\epsilon=0},
\qquad
\|J\|_F^2\approx\frac1K\sum_{k=1}^K\|Jv_k\|_2^2.
\]

它不是“用多个文本样本拼出完整 Jacobian”；每个状态内部用多个随机方向估计该状态的 Jacobian 范数，再把多个状态/文本作为统计重复。

最大 Lyapunov 使用 Benettin 重归一化：沿实际轨道逐步计算 \(v_{t+1}=J_t v_t\)，每步记录增长率并把方向重新归一化，最终

\[
\lambda_{\max}\approx\frac1N\sum_{t=0}^{N-1}\log\|J_t\hat v_t\|_2.
\]

正值表示局部扰动沿长期主方向指数放大，负值表示收缩；它与单步 Frobenius 不等价。

## 4. Pilot 计划与执行记录

目的：用训练起点和最终 checkpoint 做低成本方向门控，避免直接启动全量实验。

| 项目 | 设置 |
|---|---|
| checkpoints | `step0`、`step143000` |
| 文本/anchor | 2 篇长文档 × 2 offsets = 4 anchors/checkpoint |
| window | 64 tokens，状态维度 32768 |
| soft trajectory | burn-in 128 + eval 128 |
| hard rollout | 256 steps |
| temperature | 1.0 |
| perturbation | \(10^{-3}\) |
| Frobenius | 2 states × 4 probes |
| Lyapunov | 128 states × 1 probe |
| projections | 3 个固定随机方向，seed 1234 |
| GPU | step0: GPU5；step143000: GPU6 |

每个 anchor 约 8 秒；两 checkpoint 总计约 64 秒。原始 JSONL 保持只读，聚合 CSV、图和 figure manifest 分别写入 pilot 的 `processed/` 与 `figures/`。

## 5. Pilot 结果

### 5.1 checkpoint 聚合

| checkpoint | anchors | total Frobenius | innovation-output | Lyapunov | 正 Lyapunov 比例 | shift 平方贡献 | hard 周期比例 | 周期中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step0 | 4 | 0.992885 ± 0.000030 | 0.3041 ± 0.0063 | -0.024765 ± 0.000251 | 0/4 | 99.853% | 0/4 | 未在 256 步内检测到 |
| step143000 | 4 | 0.993562 ± 0.000956 | 0.3383 ± 0.2281 | 0.096179 ± 0.088276 | 3/4 | 99.718% | 4/4 | 12 |

### 5.2 样本级关键结果

- `step0` 四个 anchor 的 \(\lambda_{max}\) 都在 -0.025 附近，最终/初始 nearby separation 为 0.00026–0.00034；soft 轨道表现为收缩。
- `step143000` 的三个 anchor 为正 Lyapunov（0.108、0.129、0.177），nearby separation 达 1251–3896，并在 hard 路径形成长度 7、11、13 的短周期。
- `step143000 doc219@0` 是重要反例：innovation-output 仅 0.0109、Lyapunov -0.029，但 hard 路径仍在第 83 步后进入长度 112 的周期。训练后 checkpoint 不是对所有输入都处于同一相位。
- `step0` 的词表分布接近高熵均匀分布（entropy 10.626、top1 probability 0.00023），其期望 embedding 很快进入一个收缩的平均态；这不是“语言能力好”的稳定点。
- `step143000` 分布更尖锐（平均 entropy 3.80、top1 probability 0.439），长期轨道更活跃且强烈依赖文本。

图：

- `/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot/figures/pilot_total_frobenius_vs_shift.png`
- `/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot/figures/pilot_innovation_vs_lyapunov.png`
- `/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot/figures/pilot_soft_hard_behavior.png`

## 6. Pilot 能证明与不能证明什么

当前可以证明：

1. rolling-window 的总 normalized Frobenius 接近 1 主要是 shift identity 的结构性假阳性；不做分解的实验设计不成立。
2. 在固定 T=1、reset-position 算子下，训练起点与最终 checkpoint 的长期扰动行为明显不同。
3. 最终 checkpoint 存在显著 anchor 异质性；只取一个 prompt 或只比较 checkpoint 均值会造成错误结论。
4. hard argmax 的精确周期与 soft expectation 的 Lyapunov 是不同问题：负 soft Lyapunov 也可能对应很长的 hard 离散周期。

当前不能证明：

1. 训练把 Transformer 单调推向 \(\lambda=0\)；中间 checkpoint 尚未测量。
2. 最终模型是“临界”的；pilot 中多数最终 anchor 明显为正 Lyapunov，而非接近零。
3. 该结果复现了原论文的训练机制；Pythia 使用 next-token CE，且 rolling shift 没有原论文同维全输出算子的直接对应物。
4. 四个 anchor 足以代表语料分布，或一个随机 Lyapunov 探针足以稳定估计最大指数。
5. 软期望 embedding 轨道等同于真实生成；它只是为可微审计构造的松弛算子。

## 7. 主实验门控决策与计划

门控：**通过，但必须分两层执行并保留样本分布。** 原因是 checkpoint 差异大、运行成本低，同时 anchor 异质性足以否定“小样本均值即可”的方案。

### 7.1 Behavior layer

- 四 checkpoints；8 篇长文档 × 4 anchors = 32 anchors/checkpoint；所有 checkpoint 使用完全相同的 token windows。
- burn-in 512 + eval 256；hard 512。
- 记录 soft trajectory、nearby distance、entropy/top1、固定投影和精确 full-window hard cycles。
- 不计算 Frobenius/Lyapunov，避免把 GPU 时间浪费在 128 个重复 tangent 审计上。

### 7.2 Tangent layer

- 四 checkpoints；同 8 篇文档，每篇固定 anchor 0。
- burn-in 256 + eval 128。
- Frobenius 4 states × 8 probes；Lyapunov 128 states × 2 probes。
- 不重复 hard rollout。

主实验先比较训练轨迹是否经过零 Lyapunov、innovation 是否与 Lyapunov/硬周期共同变化，再决定是否需要温度、window length、absolute-position 或更多 checkpoint 扩展。任何扩展都不得把 total Frobenius≈1 单独作为成功标准。

### 7.3 主实验执行结果

四个 checkpoint 均完整完成：behavior 为 32/32 anchors，tangent 为 8/8 anchors。验证脚本确认每个 checkpoint 使用完全相同的 `(document_index, anchor_offset)` 和初始 token IDs；window 64、T=1、reset position 也完全一致。

| checkpoint | behavior n | tangent n | total Frobenius | innovation-output | Lyapunov mean ± std | 正 Lyapunov | hard cycle | 周期长度中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step0 | 32 | 8 | 0.992905 ± 0.000019 | 0.3083 ± 0.0040 | -0.02555 ± 0.00052 | 0/8 | 0/32 | 未在 512 步内检测到 |
| step1000 | 32 | 8 | 0.993550 ± 0.000859 | 0.3490 ± 0.1538 | 0.01781 ± 0.05548 | 6/8 | 32/32 | 4 |
| step16000 | 32 | 8 | 0.992907 ± 0.000523 | 0.2830 ± 0.1085 | 0.03071 ± 0.08042 | 4/8 | 32/32 | 11.5 |
| step143000 | 32 | 8 | 0.993820 ± 0.001949 | 0.2785 ± 0.2750 | 0.06684 ± 0.09927 | 5/8 | 32/32 | 13 |

![四 checkpoint 训练动力学总览](assets/rolling_next_token/main_training_dynamics_overview.png)

![shift 分解、innovation 与 Lyapunov](assets/rolling_next_token/main_tangent_decomposition.png)

主结果解释：

1. **否定 total Frobenius≈1 作为临界证据。** 四 checkpoint 的 total Frobenius 全落在 0.9929–0.9938，而 shift-only 已为 0.9921567；shift 对平方范数的平均贡献仍为 99.67%–99.85%。这个量几乎无法区分明显负 Lyapunov 与明显正 Lyapunov 的状态。
2. **训练确实改变了相位分布，但不是平滑、单调地逼近零。** step0 的 8/8 tangent anchors 都稳定收缩；三个训练后 checkpoint 都同时含正负 Lyapunov。checkpoint 均值从负值跨过零，但此后逐渐更正，而不是停在零附近。
3. **step1000 的均值最接近零不等于整个 checkpoint 临界。** 它的标准差 0.0555，8 个样本中 6 个为正，说明“均值接近零”由相位混合造成，不能替代样本级判定。
4. **hard token 动力学出现清晰离散变化。** step0 在 512 步内没有任何 exact full-window cycle；三个训练后 checkpoint 都是 32/32 检出周期。短周期与更尖锐的 token 分布共同出现，但这是 hard argmax operator 的结论，不应直接等同于 soft expectation 的 Lyapunov 相位。
5. **innovation Frobenius 不是 Lyapunov 的充分替代。** checkpoint 均值并不随训练或 Lyapunov 单调变化，且最终 checkpoint 的样本标准差几乎等于均值。单步平均敏感性没有包含方向对齐与跨步乘积信息。
6. **数值一致性支持 Lyapunov 的解释。** 在 tangent anchor 与 behavior anchor 的 32 个配对样本上，Lyapunov 与 `log10(final/initial nearby separation)` 的 Pearson/Spearman 分别为 0.679/0.646；innovation-output 与同一量仅为 0.224/0.261。这不是独立证明（两者共享轨道和 Jacobian），但说明实现内部没有出现明显方向矛盾。

### 7.4 对原论文式假设的当前判定

| 假设 | 状态 | 当前证据 | 限制 |
|---|---|---|---|
| 训练会改变循环算子的稳定相位 | 支持 | step0 全负；训练后出现正/负混合且 hard 周期率跃迁 | 只有一个模型规模和四个 checkpoint |
| 训练使系统自组织到临界点 | 不支持/未决 | 均值跨零但不单调停留；最终均值明显为正 | checkpoint 稀疏，尚未联合验证 loss/任务性能 |
| normalized Frobenius≈1 能标识临界 | 证伪（对本 rolling 定义） | 99.67%–99.85% 的平方贡献来自解析 shift identity | 结论针对 rolling full-state normalization |
| innovation sensitivity 能替代长期稳定性 | 不支持 | innovation 与 Lyapunov/nearby 的关系弱且非单调 | 仍需更多 probe 和 window-length control |
| 训练后 hard 生成进入吸引周期 | 支持（本协议内） | 三个训练后 checkpoint 均 32/32 在 512 步内精确重复完整窗口 | reset position、argmax、有限检测步数 |

最稳妥的表述是：**本报告定义的 recency-only、reset-position rolling 算子在训练早期从近均匀、收缩的 soft 平均态转为输入依赖的混合稳定性，并在 hard argmax 路径上快速进入周期吸引子；当前结果显示该自定义算子的相位穿越，但没有证明 Pythia 原生生成发生同样变化，也没有证明自组织临界性。**

## 8. 文件与可复核性

- Pilot 原始结果：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot/{step0,step143000}/raw/`
- Pilot anchor CSV：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot/processed/pilot_anchor_metrics.csv`
- Pilot checkpoint CSV：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot/processed/pilot_checkpoint_summary.csv`
- Pilot figure manifest：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot/figures/pilot_figure_manifest.json`
- Pilot logs：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/pilot/logs/`
- 主实验配置：`configs/pythia_rolling_main_{behavior,tangent}_step*.yaml`
- 主实验数据：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/main_{behavior,tangent}/`
- 主实验聚合：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/main_processed/`
- 主实验全分辨率图：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/main_figures/`
- rolling 专用可视化补充：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/visualization_supplement/`
- 参数、公式与逐图指南：`reports/rolling_next_token_visualization_guide.md`
- token-block Jacobian 后续计划：`plan/rolling_token_block_jacobian_followup_plan.md`

## 9. 未决风险

- 工程语义：当前 rolling 不使用 KV cache、Attention Sink 或原生 sliding-window attention；hard 周期可能由 greedy argmax、64-token截断、位置重置和未保留原始 sink 共同促进。
- `reset position` 是保持自治性的必要约定，但会改变真实长文本生成中的绝对位置语义；后续 position control 需要把位置相位加入状态，不能直接把随时间变化的 position offset 当作同一个自治算子。
- Nearby trajectory 不重归一化，过度收缩时受 float32 数值下限影响；正式稳定性判断以 Benettin Lyapunov 为主。
- Frobenius 是 Hutchinson 估计，主实验增加 probe 但仍应显示状态和样本离散度。
- exact hard cycle 只在有限步数内检测；“未检测到”不等于无周期。
- 本轮不下载新权重，不训练新模型；低优先级 full-embedding training 必须等本实验完成审计后另立 plan/report。

## 10. 下一轮建议（不在本轮自动执行）

关于真实序列逐 token 增长、Jacobian 应采用 token-level 还是 seq-level 的完整推导，见 `reports/rolling_application_scenario_derivation.md`；预注册协议见 `plan/generation_aligned_rolling_followup_plan.md`。

方法结论是：稳定相位继续由 full-state square Jacobian 的 Benettin Lyapunov 判断；任务敏感性采用 `H × (L·H)` new-token innovation Jacobian，并拆成 `L` 个 `H × H` token blocks。last-token block 只是其中一个位置，不能替代全部 attention context，也不能单独定义 rolling 系统的最大 Lyapunov。

执行计划已精简为两级：先比较 native growing-prefix (`N`)、现有 recency-only full recomputation (`R`) 和保留4个初始 sink token 的 fixed-memory recomputation (`S`)；只有 CE/PPL、生成退化和周期行为的工程门控通过后，才执行 R/S 的 Lyapunov 与 token-block pilot。native-position soft cocycle、窗口长度扫描和 differentiable KV-cache JVP 均后置。

优先顺序：

1. 先完成 N/R/S 工程行为门控；若 R 与 S/N 不一致，撤回对原生生成的外推。
2. 通过门控后，对 R/S 固定动态状态计算 full-state Lyapunov，并执行 token-block attribution。
3. 只有上述结果支持时，才增加中间 checkpoint、窗口长度和 position controls。
4. 最后才考虑 full-embedding predictor 重新训练；其目标、collapse baseline、CE baseline 和 checkpoint 审计必须另立计划，不能用“范数接近 1”作为训练成功标准。
