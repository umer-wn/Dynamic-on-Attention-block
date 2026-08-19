# Transformer 上验证 arXiv:1909.05176 的小型实验与训练计划

## 0. 计划状态与边界

- 当前状态：v1.1 评审修订版，仅制定计划，尚不启动实验或训练。
- 核心论文：arXiv:1909.05176，关注同维非线性算子的渐近 Jacobian、Lyapunov、Poincaré 相位与任务性能关系。
- 总目标：构造比 `final_hidden → inputs_embeds` 更接近语言模型任务的 Transformer 闭环，并检验“性能最优附近 normalized Jacobian≈1 / Lyapunov≈0”能否在语言模型中复现。
- 优先级：实验A高；实验B低。
- GPU：优先使用5/6/7。
- 大数据、checkpoint、日志：全部写入 `/home/luohaoming`。
- 每轮必须保留独立 `plan`、config、raw、processed、figure manifest 和中文 report。
- 不覆盖当前 direct-operator 实验；新结果作为并列证据。

## 1. 两个方向

### 实验A：next-token 对齐的滑动窗口闭环（高优先级）

使用冻结的 pretrained Transformer。每次只使用最后位置的 next-token distribution 产生一个新 embedding，删除窗口最旧位置并追加新位置：

$$
X_{t+1}=[X_t[:,2:L,:],\ e^+_t].
$$

这样不再要求模型同时预测整段新的 `[L,H]` hidden state，闭环含义更接近正常 next-token generation。

### 实验B：训练同维 full-embedding predictor（低优先级）

训练小型 Transformer，让整个输入 embedding block 映射到同尺寸的未来 embedding block：

$$
F_\theta:\mathbb R^{L\times H}\rightarrow\mathbb R^{L\times H}.
$$

训练过程中保存 checkpoint，检验任务性能最优 checkpoint 是否对应 normalized Frobenius≈1、Lyapunov≈0。

## 2. 必须提前处理的理论陷阱

### 2.1 滑动窗口会人为带来 Jacobian≈1

滑动窗口算子为：

$$
F(X)=
\begin{bmatrix}
x_2\\
x_3\\
\vdots\\
x_L\\
g_\theta(X)
\end{bmatrix},
$$

其 Jacobian 包含确定性的 shift identity：

$$
J_F=
\begin{bmatrix}
0&I&0&\cdots&0\\
0&0&I&\cdots&0\\
\vdots&&&\ddots&\\
0&0&0&\cdots&I\\
\frac{\partial g_\theta}{\partial x_1}&
\frac{\partial g_\theta}{\partial x_2}&
\cdots&&
\frac{\partial g_\theta}{\partial x_L}
\end{bmatrix}.
$$

仅 shift 部分就贡献：

$$
\|J_{\mathrm{shift}}\|_F^2=(L-1)H.
$$

对总维数 $D=LH$：

$$
\rho_{\mathrm{shift}}
=\frac{\|J_{\mathrm{shift}}\|_F}{\sqrt{LH}}
=\sqrt{\frac{L-1}{L}}.
$$

当 $L=64$：

$$
\rho_{\mathrm{shift}}\approx0.9922.
$$

因此，若滑动窗口 total normalized Frobenius 接近1，可能只是复制/移位结构造成，不能直接当作论文结论复现。

### 2.2 强制设置三个对照指标

实验A必须同时报告：

1. total normalized Frobenius：完整 rolling-window 算子；
2. shift-only baseline：追加常数/零向量的纯移位算子；
3. innovation Jacobian：只测新生成 embedding：

$$
J_{\mathrm{new}}=\frac{\partial e^+_t}{\partial X_t}.
$$

total Frobenius 应满足分解：

$$
\rho_{\mathrm{total}}^2
=\frac{L-1}{L}
+\frac{\|J_{\mathrm{new}}\|_F^2}{LH}.
$$

只有超出 shift baseline 的部分才来自模型 next-token dynamics。

innovation norm 必须同时给出两种归一化，避免矩形 Jacobian 的分母含义混乱：

$$
\rho_{\mathrm{innovation,total}}
=\frac{\|J_{\mathrm{new}}\|_F}{\sqrt{LH}},
$$

它表示 innovation 对完整同维 rolling operator 的增量贡献；以及：

$$
g_{\mathrm{new,output}}
=\frac{\|J_{\mathrm{new}}\|_F}{\sqrt H},
$$

它表示每个新输出 hidden dimension 的 RMS 输入敏感性。后者是矩形映射诊断，没有论文中同维临界阈值1，不能单独套用论文结论。

### 2.3 Lyapunov 是主要相位判据

虽然一次 shift 保留 $L-1$ 个位置，但任一旧位置在 $L$ 步后都会离开窗口。长时间 Benettin Lyapunov 可以判断新 token feedback 是否持续扩张：

$$
\widehat\lambda_{\max}
=\frac1T\sum_t\log\|J_tv_t\|.
$$

所以实验A主结论优先级为：

```text
Lyapunov
> shift-corrected innovation gain
> nearby separation
> total normalized Frobenius
```

不能单独用 total Frobenius≈1 宣称复现论文。

## 3. 实验A：算子定义

## 3.1 可微 soft-next-token rolling operator（主算子）

当前窗口：

$$
X_t=[x_{t,1},\ldots,x_{t,L}]in\mathbb R^{L\times H}.
$$

冻结 Transformer 前向：

$$
\ell_t=\operatorname{logits}_\theta(X_t)_{L,:}\in\mathbb R^V.
$$

只取最后位置 next-token logits，而不是所有位置的 final hidden。

温度概率：

$$
p_t=\operatorname{softmax}(\ell_t/T).
$$

期望 next-token embedding：

$$
e^+_t=p_tE_\theta\in\mathbb R^H.
$$

窗口更新：

$$
F^{\mathrm{soft}}_\theta(X_t)
=[x_{t,2},\ldots,x_{t,L},e^+_t].
$$

特点：

- 与 next-token prediction 对齐；
- 每步只生成一个新位置；
- 固定 `[L,H]`，可迭代；
- 连续、确定性、可计算 JVP；
- 不是实际离散 sampling，但比 full-hidden 回灌更接近语言建模语义。

## 3.2 hard argmax rolling operator（行为控制组）

$$
w^+_t=\arg\max_v\ell_{t,v},
$$

$$
e^+_t=E_\theta[w^+_t].
$$

然后同样 shift+append。

用途：

- 生成真实 token 序列；
- 检查是否进入 token 周期、重复、退化输出；
- 计算轨迹、recurrence 和 token-cycle statistics。

限制：argmax 不可微，不计算 Jacobian/Frobenius/Lyapunov。

## 3.3 stochastic sampling operator（次要控制）

按 temperature/top-p 采样 token 后 append。只用于验证 soft/argmax 轨迹是否代表正常生成，不作为论文确定性动力系统主证据。

## 3.4 teacher-forced sliding observation（非自治控制）

每步追加真实 next token embedding：

$$
X_{t+1}=[X_t[:,2:L,:],E(w^{\mathrm{true}}_{t+1})].
$$

这依赖外部真实 token，严格来说是受驱动/非自治系统，不可用于论文 autonomous edge-of-chaos 结论。用途仅为：

- 比较 autonomous rollout 与真实语料路径偏离速度；
- 记录 next-token CE、PPL、top-k accuracy；
- 判断动力学变化是否只是生成质量崩坏。

## 4. 实验A：position 与 mask

### 4.1 autonomous reset-position（论文对齐主版本）

每个 rolling step 都使用：

```text
position_ids = 0 ... L-1
attention_mask = all ones
```

这样 $F_\theta$ 不显式依赖时间 $t$，保持 autonomous：

$$
X_{t+1}=F_\theta(X_t).
$$

代价是它不等同于标准长文本生成：每次 shift 后位置都重新编号，原来位置1的 token 下一步会使用位置0的位置编码。报告必须把它称为 `autonomous reset-position operator`，不能简称“原生生成动力学”。

### 4.2 absolute-position（生成对齐控制组）

position id 随生成步增加，更接近真实 autoregressive generation，但算子变成：

$$
X_{t+1}=F_{\theta,t}(X_t),
$$

即非自治系统。其轨迹可比较，不能直接套用论文单一 $F$ 的 Jacobian threshold。

### 4.3 padding

选取完整长度初始窗口，主 rollout 不含 padding。若文档尾部不足窗口，直接丢弃该 anchor，不把 padding 引入 rolling attractor。

## 5. 实验A：长序列 sample 设计

## 5.1 为什么要长序列

autonomous rollout 中，初始真实窗口只有前 $L$ 个 token；生成 $L$ 步后，所有真实 token 都已离开窗口。因此“长原始文本”不会直接持续进入 autonomous dynamics。

长序列的正确用途是：

- 从同一语义连贯文档取得多个时间有序 anchor window；
- 建立 teacher-forced 对照路径；
- 比较不同上下文位置的初始吸引域；
- 避免8个短片段只代表极少语义状态。

## 5.2 样本筛选

运行前先做 dataset preflight。WikiText raw 行不一定等于完整文章，不能直接假设存在8条长度≥1024的单行样本。应按文章标题/空行边界重建 document，输出 document-id、token count、anchor offsets manifest；若仍不足8篇，则降低最短长度或改用已缓存的长文本数据，但不得跨 checkpoint 改变样本。

行为层主数据建议：

```text
dataset: WikiText-2 validation（先保持与现有实验一致）
documents: 8
minimum usable tokens per document: 1024
window length L: 64
anchors per document: 4
anchor offsets: 0, 256, 512, 768（按可用长度调整）
total initial windows: 32
```

所有 checkpoint 使用完全相同的 document IDs、token IDs、anchor offsets。

## 5.3 两级运行

### Pilot

```text
checkpoints: step0, step143000
documents: 2
anchors/document: 2
rollout: 256
temperature: 1.0
```

目的：验证算子、position、JVP、shift baseline 和存储字段。

### Main：拆成行为层和切向层

行为层（便宜，不做JVP）：

```text
checkpoints: step0, step1000, step16000, step143000
documents: 8
anchors/document: 4
rollout: 512
operators: soft trajectory + hard argmax
JVP metrics: off
```

切向层（昂贵，主论文证据）：

```text
checkpoints: 4
documents: 8
anchors/document: 1（预注册每篇文档第一个有效anchor）
state burn-in: 256
measurement: 128
temperature: 1.0
Frobenius states/probes: 4/8
Lyapunov probes: 2
Lanczos: 仅2个checkpoint×2个状态
```

温度敏感性只在 `step0/step143000 × 4 anchors` 上跑 `T=0.7/1.0/1.3`。只有出现明确温度依赖且资源允许，才扩展到全部 checkpoint。不得一次性执行 `4 checkpoints × 32 anchors × 3 temperatures × JVP`。

## 6. 实验A：核心指标

### 6.1 语言建模/生成指标

- teacher-forced token-weighted CE/PPL；
- autonomous token entropy；
- unique-token ratio；
- repetition-2/3/4；
- hard argmax cycle length；
- soft distribution entropy；
- soft embedding 到最近真实 token embedding 的距离；
- autonomous 与 teacher-forced window 的距离。

### 6.2 动力学指标

- relative step delta；
- nearby separation；
- Benettin maximal Lyapunov；
- lag recurrence；
- stationarity half-window drift；
- fixed random projection trajectory；
- return map；
- 每 sample Projected Poincaré；
- 多吸引子/吸引域聚类。

投影同时输出两套：

- full-window projection：观察完整 rolling state；
- newest-token/innovation projection：只投影最后一个位置，减少 shift-copy 主导。

所有 Poincaré 先按 `document × anchor` 单独计算，再画分面图；合并图不能用可见点簇数量推断周期或吸引子数量。

### 6.3 Jacobian 指标

- total normalized Frobenius；
- analytically expected shift baseline；
- empirical shift-only JVP baseline；
- innovation Jacobian norm；
- total top singular value（少量状态 Lanczos）；
- innovation top singular value；
- stable rank：

$$
r_s=\frac{\|J\|_F^2}{\sigma_{\max}^2};
$$

- Frobenius–Lyapunov scatter。

## 7. 实验A：预注册假设

### H-A1：任务对齐改善论文可比性

soft rolling operator 的 $|\lambda|$ 或临界距离在高性能 checkpoint 附近更小。方向不预设为“从正值单调下降到0”，因为 Transformer 可能从稳定侧或混沌侧接近，也可能完全不接近。

支持条件：

- 多数文本/anchor 的 $\lambda$ 趋势一致；
- 结果对 burn/eval 和 probe 数稳定；
- 不仅 total Frobenius 接近1，shift-corrected innovation 与 Lyapunov 也接近临界。

### H-A2：total Frobenius≈1主要是 shift artifact

若四个 checkpoint 的 total Frobenius 都接近：

$$
\sqrt{(L-1)/L},
$$

但 innovation gain 和 Lyapunov 不随性能一致变化，则 normalized Jacobian≈1 是窗口结构导致，不能支持论文。

### H-A3：hard token dynamics 与 soft tangent dynamics一致

若 soft operator 的相位变化对应 hard argmax 的周期长度、重复率和吸引子几何变化，则 soft JVP 有行为意义；否则 soft relaxation 只是数学替代。

### H-A4：训练后模型未必趋向临界

如果后期 checkpoint 的 innovation Lyapunov 明显负且生成质量更高，则论文“性能最优靠近边缘”在该 Transformer rolling operator 上被证伪。

## 8. 实验A：配置草案

```yaml
experiment_name: pythia_rolling_next_token_criticality
seed: 1234
device: cuda
dtype: float32
offline: true

models:
  - {name: EleutherAI/pythia-70m, revision: step0}
  - {name: EleutherAI/pythia-70m, revision: step1000}
  - {name: EleutherAI/pythia-70m, revision: step16000}
  - {name: EleutherAI/pythia-70m, revision: step143000}

dataset:
  name: wikitext
  config: wikitext-2-raw-v1
  split: validation
  min_document_tokens: 1024
  documents: 8
  behavior_anchors_per_document: 4
  tangent_anchors_per_document: 1
  window_length: 64

rolling_operator:
  mode: soft_expected_next_token
  temperature: 1.0
  shift_old_tokens: true
  position_mode: reset
  behavior_rollout_steps: 512
  tangent_burn_in_steps: 256
  tangent_measurement_steps: 128
  hard_argmax_control: true
  teacher_forced_control: true

metrics:
  perturbation_epsilon: 1.0e-3
  lyapunov_probes: 2
  frobenius_eval_states: 4
  frobenius_probes: 8
  innovation_probes: 8
  lanczos_states: 2
  lag_windows: [1, 2, 4, 8, 16, 32, 64, 128]
  projection_count: 3
  projection_seed: 1234
  projection_targets: [full_window, newest_token]
```

正式运行前根据 pilot 显存/时间减少或增加 probes，但所有变更写入 config 和 report。

## 9. 实验A：代码改造计划

新增而不破坏现有 direct operator：

```text
src/rolling_dynamics.py
├── soft_next_token_embedding()
├── hard_next_token_embedding()
├── rolling_window_update()
├── shift_only_operator()
├── innovation_jvp()
└── token_cycle_metrics()

scripts/compute_rolling_next_token_dynamics.py
scripts/analyze_rolling_next_token_criticality.py
tests/test_rolling_dynamics.py
```

必须测试：

1. shape `[B,L,H] → [B,L,H]`；
2. 旧位置严格左移一位；
3. soft 末位置等于 `softmax(last_logits/T) @ E`；
4. hard 末位置等于 argmax token embedding；
5. reset position 下相同输入得到相同输出；
6. shift-only normalized Frobenius 等于 $\sqrt{(L-1)/L}$；
7. total Frobenius 分解与 innovation contribution 数值一致；
8. padding anchor 被拒绝或严格 mask；
9. projection seed 跨 checkpoint 一致；
10. hard operator 不进入 JVP 路径。
11. innovation 的两种归一化均与手算小矩阵一致；
12. document/anchor manifest 跨 checkpoint 完全一致；
13. dropout关闭、`model.eval()`，保证算子确定性；
14. reset-position 与 absolute-position 被写入不同实验名，禁止混合聚合。

## 10. 实验A：输出结构

```text
/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/
├── configs_snapshot/
├── pilot/
├── main/
│   ├── raw/
│   ├── processed/
│   ├── logs/
│   └── manifests/
└── figures/

/data1/luohaoming/model_feature/
├── plan/transformer_paper_validation_small_experiments_plan.md
├── reports/rolling_next_token_criticality_report.md
└── reports/assets/rolling_next_token_criticality/
```

## 11. 实验A：主图

1. total Frobenius、shift baseline、innovation contribution 对比；
2. checkpoint×sample/anchor Lyapunov；
3. innovation Frobenius–Lyapunov；
4. loss/PPL–Lyapunov；
5. hard token cycle length/repetition；
6. soft entropy 与 nearest-token distance；
7. reset-position vs absolute-position；
8. teacher-forced vs autonomous window distance；
9. 每 checkpoint 的8 sample/32 anchor Poincaré 小多图；
10. burn/probe/window sensitivity。

## 12. 实验A：成功、证伪和停止条件

### 支持论文 LLM 版本

只有同时满足才支持：

- 最优/后期 checkpoint 的 $\lambda$ 靠近0；
- 早期/低性能 checkpoint 明显偏正或偏负；
- innovation gain 与 Lyapunov方向一致；
- total≈1不是 shift baseline 单独解释；
- 多 sample/anchor 稳定；
- Poincaré/recurrence 提供相容几何证据。

### 证伪或不支持

- total≈1但全部由 shift identity解释；
- Lyapunov与性能无一致关系；
- soft 和 hard behavior完全脱钩；
- 结果只在一个温度、一个 anchor 或一个窗口长度出现；
- 后期高性能 checkpoint 稳定远离0。

### Pilot停止条件

- soft expected embedding快速塌缩到词表均值；
- float32 JVP 不稳定；
- 单 checkpoint pilot 预计超过可接受GPU预算；
- reset position 导致明显非语言行为且 hard control立即重复。

停止不代表放弃方向，应先调整 temperature/window 和位置策略，再决定是否进入main。

## 12.1 统计单位与不确定性

32个 anchor 不是32个独立文档。4个 anchor 来自同一文档，存在强相关，不能把它们当作 i.i.d. 样本计算过窄误差条。

主报告采用分层汇总：

```text
trajectory state
→ anchor metric
→ document mean
→ checkpoint mean/std over 8 documents
```

推荐使用 document-level bootstrap 或 mixed-effects model：

$$
y_{s,a}=\mu_{\mathrm{checkpoint}}+u_s+\epsilon_{s,a},
$$

其中 $u_s$ 是文档随机效应。图中可以显示全部 anchor 点，但置信区间以 document 为重采样单位。

## 12.2 paper-exact 与 windowed protocol 分开

论文对齐基线必须独立输出：

```text
paper-exact-like:
  total iterations: 500
  sample point: final iteration
  nearby: initial vs final
  normalized Jacobian: final-state/sample average
```

当前稳健性协议：

```text
windowed:
  burn-in: 256
  eval: 128
  Benettin tangent initialized after burn-in
```

两者不得合并为一个指标。若结论不同，优先报告窗口依赖性，而不是选择更符合预期的一套。

## 12.3 计算预算复核

原计划 `4 checkpoints × 32 anchors × 512 steps` 已有65,536次 soft operator forward；若同时跑 reference/nearby、hard control、teacher forcing和JVP，成本远超“小型实验”。

修订后：

- 行为层保留32 anchors，但关闭JVP；
- 切向层缩到8 anchors/checkpoint、burn256+eval128；
- temperature只跑2 checkpoints×4 anchors；
- Lanczos只做极小子集。

Pilot必须记录：

```text
seconds/operator forward
seconds/JVP step
peak GPU memory
trajectory JSONL size
```

据此生成正式预算表后才能启动main。

## 12.4 实现效率约束

soft operator只需要最后位置的 vocabulary logits。实现应尽量避免保存 `[B,L,V]` 的完整 logits：

```text
final_hidden[:, -1, :]
→ lm_head
→ logits_last [B,V]
```

同时：

- `use_cache=False`，保持固定窗口映射和JVP正确性；
- `model.eval()`，关闭dropout；
- 不在JVP路径保存不需要的 hidden states/attentions；
- 先测 float32，bfloat16 只作速度控制，不能直接承担微小Lyapunov符号判断。

## 13. 实验B：研究问题（低优先级）

训练一个同维 Transformer operator：

$$
F_\theta:\mathbb R^{L\times H}\to\mathbb R^{L\times H},
$$

使任务性能可以与训练中 checkpoint 的渐近稳定性直接关联。

核心假设：full-block embedding prediction 的最佳验证性能可能出现在：

$$
\rho_{\mathrm{geo}}\approx1,
\qquad
\lambda_{\max}\approx0.
$$

该假设必须允许被证伪；不能通过 identity reconstruction 人为保证 Jacobian=1。

## 14. 实验B：禁止使用的平凡任务

不能训练：

$$
F(X)=X
$$

或以同一输入 block 为目标的 autoencoder identity loss。否则最佳解天然：

$$
J=I,\quad\rho=1,\quad\lambda=0,
$$

这只是把论文结论写进目标函数，不能算实证。

也要警惕显式 shift-copy target，因为它同样可能通过 identity block 把 Frobenius推向1。

## 15. 实验B：推荐任务定义

### 15.1 主任务：next-block embedding prediction

输入 block：

$$
X_t=E(w_{t:t+L-1}).
$$

目标为不重叠的未来 block：

$$
Y_t=\operatorname{sg}\left(E_{\mathrm{target}}(w_{t+L:t+2L-1})\right).
$$

模型一次并行输出：

$$
\widehat Y_t=F_\theta(X_t)\in\mathbb R^{L\times H}.
$$

训练后可自主迭代：

$$
X_{n+1}=F_\theta(X_n).
$$

这是同维 full-block operator，结构上更接近论文，而不是标准 next-token CE。

### 15.2 较容易控制：one-step shifted embedding prediction

$$
Y_i=E(w_{i+1}).
$$

它仍有 next-token语义，只是输出连续 embedding。作为任务难度控制，不作为“非 next-token”主结果。

### 15.3 CE baseline

使用相同架构、数据、训练步数训练标准 next-token CE 模型，确保 full-embedding 结果不是架构或优化器造成。

## 16. 实验B：小模型架构

建议从头训练或以固定 embedding codebook 初始化：

```text
architecture: decoder-only Transformer
layers: 4
hidden size: 256
attention heads: 4
MLP size: 1024
context/block length: 64
dropout: 0 或 0.1（预注册）
parameters: 约10M–20M，取决于词表/embedding设置
```

输出 head：

```text
final hidden [B,L,H]
→ linear H→H
→ predicted future embeddings [B,L,H]
```

## 17. 实验B：target embedding 与 collapse 防护

若 target embedding 与模型同时自由训练，可能出现所有 token embedding 塌缩到常数的平凡解。

优先方案：

- 使用缓存的 pretrained tokenizer embedding 作为固定 target codebook；
- target 分支 stop-gradient；
- 对 target embedding 做固定 whitening/LayerNorm；
- 输入 embedding 可选择固定或独立训练，作为消融。

损失：

$$
\mathcal L_{\mathrm{cos}}
=\frac1{BL}\sum_{b,i}left(1-\cos(\widehat y_{b,i},y_{b,i})\right),
$$

$$
\mathcal L_{\mathrm{mse}}
=\frac1{BLH}\|\operatorname{LN}(\widehat Y)-\operatorname{LN}(Y)\|_2^2,
$$

$$
\mathcal L
=\lambda_{\mathrm{cos}}\mathcal L_{\mathrm{cos}}
+\lambda_{\mathrm{mse}}\mathcal L_{\mathrm{mse}}
+\lambda_{\mathrm{var}}\mathcal L_{\mathrm{variance}}.
$$

初始建议：

```text
lambda_cos = 1.0
lambda_mse = 1.0
lambda_var = 0.1
```

必须监控：

- prediction variance；
- pairwise cosine；
- effective rank；
- nearest-token retrieval accuracy；
- mean prediction norm；
- constant-output baseline。

## 18. 实验B：训练组

最小可解释设计：

| 组 | 训练目标 | 用途 |
|---|---|---|
| B1 | next-block embedding | 主假设 |
| B2 | shifted next-embedding | 难度/任务对齐控制 |
| B3 | standard next-token CE | 标准语言模型基线 |
| B4 | constant/mean predictor | collapse baseline |

资源不足时只训练B1和B3。

## 19. 实验B：训练参数初稿

```yaml
seed: 1234
dtype: bfloat16_or_float32
optimizer: AdamW
learning_rate: 3.0e-4
weight_decay: 0.01
warmup_steps: 500
max_steps: 20000
batch_tokens_target: 32768
gradient_clip: 1.0
checkpoint_steps: [0, 100, 500, 1000, 2000, 5000, 10000, 20000]
validation_every: 500
```

先用2k-step pilot 检查 loss、variance 和 retrieval；通过后才训练20k。

## 20. 实验B：checkpoint 动力学审计

对每个训练 checkpoint：

1. 在固定8/32个验证 block 上迭代同维 operator；
2. paper-exact baseline：500步，最终状态 Jacobian/nearby；
3. windowed protocol：burn512+eval128；
4. normalized Frobenius；
5. Benettin Lyapunov；
6. nearby separation；
7. Poincaré/return map；
8. collapse/divergence；
9. 验证 loss、cosine、retrieval accuracy；
10. 性能–$|\lambda|$、性能–$|\log\rho|$ 关系。

## 21. 实验B：预注册判断

### 支持

- 验证性能随训练提升；
- 最优 checkpoint 附近 $\rho\approx1$ 且 $\lambda\approx0$；
- 早晚 checkpoint 跨越稳定/周期/混沌相位；
- Poincaré 与数值指标一致；
- 非 identity/shift baseline；
- 多 seed 可复现。

### 不支持

- 最优 checkpoint 的 $\rho$、$\lambda$ 明显远离边缘；
- norm≈1来自 collapse、identity 或固定 target scaling；
- 不同 seed 方向相反；
- 只有训练 loss 与 norm 同时下降但无相变；
- full embedding 任务无法学到超过 mean predictor 的结果。

## 22. GPU与执行顺序

### 阶段A0：实现与CPU单测

- 不占GPU；
- 完成 rolling operator、shift decomposition 和小张量解析测试。

### 阶段A1：GPU pilot

- GPU5：step0 soft/hard；
- GPU6：step143000 soft/hard；
- GPU7：shift-only与position control或空闲备用。

### 阶段A2：四 checkpoint main

- GPU5/6/7 并行前三个；
- 最后一个接续空闲GPU；
- T=1完成审计后才开启温度扩展。

### 阶段B0：训练可行性 pilot

- GPU5：B1 next-block；
- GPU6：B3 CE baseline；
- GPU7：验证/Jacobian审计或B2控制。

阶段B必须等阶段A报告完成后再决定是否执行。

## 23. 报告与留痕

### 实验A

```text
plan/transformer_paper_validation_small_experiments_plan.md
reports/rolling_next_token_criticality_report.md
```

报告必须单列：

- total Frobenius 中 shift identity 占比；
- innovation Jacobian；
- hard/soft差异；
- reset/absolute position差异；
- 每 sample/anchor Poincaré；
- paper-exact vs windowed协议。

### 实验B

```text
plan/full_embedding_predictor_training_plan.md（启动前细化）
reports/full_embedding_predictor_training_report.md
```

每次训练记录：

- git commit/diff；
- config snapshot；
- seed/GPU/环境；
- loss curve；
- collapse diagnostics；
- checkpoint manifest；
- 动力学审计结果；
- 能证明/不能证明的假设。

## 24. 验收清单

### 实验A启动前

- [ ] shift-only解析值单测通过；
- [ ] soft/hard最后位置与手算一致；
- [ ] hard路径明确禁止JVP；
- [ ] 32个anchor跨checkpoint一致；
- [ ] reset-position主协议固定；
- [ ] source文档和token IDs manifest写入 `/home`；
- [ ] 不下载新权重；
- [ ] pilot成本估算完成。

### 实验B启动前

- [ ] 明确不是 identity reconstruction；
- [ ] target codebook固定/stop-gradient；
- [ ] mean predictor baseline完成；
- [ ] collapse监控可用；
- [ ] CE同架构对照配置完成；
- [ ] 2k-step pilot先行；
- [ ] 数据与checkpoint全部写 `/home`。

## 25. 推荐决策

先执行实验A，但把“滑动窗口能还原论文结论”视为待检验假设，而不是预设结论。由于 shift identity 会让 total normalized Frobenius 天然接近1，实验A若不做 shift decomposition，结果几乎必然产生假阳性。

实验A最有价值的输出应是：

$$
\text{训练性能}
\leftrightarrow
\lambda_{\max}
\leftrightarrow
\text{innovation Jacobian}
\leftrightarrow
\text{hard token behavior}.
$$

实验B在实验A完成后再启动。full-embedding predictor 确实更接近论文的同维完整算子，但训练目标不得包含 identity/shift 的平凡答案，否则 normalized Jacobian≈1将是目标设计的产物，而不是训练自发达到的临界性。
