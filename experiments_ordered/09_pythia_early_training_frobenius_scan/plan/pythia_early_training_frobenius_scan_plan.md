# Pythia 前 100 个 checkpoint：test loss、single-token normalized Frobenius 与训练程度实验计划

状态：P0 分段 profiling 已完成；正式粗扫描暂停，先执行 P1/P2 小规模门控  
日期：2026-07-14  
模型：`EleutherAI/pythia-70m`  
数据根：`/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/`

## 1. 核心目标

本实验的首要目标不是寻找“Frobenius 恰好等于 1”的 checkpoint，而是检验：

> 在模型架构、token 集、single-token feedback 算子、轨迹长度和 Jacobian 定义全部固定后，token-level normalized Frobenius 是否与 attention 模型的训练程度存在真实、可复现且不由少量 token 驱动的关系。

训练程度使用两个互补变量表示：

1. `training_step`：优化过程中的 checkpoint step；
2. 固定测试集上的 token-weighted causal cross-entropy：模型实际 next-token 能力的代理量。

其中 test loss 是主要解释变量，training step 是次要解释变量。原因是 step 只表示训练进度，不能保证能力单调改善；若 loss 后期反升，step 与能力会分离。

## 2. 主要假设

### H1：训练相关性

checkpoint 的 test loss 变化与 16 个固定 token 的末端 normalized Frobenius 分布变化有关。主要检验量为 checkpoint 内 token 中位数与 test loss 的关系，同时展示全部 token 点。

### H2：不只是 step 的伪相关

若 loss 先下降后上升，则 normalized Frobenius 应优先随 loss/能力变化，而不是仅随 step 单调变化。反升区间是区分这两种解释的关键自然对照。

### H3：关系跨 token 可重复

关系不能只由某一个词频层或少量 token 驱动；同一 token 跨 checkpoint 的配对变化方向应具有一致性，并报告频率层分组结果。

### H4：几何变化与局部线性变化可区分

2D 投影轨迹用于观察吸引子几何是否随训练变化；它不能替代 Jacobian，也不能单独证明临界、混沌或稳定性。

## 3. 固定定义

### 3.1 Single-token dynamics

只使用已有 G1 `isolated_token` 算子。对 token id `w`，当前 checkpoint 的输入 embedding 为：

$$
x_0=E_\theta[w]\in\mathbb{R}^{512}.
$$

每个 dynamics step 固定模型权重，执行：

$$
x_{t+1}=F_\theta(x_t)
=M_\theta(\text{inputs\_embeds}=x_t[None,None,:])_{0,0,:}.
$$

输入和输出均为一个 token 的 512 维向量。序列长度为 1，`attention_mask=[1]`、`position_id=[0]`、无 padding、无 KV cache。此循环不是训练、不是文本自回归生成，也不经过 LM head/softmax；它是在冻结 checkpoint 上反复回灌 final hidden state。

### 3.2 Token-level Jacobian 与 normalized Frobenius

在末端状态 `t=767` 计算精确方阵：

$$
J_{767}=\frac{\partial F_\theta(x_{767})}{\partial x_{767}}
\in\mathbb{R}^{512\times512}.
$$

主要指标为：

$$
\rho=\frac{\lVert J_{767}\rVert_F}{\sqrt{512}}
=\sqrt{\frac{1}{512}\sum_{i,j}J_{ij}^2}.
$$

`rho=1` 表示奇异值平方的均值为 1，不等价于最大奇异值为 1，也不等价于最大 Lyapunov 为 0。

P0/P2 反思后，将精确 Jacobian 固定为两个预注册状态：`t=0` 与 `t=767`。二者不被当作两个独立样本，而是同一 token 的配对条件：

- `rho_self_t0 = ||J_theta(E_theta[w])||F/sqrt(512)`：当前 checkpoint 自身 token embedding 上的局部增益；
- `rho_tail_t767 = ||J_theta(x_767)||F/sqrt(512)`：该 checkpoint 的 768-step dynamics 到达的末端状态附近增益。

P2 发现 step16000 的 16 个 `rho_tail_t767` 标准差仅 `2.28e-7`，提示不同 token 很可能进入同一末端吸引子。因此 tail 指标同时受权重和吸引子选择影响，不能单独当作“训练程度的 Jacobian”。为进一步隔离权重变化，增加固定 state-bank 对照：

$$
rho^{common}_{c,w}
=\frac{||J_{theta_c}(E_{step1000}[w])||_F}{\sqrt{512}}.
$$

即所有 checkpoint 都在同一组 step1000 token vectors 上求导。它是数学上的 common-coordinate control；由于网络内部表示基底也会随训练漂移，它仍不是因果识别，但能区分“输入状态变了”与“同一坐标点上的算子权重变了”。

### 3.3 Test loss

数据为本地缓存的 WikiText-2 `test`，固定最先出现的 128 个非空文本样本，最大长度 64。对 padding 位置设 `label=-100`，只累计有效的 next-token targets：

$$
L_c=\frac{\sum_s \mathrm{NLL}_s}{\sum_s n_s},
\qquad
\mathrm{PPL}_c=\exp(L_c).
$$

每个样本保存 `nll_sum` 和 `predicted_token_count`，使相邻 checkpoint 能在完全相同的 128 个样本上做 paired bootstrap。报告必须称为“固定 128 样本 test-loss estimate”，不能冒充完整 benchmark。

## 4. 固定样本与轨迹协议

- checkpoint 之间固定同一组 16 个 token：4 个经验词频层各 4 个；
- token manifest：`/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/frequency_audit/selected_tokens.jsonl`；
- dynamics steps：768；
- 精确 Jacobian：只在 state `t=767`；
- dtype：float32；autocast 关闭；模型 `eval()`，全部参数冻结；
- 投影：固定随机单位向量，seed `1234`，跨 token/checkpoint 使用相同生成规则；
- 2D 图：`z0=<q0,x_t>` 与 `z1=<q1,x_t>`，颜色编码 dynamics step；
- 不计算本轮非必要的 Hutchinson、Lyapunov、nearby distance 或旧 product metric；
- 权重缓存和所有原始数据保留在 `/home/luohaoming`。

## 5. Checkpoint 抽样与自适应补点

### 5.1 粗扫描

Pythia 常规 checkpoint 的前 100 个定义为 `step1000...step100000`。粗扫描每隔 4 个 checkpoint 取一次：

```text
step1000, step5000, ..., step97000
```

共 25 个 checkpoint，均运行 test loss、16-token trajectory、`t=767` 精确 Jacobian 和 2D 投影。另对 `step0` 和边界 `step100000` 运行 loss-only sentinel；它们不改变“25 个粗扫描 dynamics checkpoint”的定义。

### 5.2 “插值”的严格含义

本计划中的插值不是对 loss 曲线做数值插值，而是加载缺失的真实 1000-step Hugging Face revision 并重新评估，例如在 `step21000` 附近实际补跑 `step18000/19000/20000/22000/23000/24000`。

### 5.3 显著下降和显著反升

对相邻 checkpoint `a,b`：

$$
\Delta L_{a\to b}=L_b-L_a.
$$

对 128 个固定样本做 10,000 次 paired bootstrap（seed `1234`）；每次重采样后仍按 token 数加权聚合 NLL。预注册判定：

- 显著下降：95% CI 上界 `<0`，且 `Delta L <= -log(1.01)`；
- 显著反升：95% CI 下界 `>0`，且 `Delta L >= log(1.01)`。

`log(1.01)` 对应至少约 1% 的 perplexity 相对变化，用于避免把极小的固定测试集波动叫作反转。此阈值是本实验的操作性阈值，不来自原论文，也不得看完结果后修改。

### 5.4 自适应搜索流程

1. 粗 loss 曲线完成后，寻找全局最低点及所有“入边下降、出边上升”的粗候选；
2. 对主候选左右各一个 4000-step 粗区间，补齐其中所有真实 1000-step checkpoint；
3. 在密集序列中寻找“前一段显著下降、后一段显著反升”的 checkpoint；
4. 若未找到，围绕当前最低点逐个向外扩展相邻粗区间，并继续补齐真实 checkpoint；
5. 最多评估 `step1000...step100000` 全部 100 个 loss；若仍无统计显著反升，结论必须写为“前 100 个 checkpoint 内未发现显著反升”，不得强行指定转折点；
6. 对最终确认的主反转带，补跑其中所有 1000-step checkpoint 的完整 16-token dynamics/Jacobian，并在左右各保留一个 checkpoint 作为 guard；
7. 若只有描述性反升而 bootstrap 未通过，允许绘图和补跑，但标签必须是 `descriptive/unconfirmed`，不进入显著性结论。

主反转定义为所有有效局部反转中 test loss 最低者；其他反转保留为次要候选，避免事后只挑最符合预期的一处。

## 6. “真实关系”的分析标准

不会仅凭一条 checkpoint 均值曲线下结论。至少报告：

1. 每个 checkpoint 的 16 个 token 点、中位数和 IQR；
2. 同一 token 相邻 checkpoint 的配对差；
3. Spearman `rho(Frobenius, test loss)` 与 `rho(Frobenius, training step)`；
4. checkpoint-level 相关性的 bootstrap CI，但明确 checkpoint 数量有限且自相关；
5. token fixed-effect / within-token centered 分析，用来排除 token 固有尺度差异；
6. 四个频率层分层结果和 leave-one-token-out 敏感性；
7. 粗扫描与自适应补点使用不同标记，防止把事后密集抽样误认为预先均匀抽样；
8. 在 loss 反升带检查 Frobenius 是否随 loss 同向回转。只有这项成立，才支持“更接近能力而非仅随 step”的解释。

支持“存在关系”要求：效应方向在大多数 token 中一致、相关 CI 不跨 0、leave-one-token-out 不由单个 token 决定，并在反转带具有相符方向。否则状态为未决或不支持。

即使满足上述条件，实验仍只能证明 checkpoint 间的统计关联，不能证明训练因果，也不能推广为所有 Transformer 或完整序列 Jacobian 的规律。

## 7. 图表

1. 粗扫描 test loss/PPL：真实 checkpoint 点，step0 与 step100000 sentinel 单独标记；
2. 反转带 loss 局部放大图：paired bootstrap CI，标注显著下降/反升；
3. normalized Frobenius 随 step：全部 16 token、checkpoint 中位数/IQR、参考线 1；
4. token × checkpoint Frobenius 热图；
5. Frobenius 对 test loss 散点：粗扫描和补点不同形状，按 step 着色；
6. 对齐面板：反转带 loss 与 Frobenius 分上下两个共享 x 轴的图，禁止双 y 轴；
7. 四个代表 token 的 checkpoint 2D 轨迹小多图，所有子图固定投影和轴范围；
8. 反转带所有 checkpoint 的 2D 轨迹图，逐图注明 token、checkpoint、`t=0/767`。

2D 投影只描述几何压缩、漂移或周期样式；它不参与 Frobenius—训练程度相关性的显著性判定。

## 8. 输出与留痕

```text
/home/luohaoming/model_feature_experiments/pythia_early_single_token_scan/
  raw/<checkpoint>/
  jacobians/<checkpoint>/
  processed/
  figures/
  manifests/
  logs/
  status/
```

仓库新增：

```text
plan/pythia_early_training_frobenius_scan_plan.md
scripts/compute_pythia_early_single_token_scan.py
scripts/analyze_pythia_early_single_token_scan.py
scripts/plot_pythia_early_single_token_scan.py
scripts/launch_pythia_early_single_token_scan.sh
tests/test_pythia_early_single_token_scan.py
reports/pythia_early_training_frobenius_scan_report.md
```

每个 checkpoint 写 `run_complete.json`；失败写独立日志且可幂等重跑。manifest 记录 revision、commit hash（若可得）、GPU、软件版本、token manifest hash、dataset fingerprint、projection seed 和抽样阶段 `coarse/adaptive/sentinel`。

## 9. 资源与执行顺序

### 9.1 P0：分段 profiling（已完成）

`step1000 × token35408 × 768 steps` 在 GPU 5 的结果：

| 阶段 | 在线路径/s | 离线路径/s | 结论 |
|---|---:|---:|---|
| model + tokenizer load | 596.212 | 2.823 | 主要瓶颈是 Hugging Face HEAD 超时重试，不是权重反序列化 |
| dataset first-128 load | 83.165 | 63.774 | `load_dataset` 仍做 Hub metadata 检查，不能每 checkpoint 调用 |
| 128 样本逐条 loss | 1.431 | 1.473 | 不是瓶颈 |
| 128 样本 batch16 loss | 0.264 | 0.246 | 与逐条聚合 loss 绝对差 `7.79e-8` |
| 当前双轨迹（含未使用 nearby） | 13.293 | 13.350 | 新实验不需要 nearby，属于重复计算 |
| 只保留投影的单轨迹 | 6.844 | 7.207 | 末端 state 与双轨迹主支最大差为 0 |
| exact Jacobian chunk16 | 0.507 | 0.488 | 精确 Jacobian 本身不是瓶颈 |
| exact Jacobian chunk128 | 0.078 | 0.076 | 与 chunk16 allclose；最大元素差 `1.19e-7`，normalized Frobenius 完全相同 |

观测 `rho=0.6867641833`、loss `6.1510930920`。两次 profiling 的 dataset text SHA256 均为 `3365c25e...64bbfbbf`。因此先前 10 分钟 smoke 不能解释为 Jacobian 太贵：约 96% 时间来自在线 model/dataset metadata 重试。

P0 后锁定的等价优化：

- test loss 改为 batch16，但仍保存 128 行逐样本 NLL/count；
- trajectory 去掉未使用的 nearby 分支，只计算同一条主轨迹；
- exact Jacobian 使用 chunk128，并保留 `[512,512]` matrix；正式扫描固定计算 `self_t0`、`tail_t767`，并在 step1000 common state-bank 上计算权重对照；
- normalized Frobenius 只需矩阵平方和，不为本研究目标额外执行完整 SVD；
- 这些变化均有数值 parity 证据，不改变目标指标或算子。

### 9.2 P1：I/O 准备门控

1. 从已缓存 WikiText Arrow 数据只读取一次，将固定 128 个文本和 sample id 固化到 `/home/luohaoming/.../manifests/wikitext_test_first128.jsonl`；此后 checkpoint worker 禁止调用 `load_dataset`；
2. checkpoint 权重下载与 GPU 计算分离。先预取真实 revision，写入 snapshot hash，随后 worker 强制 `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`local_files_only=True`；
3. 先测试一个尚未缓存的 `step5000`。若网络无法预取，状态写为 `weight_prefetch_blocked`，不让每个 GPU worker 重复等待 Hub timeout；
4. 只有 test manifest hash、模型 snapshot 和 tokenizer 文件均存在，checkpoint 才能进入 GPU 队列。

2026-07-15 更新：服务器非交互环境中官方 `huggingface.co:443` 仍不可达，DNS 将 `huggingface.co` 解析到 `157.240.2.36` 且 HTTPS 超时；`hf-mirror.com` TLS 可用。因此本轮正式预取使用：

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=/home/luohaoming/model_feature_cache/hf_cache
HF_HUB_DISABLE_XET=1
```

实际执行环境核查为 `/data1/luohaoming/langurage_feature/venv/bin/python`，该环境包含 `torch`、`transformers`、`huggingface_hub`、`datasets`、`numpy`，并与已完成 pilot 日志中的 `torch_version=2.8.0+cu128`、`python_version=3.9.19` 一致。系统 `/usr/bin/python` 不含实验依赖，不用于本实验。

预取脚本支持两层并发并在每个 revision 的 status JSON 中记录参数：

- `--max-workers`：单个 checkpoint 内的文件级并发，默认 `8`；
- `--revision-workers`：多个 checkpoint revision 并发下载，默认 `2`。

已验证 `step5000` 与 `step9000` 通过镜像下载成功，随后完成 full dynamics、`tail_t767` Jacobian、`self_t0/common_step1000_state` controls、分析与图表更新。后续 coarse scan 先运行后台多线程预取；正式 GPU 批量脚本等待该预取结束后再进入计算，避免多个下载进程争用同一 HF cache。

2026-07-15 15:10 再更新：粗扫预下载在 `--max-workers=8`、`--revision-workers=2` 下连续出现 `ChunkedEncodingError / IncompleteRead`，失败 revision 包括 `step17000/step21000/step25000/step29000`。这些错误均发生在大文件下载中途，说明主要问题是镜像长连接不稳定和并发过高，而不是认证、revision 缺失或模型文件不兼容。已修复 `scripts/prefetch_pythia_checkpoints.py`：

- 每个 revision 增加最多 4 次 retry；
- 首次失败后逐步降低文件级并发，最终使用 `max_workers=1`；
- 默认预取改为 `PREFETCH_MAX_WORKERS=2`、`PREFETCH_REVISION_WORKERS=1`；
- status JSON 记录 `attempts_requested` 和逐次 attempt 的 worker、状态、错误与耗时；
- 依赖 HuggingFace cache 的 `.incomplete`/已完成 blob 续传机制，不删除旧缓存。

旧高并发预下载和等待 wrapper 已停止并重启为稳态 batch。重启后先前失败的 `step17000/step21000/step25000/step29000` 被 cache hit/续传识别为 complete，当前预取继续从后续 checkpoint 向前推进。

2026-07-15 20:36 下载目标临时调整：当前优先目标不是立即完成 GPU 扫描，而是为次日测试补齐 checkpoint 权重缓存。审计 HF cache 的 `refs/snapshots` 后发现，旧 status 中的 complete 不一定代表可被模型加载；若 snapshot 只有 tokenizer/config 而缺少 `model.safetensors`，仍视为未完整。为减少链路压力，新增 `scripts/prefetch_pythia_early_minimal_weights.sh`：

- 直接按 cache 实物判断是否缺 `config.json` 或 `model.safetensors`；
- 只下载运行所需最小文件：`.gitattributes`、`config.json`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`model.safetensors`；
- 不再下载冗余的 `pytorch_model.bin`；
- 串行 checkpoint、单文件 worker、最多 12 次 retry，并设置较长下载 timeout；
- 输出缺失列表到 `status/minimal_weight_revisions.txt`，仍使用 `status/prefetch/*.json` 记录结果。

这一步的验收标准改为：目标 coarse/sentinel checkpoint 的 snapshot 中均存在 `config.json` 与 `model.safetensors`，可供次日 `local_files_only=True` 加载；GPU full scan 可在权重缓存稳定后再启动。

### 9.3 P2：3-checkpoint 科学 pilot

先只运行 `step1000/step5000/step9000`：分别优先 GPU 5/6/7，每个 checkpoint 均使用完整 16-token cohort、768-step 单轨迹、`t=767` exact `[512,512]` Jacobian、固定 2D 投影和 128-sample loss。

2026-07-14 执行时，官方 Hub 的 `step5000` 预取在 130.39 秒后因 `huggingface.co:443` connection timeout 失败；`step1000` 本地 cache hit 仅需 0.001 秒。为继续验证方法而不把 GPU 耗在网络重试上，先执行缓存替代门控 `step0/step1000/step16000`。该替代 pilot 只验证代码、配对、开销和是否存在值得继续扫描的初步变化，不替代原定早期等间隔 pilot，也不进入 25 点粗扫描的预注册样本定义。网络恢复后仍须补回 `step5000/step9000`。

门控：

- 3 个 checkpoint 的 dataset hash、sample id、token id、projection seed 完全一致；
- 每 checkpoint 恰好 16 个 Jacobian，shape 均为 `[512,512]`；
- 每 token 恰好 769 个投影状态（含 `t=0`）；
- batch loss 与已记录 step1000 顺序版 loss 误差不超过 `1e-6`；
- 实测 wall time 与 P0 推算同量级，且 GPU 峰值显存安全；
- 初步图必须展示 16 个 token 离散度，不能只画 3 个均值。

P2 report 完成并确认没有 operator/Jacobian/sample 对齐错误后，才进入原 25-checkpoint P3 粗扫描。不能为了追赶进度绕过此门控。

### 9.4 P3/P4：粗扫描和自适应反转搜索

P2 通过后，GPU 5/6/7 各运行一条 checkpoint 队列，每卡同时只加载一个 checkpoint。每批 revision 先预取、后离线计算。25 点粗扫描完成后才运行 paired-bootstrap 自适应选择；loss-only 补点仍先预取真实 revision，确认反转带后再补完整 dynamics。

正式批量脚本 `scripts/launch_pythia_early_single_token_scan.sh` 的 coarse full 阶段会在 GPU5/6/7 上分发 25 个 checkpoint，并在 full dynamics 后自动补跑 `scripts/compute_pythia_single_token_jacobian_controls.py`，保证 `self_t0`、`tail_t767`、`common_step1000_state` 三类 Jacobian 条件都进入 processed CSV 和最终报告。sentinel/adaptive 阶段沿用同一预取、分析、绘图与留痕逻辑。

最终验证 checkpoint 数、128 个 sample id、16 个 token id、768 个 trajectory transitions、Jacobian shape 和 projection seed 完全一致；检查无本项目残留 GPU 进程后写 report。

## 10. 停止条件

- P0：1-token 在线/离线分段 profiling 完成（已达到）；
- P1：固定 test manifest 和一个新 revision 的权重预取门控通过；
- P2：3-checkpoint × 16-token 小规模科学 pilot 及中间报告完成；
- P3：25 个粗 dynamics checkpoint 与 27 个粗/sentinel loss 点完成；
- P4：找到显著反转带，或穷尽前 100 个 loss 后证明未找到；
- P5：反转带完整 dynamics/Jacobian、图表、processed CSV 和报告完成；
- 若下载、数据 fingerprint、Jacobian shape、token 对齐或 GPU 数值出现异常，停止对应队列并保留现场，不以缺失点静默作图。
