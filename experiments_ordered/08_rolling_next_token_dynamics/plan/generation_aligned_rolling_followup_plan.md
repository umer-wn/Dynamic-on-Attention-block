# Rolling 工程有效性与切向复测计划（精简版）

状态：计划中，尚未执行  
日期：2026-07-14  
方法与工程审计：`reports/rolling_application_scenario_derivation.md`

## 1. 先更正问题定义

现有 rolling main 的实际算子是：固定64-token窗口、每步全窗口重算、位置重置为 `0..63`、`use_cache=False`，soft 路径把下一 token 分布变成词嵌入期望。它是合法的固定维自治映射，但不是 Pythia 原生生成或原生滑窗注意力。

本轮不再同时铺开 native-position cocycle、可微 KV cache、多个窗口长度和全 checkpoint 大网格；先回答一个更基础的问题：**现有算子能否代表合理的有限记忆生成场景，还是其周期/相位主要由截断、位置重置和 sink 丢失造成？**

## 2. 三个配对基线

### N：Pythia 原生增长前缀

- hard token；
- 原生 RoPE position 与 `DynamicCache`；
- 前缀随生成增长，不截断；
- 总长度严格不超过 `max_position_embeddings=2048`。

这是 Pythia 在上下文上限内的正常工程基线，不是固定维自治系统，不计算论文式固定状态 Jacobian。

### R：现有 recency-only rolling

复用现有定义：

\[
R(X_t)=[x_{t,2},\ldots,x_{t,L},g_\theta(X_t)],\qquad L=64.
\]

- 只保留最近64个 token/state；
- 每步全窗口重算；
- position 重置；
- 不使用 KV cache，不保留固定 sink token。

已有动力学数据不重跑。新增的性能/生成对照明确标记为 `recency-only full-recompute baseline`。

### S：sink-preserving fixed-memory rolling

保留最初 `s=4` 个真实 prompt token 作为固定条件，同时滚动最近 `L-s=60` 个 token：

\[
S(C,R_t)=\bigl(C,\operatorname{shift}(R_t),g_\theta(C,R_t)\bigr),
\qquad C=[x_1,\ldots,x_s].
\]

第一轮使用透明的**全窗口重算**实现，position 在当前 cache/window 内连续编号；不下载外部 SinkCache，不先实现复杂 KV 重旋转。它不是高效 StreamingLLM 实现，但可隔离“保留初始 attention sinks”是否改变行为。

## 3. Stage 0：先做工程行为门控

### 3.1 最小规模

- checkpoints：`step0`、`step143000`；若二者有定性差异，再加入 `step1000`；
- 数据：同一批4个长 validation documents、每文档1个固定 anchor；
- `L=64`、`s=4`；
- teacher-forced targets：每文档连续512 tokens；
- hard greedy generation：每 anchor 512 tokens；
- N 路径确保 prompt+generation 不超过2048；
- GPU：优先5/6/7；不下载新权重。

### 3.2 指标

对 N/R/S 使用相同目标 token，报告：

- token-weighted cross-entropy 与 perplexity；
- top-1 accuracy、entropy；
- greedy repetition-1/2/3/4、unique-token ratio、exact window cycle；
- N/R/S 生成 token 的首次分歧位置及累计一致率；
- 在预注册的稀疏步读取 attention，报告前4个槽位与最近槽位的 attention mass。

attention 只用于诊断“是否存在 sink-like concentration”，不直接替代 Jacobian 或 Lyapunov。

### 3.3 门控判定

- 若 R 与 N/S 的 PPL、重复和周期行为接近，现有 rolling 的工程近似得到支持；
- 若 R 明显退化或进入周期，而 S/N 不出现，则旧结果的“应用生成”解释被证伪，旧动力学只保留为 R 算子的性质；
- 若 S 仍与 N 显著不一致，说明64-token固定记忆本身不足，停止把 rolling 外推为 Pythia 正常使用；
- 只有通过该门控，才进入 Stage 1。

## 4. Stage 1：通过门控后再计算切向量

仅比较 R 与 S 的 fixed-dimensional soft map；不在本轮实现增长 KV-cache 的可微 Jacobian。

### 4.1 状态和 Lyapunov

R 的动态状态是全部 `LH` 个 embedding features。S 的初始 sink `C` 是固定条件，不参与扰动；动态状态仅为最近 token：

\[
R_t\in\mathbb R^{(L-s)\times H},\qquad
J_S\in\mathbb R^{(L-s)H\times(L-s)H}.
\]

这样不会把固定 sink 坐标误当成 identity/neutral directions。稳定性主指标仍是 Benettin 最大 Lyapunov：

\[
\widehat\lambda_{\max}
=\frac1T\sum_{t=1}^{T}\log\|J_t v_t\|_2,
\qquad v_{t+1}=\frac{J_t v_t}{\|J_t v_t\|_2}.
\]

### 4.2 new-token 与 token-block Jacobian

\[
J_{\rm new}=\frac{\partial e_{t+1}}{\partial\operatorname{vec}(X_t)}
=[J_1,\ldots,J_L],\qquad J_\ell\in\mathbb R^{H\times H}.
\]

所有 blocks 都可报告为条件敏感性；但 S 的 sink blocks 不计入动态 Lyapunov 状态。验证：

\[
\|J_{\rm new}\|_F^2\approx\sum_{\ell=1}^{L}\|J_\ell\|_F^2.
\]

pilot 只取两个 checkpoint、4 documents、每轨迹4个均匀状态、每 block 8 probes。只有符号/排序不稳定时才增加 probe 或 checkpoint。

## 5. 本轮明确删除或后置的内容

- 删除原计划中的“四臂同时执行”；
- native/absolute-position soft cocycle 后置，因为它不能回答 sink/cache 是否合理；
- `L∈{32,64,128}` 扫描后置到 S 与 N 已基本一致之后；
- differentiable KV-cache JVP 后置，不作为工程门控；
- full-embedding predictor 重新训练继续单独立项；
- full-state normalized Frobenius 只做 shift 分解审计，禁止以 `≈1` 标记临界。

## 6. 实现与验收

启动 GPU 前必须通过：

1. N 路径逐 token logits 与无 cache 全前缀重算一致；
2. R 与旧 hard operator 在相同输入上逐步一致；
3. S 始终保留完全相同的前4个 token，滚动区长度始终为60；
4. 三路径 target 对齐，无 next-token off-by-one；
5. attention mass 的 key 位置与 mask 可人工复核；
6. token-block 重构误差不超过10%；
7. manifest 固定 checkpoint、document、anchor、token 数、position 规则和随机种子；
8. 原始数据写入 `/home/luohaoming`，旧 rolling 数据只读。

## 7. 输出与留痕

```text
/home/luohaoming/model_feature_experiments/rolling_engineering_validity/
  configs_snapshot/
  behavior_gate/
  tangent_pilot/
  processed/
  figures/
  logs/
  manifests/
```

仓库交付：

```text
plan/generation_aligned_rolling_followup_plan.md
reports/rolling_application_scenario_derivation.md
reports/rolling_engineering_validity_report.md
reports/assets/rolling_engineering_validity/
```

执行顺序固定为：行为门控 → 读取报告 → 决定是否做切向 pilot。未通过门控时，不自动扩大实验。
