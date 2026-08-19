# Rolling next-token token-block Jacobian follow-up 计划

状态：计划中，等待 rolling 工程行为门控  
日期：2026-07-13  
触发原因：现有 rolling 主实验只保存 full innovation Frobenius 汇总，不能从已有 JSONL 反推出单个输入 token 的 Jacobian 贡献。

方法定位更新（2026-07-14）：本计划的 `H × H` token blocks 用于解释 `H × (L·H)` new-token innovation Jacobian 的上下文位置贡献；它们不替代 `LH × LH` rolling full-state Jacobian，也不单独定义最大 Lyapunov。该 pilot 不再作为下一轮第一项 GPU 工作：先按 `plan/generation_aligned_rolling_followup_plan.md` 完成 native / recency-only / sink-preserving 三路径的工程行为门控，通过后才执行本计划。

## 1. 问题定义

rolling soft operator 的新 token 输出为：

\[
e_{\mathrm{new}}=g_\theta(X),
\qquad
X=[x_1,\ldots,x_L]\in\mathbb R^{L\times H}.
\]

当前已经测量：

\[
J_{\mathrm{new}}
=\frac{\partial e_{\mathrm{new}}}{\partial\operatorname{vec}(X)}
\in\mathbb R^{H\times LH}.
\]

下一组实验将其按输入 token 分块：

\[
J_{\mathrm{new}}
=[J_1,J_2,\ldots,J_L],
\qquad
J_\ell=\frac{\partial e_{\mathrm{new}}}{\partial x_\ell}
\in\mathbb R^{H\times H}.
\]

特别报告最后 token：

\[
J_{\mathrm{last}}=J_L
=\frac{\partial e_{\mathrm{new}}}{\partial x_L}.
\]

## 2. 为什么不能用现有数据后处理

现有 trajectory JSONL 只保存：

- full/newest 三个随机投影标量；
- state norm、step delta、nearby distance、entropy/top1；
- summary 中的 full innovation Frobenius 汇总。

没有保存完整连续状态、JVP 输出向量或每个 token block 的随机探针结果。现有指标只约束：

\[
\|J_{\mathrm{new}}\|_F^2
=\sum_{\ell=1}^{L}\|J_\ell\|_F^2,
\]

一个总和不能唯一确定 64 个位置贡献。`projection_newest_*` 是状态投影，不是导数，不能代替 `J_last`。

实验可复用现有 checkpoint、token manifest、anchor 和确定性轨迹配置，但必须重新加载模型、重放轨迹并执行新的 JVP。

## 3. 估计方法

对位置 \(\ell\) 生成只在该 token 非零的 Rademacher probe：

\[
v_{\ell,k}=[0,\ldots,w_k,\ldots,0],
\qquad
w_k\in\{-1,+1\}^{H}.
\]

计算：

\[
J_{\mathrm{new}}v_{\ell,k}=J_\ell w_k.
\]

利用 Hutchinson 恒等式：

\[
\|J_\ell\|_F^2
\approx
\frac1K\sum_{k=1}^K\|J_\ell w_k\|_2^2.
\]

定义 token-output normalization：

\[
\rho_\ell=\frac{\|J_\ell\|_F}{\sqrt H}.
\]

则应满足数值一致性：

\[
\rho_{\mathrm{innovation,output}}^2
\approx
\sum_{\ell=1}^{L}\rho_\ell^2,
\]

\[
\rho_{\mathrm{innovation,total}}^2
\approx
\frac1L\sum_{\ell=1}^{L}\rho_\ell^2,
\]

\[
\rho_{\mathrm{rolling,total}}^2
\approx
\frac{L-1}{L}
+\frac1L\sum_{\ell=1}^{L}\rho_\ell^2.
\]

## 4. Pilot

- checkpoints：`step0`、`step143000`；
- anchors：沿用 `doc264@0`、`doc264@256`、`doc219@0`、`doc219@256`；
- window：64，hidden：512；
- operator：soft expected next token，T=1，reset positions；
- burn-in：256；eval：128；
- 状态采样：在 eval 窗口均匀选 4 个状态，不再只取连续最后 4 个；
- 每个 token block：8 个 Rademacher probes；
- dtype：float32；
- GPU：优先 GPU5/6 并行；
- 数据根：`/home/luohaoming/model_feature_experiments/rolling_next_token_criticality/token_block_jacobian_pilot`。

Pilot 门控：

1. block-sum 与已有 full innovation Frobenius 的相对误差不超过预注册容差 10%；
2. probe 重复的 token profile 排序基本稳定；
3. 最后 token 贡献与全部历史 token 贡献能被明确区分；
4. 运行成本允许扩展到四 checkpoint。

## 5. Main（pilot 通过后）

- 四 checkpoints：`step0`、`step1000`、`step16000`、`step143000`；
- 8 个匹配文档 anchor0；
- 每 anchor 均匀 4 states；
- 每 block 8 probes；
- 输出每个状态、每个 token position 的 \(\rho_\ell\) 和 probe dispersion；
- 同时聚合相对 lag：`lag=0` 表示最后输入 token，`lag=63` 表示最早 token。

核心图：

1. checkpoint × token lag 的 \(\rho_\ell\) 热图；
2. 最后 token 与历史 token 总贡献比较；
3. block-sum 重构值与 full innovation 实测值一致性；
4. token-block profile 与 Lyapunov 的样本级散点；
5. 各 checkpoint 的 attention-distance/Jacobian-distance 对照（仅作为相关性，不宣称因果）。

## 6. 能回答与不能回答

能回答：

- 新 embedding 对最后 token 和历史 token 各有多大局部敏感性；
- full innovation Frobenius 是由少数位置还是全窗口共同贡献；
- 训练是否改变 Jacobian 的时间/位置分布。

不能单独回答：

- token block Frobenius 是否等于最大奇异值；
- 哪个具体 attention head 导致该贡献；
- token-block profile 是否足以判断混沌；
- total rolling Frobenius 接近 1 是否代表临界。

## 7. 留痕

启动前补充 YAML 配置和单元测试；每次执行写独立 log、manifest、plan/report。原始 rolling 结果保持只读，不覆盖现有 summary 或图。
