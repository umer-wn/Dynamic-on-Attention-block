# 单 Token 词频分层动力学 Pilot 报告

状态：四 checkpoint Pilot 扩展完成并通过完整性审计；Main 暂不自动启动。
日期：2026-07-13  
计划：`plan/single_token_frequency_dynamics_plan.md`  
数据根：`/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/`

## 1. 本轮回答什么

本轮不使用 sliding window，也不做 next-token embedding expectation。模型权重固定，不发生训练；“step”只表示把同一个隐藏状态算子反复回灌一次。Pythia-70M 的 hidden size 为 $H=512$，三组实验的观测量和 Jacobian 都只针对目标 token：

1. G1 `isolated_token`：序列长度1，目标 token hidden state 单独循环；
2. G2 `frozen_context`：前31个原始 input embedding 每一步原样重新填入，只有末 token 更新；
3. G3 `dynamic_context`：长度32的所有位置都回灌上一轮 final hidden state，但 Jacobian 只取末 token 输出对末 token 输入的条件 block。

三组都没有 padding，`attention_mask` 全1；position id 固定为 `0..L-1`；使用 causal attention；输入和输出分别为 `[1,L,512]`，去除 batch 维后 G1 状态为 `[512]`、G2 状态为 `[512]`、G3 完整状态为 `[32,512]`。

## 2. 算子与 Jacobian

令 Transformer base model（不经过 LM head）最终 hidden state 为 $T_\theta(X)$。三组目标映射为

$$
F_1(x)=T_\theta([x])_0,
$$

$$
F_2(x;C_0)=T_\theta([C_0;x])_{L-1},
$$

$$
X_{t+1}=T_\theta(X_t),\qquad
F_{3,t}(x)=T_\theta([X_t^{0:L-1};x])_{L-1}.
$$

因此 exact target Jacobian 始终是

$$
J_t=\frac{\partial F(x_t)}{\partial x_t}\in\mathbb R^{512\times512}.
$$

G3 的 $J_t$ 是沿实际完整轨迹 $X_t$ 计算的 target-conditional block，不是完整的 $16384\times16384$ Jacobian。因 causal mask，末 token 输入不应影响更早位置的同一步输出；smoke 显式检查

$$
\frac{\partial T_\theta(X_t)_{0:L-1}}{\partial X_t[L-1]}=0.
$$

## 3. 词频与样本

词频来自缓存的 WikiText-2 train tokenizer token count，不声称等于 Pythia 的真实预训练频率。只保留至少有两个长度32 context 的 lexical token type，再按 eligible rank 四等分。固定 seed 1234，每层选4个 token，共16个；所有 checkpoint 使用完全相同的 token id 和 context。

frequency audit 已得到：

- token 总数：2,419,745；
- observed token type：33,160；
- context 合格 lexical token type：29,320；
- 四层各4个 token，共16个。

## 4. 轨迹协议与收敛判据

每条轨迹运行768步。`t=0..511` 是完整保存但不用于渐近汇总的 transient；`t=512..767` 是256步 evaluation window。nearby trajectory 只在目标 token 初态加入范数为 $10^{-3}$ 的固定随机扰动。

收敛标签只使用三项：

$$
r_t=\frac{\|x_{t+1}-x_t\|_2}{\max(\|x_{t+1}\|_2,10^{-12})},
$$

$$
g_d=\frac1T\log\frac{d_T}{d_0},
$$

$$
\widehat\lambda=\frac1T\sum_t\log\|J_tv_t\|_2,
\qquad v_{t+1}=\frac{J_tv_t}{\|J_tv_t\|_2}.
$$

每条轨迹使用两条确定性 Benettin probe。`nearby_distance` 落入

$$
d_{\mathrm{floor}}=8\epsilon_{\mathrm{fp32}}\max(\|x_t\|_2,1)
$$

以下时，有限差分增长符号不再可信，标签强制降级为 `unresolved`。该阈值随每条轨迹写入 summary，不能在看到结果后修改。

标签规则：

| 标签 | tail relative delta | nearby growth | 两条 Lyapunov |
|---|---:|---:|---:|
| stable fixed-point candidate | `<1e-6` | `<0` | 均 `<0` |
| stable non-fixed candidate | `>=1e-6` | `<0` | 均 `<0` |
| expanding/chaotic candidate | `>=1e-6` | `>0` | 均 `>0` |
| unresolved | 其他或 numerical floor | 其他 | 异号/含0/证据冲突 |

这里的 “chaotic candidate” 仍不是无限时间混沌证明；G3 的 Lyapunov 是 target-conditional finite-time exponent。

## 5. 图像怎样看

### 5.1 收敛图

- `relative_step_delta`：对数纵轴；趋近数值零支持固定点候选，非零平台可能是周期、准周期、混沌或仍在暂态；
- `nearby_distance`：看 evaluation window 内总体收缩/扩张，并同时看是否触及 float32 floor；
- `Lyapunov`：固定零线；两条 probe 都为负支持局部切向收缩，都为正才支持扩张候选，异号记为未决。

三者必须联合读取。Frobenius、奇异谱、return map 或 Poincaré 点形状都不进入标签。

### 5.2 2D/3D 固定投影轨迹

使用跨 checkpoint、group、token 和 context 共享的4个随机单位向量：

$$
z_k(t)=q_k^Tx_t,\quad k=0,1,2,3.
$$

2D 轨迹画 $(z_0,z_1)$ 并用颜色表示 dynamics step；3D 轨迹画 $(z_0,z_1,z_2)$，明确标出起点和终点。它们用于发现轨迹卷曲、成环、成簇或投影重叠，不能单独证明高维状态收敛。

### 5.3 2D/3D Projected Poincaré

只取 evaluation window；每条轨迹以自己的 `median(z0)` 为截面，只保留从下向上的 crossing，并在线性插值后画：

- 2D section：$(z_1,z_2)$；
- 3D section：$(z_1,z_2,z_3)$。

无 crossing 是合法结果。有限点簇只有在 relative delta、nearby 和 Lyapunov 同时支持时才能讨论渐进周期候选；数值 floor 附近的微小 crossing 不得解释成周期。不同 token/context 的绝对坐标不混画。

## 6. Smoke 门控

step0 smoke 覆盖四个频率层各1个 token、三组算子和一个 context：

- 12 条轨迹完整；
- 36 个 exact Jacobian 均严格为 `[512,512]`；
- G3 `causal_cross_gradient_max=0`；
- 四个 projection 字段齐全；
- 无 OOM、NaN/Inf 或缺失频率层。

smoke 只用于工程门控，其48步轨迹不进入科学主结论。

## 7. Pilot 结果

### 7.1 完整性

- 8/8 shard 完成；
- 160 条轨迹、每条768步，共122,880个 trajectory row；
- 288 个 exact Jacobian，shape 全为 `[512,512]`；
- 4个固定投影跨 checkpoint/group/token/context 一致；
- `max_causal_cross_gradient=0`；
- 单元测试6项通过，无残留 `compute_single_token_dynamics` GPU 进程；
- 生成49张 PNG，每张有 JSON/Markdown sidecar。

审计汇总：`/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/processed/pilot_audit.json`。  
主收敛图：`/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/figures/convergence_diagnostics.png`。

![三项收敛判据总览](assets/single_token_frequency_dynamics/convergence_diagnostics.png)

### 7.2 联合收敛标签

| checkpoint | group | n | fixed candidate | non-fixed stable candidate | expanding/chaotic candidate | unresolved | nearby floor |
|---|---|---:|---:|---:|---:|---:|---:|
| step0 | isolated | 16 | 0 | 0 | 11 | 5 | 0 |
| step0 | frozen | 32 | 0 | 1 | 0 | 31 | 31 |
| step0 | dynamic | 32 | 0 | 0 | 0 | 32 | 32 |
| step143000 | isolated | 16 | 0 | 0 | 0 | 16 | 16 |
| step143000 | frozen | 32 | 2 | 16 | 0 | 14 | 1 |
| step143000 | dynamic | 32 | 0 | 0 | 0 | 32 | 32 |

`unresolved` 数量高不是“没有任何稳定证据”，而是严格执行 numerical-floor/冲突降级：160条中112条 nearby trajectory 进入 float32 floor，不能再用有限距离的符号完成三证据闭环。

### 7.3 三项指标的主要结构

中位数如下：

| checkpoint/group | tail relative delta | Lyapunov / step | 有效 nearby growth / step |
|---|---:|---:|---:|
| step0 isolated | 0.07645 | +0.01032 | +0.00353 |
| step0 frozen | $1.36\times10^{-7}$ | -0.03738 | -0.00052（仅1条未触 floor） |
| step0 dynamic | 0.06392 | -0.11512 | 全部触 floor |
| step143000 isolated | $3.56\times10^{-8}$ | -0.25006 | 全部触 floor |
| step143000 frozen | $1.78\times10^{-6}$ | -0.27113 | -0.00058（31条有效） |
| step143000 dynamic | $4.69\times10^{-8}$ | -0.25919 | 全部触 floor |

可以得到三个谨慎结论：

1. `step0 isolated` 是当前最明确的扩张组：13/16 token 的两条 Lyapunov probe 均为正、14/16 的有效 nearby growth 为正，最终11/16满足三项扩张候选条件；仍不能称为无限时间混沌证明。
2. 训练后的 step143000 三组 Lyapunov 全负，isolated/dynamic 的 relative delta 也降到约 $10^{-8}$；但 nearby 已进入 float32 floor，所以严格标签仍为未决，而不是强行写成稳定定点。
3. operator/context choice 改变动力学定性。step0 的 isolated 呈扩张，而 frozen/dynamic 的 target-conditional Lyapunov 均为负；因此“临界性”不能脱离输入回灌算子的定义讨论。

### 7.4 Exact normalized Frobenius

tail exact normalized Frobenius 中位数：

| checkpoint | isolated | frozen | dynamic |
|---|---:|---:|---:|
| step0, t512/t767 | 0.684/0.684 | 0.676/0.676 | 0.638/0.634 |
| step143000, t512/t767 | 0.469/0.469 | 0.459/0.459 | 0.463/0.463 |

初始 token embedding 操作点的中位数反而很大：step0 约33–37，step143000 约43–44；经过循环后均降到1以下。这直接反驳了“H1：去掉 shift 后 tail normalized Frobenius 仍自然接近1”的强版本，也再次说明 Frobenius 接近/偏离1不能代替 Lyapunov 与轨迹判据。

### 7.5 词频初步结果

四个频率层的 Lyapunov 中位数没有呈现稳定单调趋势。例如 step0 isolated 四层分别约 `0.0115, 0.0096, 0.0121, 0.0102`；step143000 isolated 约 `-0.2494, -0.2525, -0.2483, -0.2507`。Pilot 每层只有4个 token，且高频层内部 count 从37到976跨度很大，因此 H2“经验词频决定局部稳定性”当前既未获支持，也不能据此证伪；Main 前应先改进分层/匹配和统计功效。

### 7.6 Poincaré 与三投影审计

- step0 isolated 的代表轨迹可显示持续回环/散布，而某些 section 只有2个 upward crossing；少量 crossing 不足以判周期。
- step143000 dynamic 的代表轨迹在二维/三维投影中快速聚到很小区域，但 section 的29个点只在约 $10^{-6}$ 尺度抖动，同时 nearby 已触 numerical floor；这些点应解释为数值分辨率附近的 projected jitter，不能写成29周期或混沌点云。
- Poincaré 图使用单轨迹截面，2D画 $(\Delta z_1,\Delta z_2)$，3D画 $(\Delta z_1,\Delta z_2,\Delta z_3)$；仅为显示减去 crossing 均值，不改变 crossing 或几何关系。

因此，本轮图像与主判据的关系是：图像说明“轨迹投影长什么样”，三项指标决定“是否允许给稳定/扩张标签”。

step0 isolated 代表轨迹与截面：

![step0 isolated 三投影轨迹](assets/single_token_frequency_dynamics/trajectory__step0__isolated_token__bin2__token10692.png)

![step0 isolated 2D/3D Projected Poincaré](assets/single_token_frequency_dynamics/poincare__step0__isolated_token__bin2__token10692.png)

step1000 isolated 代表轨迹与截面（当前 isolated-token Lyapunov 最接近零，但截面仅2个 crossing，不能据此判周期）：

![step1000 isolated 三投影轨迹](assets/single_token_frequency_dynamics/trajectory__step1000__isolated_token__bin2__token10692.png)

![step1000 isolated 2D/3D Projected Poincaré](assets/single_token_frequency_dynamics/poincare__step1000__isolated_token__bin2__token10692.png)

step16000 dynamic 代表轨迹与截面（强负 target-conditional Lyapunov；截面坐标接近数值分辨率，5个 crossing 不等于5周期）：

![step16000 dynamic 三投影轨迹](assets/single_token_frequency_dynamics/trajectory__step16000__dynamic_context__bin2__token10692.png)

![step16000 dynamic 2D/3D Projected Poincaré](assets/single_token_frequency_dynamics/poincare__step16000__dynamic_context__bin2__token10692.png)

step143000 dynamic 代表轨迹与截面（注意 Poincaré 轴为约 $10^{-6}$ 的 centered jitter）：

![step143000 dynamic 三投影轨迹](assets/single_token_frequency_dynamics/trajectory__step143000__dynamic_context__bin2__token10692.png)

![step143000 dynamic 2D/3D Projected Poincaré](assets/single_token_frequency_dynamics/poincare__step143000__dynamic_context__bin2__token10692.png)

### 7.7 假设状态

| 假设 | Pilot 状态 | 理由 |
|---|---|---|
| H1 去掉 shift 后 Frobenius 仍接近1 | 强版本被反驳 | tail 中位数约0.46–0.68 |
| H2 词频与稳定性相关 | 未决/暂无支持 | 四层 Lyapunov 无稳定单调趋势，样本小 |
| H3 训练改变频率—动力学关系 | 未决 | checkpoint 主效应强，但 frequency interaction 尚未统计验证 |
| H4 exact Jacobian 可校准 JVP | 部分支持 | shape/矩阵/JVP路径通过；仍需正式 parity 误差表 |
| H5 不同词频进入不同吸引子 | 未决 | token 间有差异，但未形成可复现频率分层 |
| H6 context 改变 target 稳定性 | 支持 | 同 checkpoint 的 group Lyapunov/relative 行为明显不同 |
| H7 dynamic context 不同于 frozen | 支持但受 floor 限制 | step0 relative 与 Lyapunov差异明显；nearby 多数不可判 |

## 8. 当前缺陷与限制

1. G1 的目标位置是 position0，而 G2/G3 是 position31，group effect 混有 position/context 差异；必须用 length1 equivalence 和 position ablation 审计。
2. WikiText-2 经验词频不是预训练 corpus 词频，不能做频率因果结论。
3. G3 只报告 target block，不排除 prefix 子系统存在更强方向。
4. float32 nearby 在强收缩后会进入分辨率 floor；此时必须依赖更高精度或 epsilon ablation，而不是补写稳定结论。
5. 投影 Poincaré 可能丢失相位信息；2D 与3D不一致时应优先认为投影不具判别力。
6. Pilot 保存了所有目标 token state 和每步 G3 prefix norm/delta，但当前实现没有另存 G3 在 `t=0/512/767` 的完整 `[32,512]` snapshot；这不影响本轮目标级指标，Main 前必须补齐以增强复现性。
7. 四 checkpoint 合计232/320条 nearby trajectory 进入预注册 float32 floor，是当前最大的判定瓶颈；下一轮应优先做 float64 state-loop 或多 epsilon 校准，而不是直接扩大 token 数进入 Main。原始两-checkpoint Pilot 的历史统计为112/160。

## 9. 输出与复现

计划输出：

```text
/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/
  frequency_audit/
  smoke/
  pilot/step0/shard0..3/
  pilot/step1000/shard0..3/
  pilot/step16000/shard0..3/
  pilot/step143000/shard0..3/
  processed/
  figures/
  logs/
```

每张 PNG 同名生成 `.manifest.json` 和 `.md` sidecar，记录绝对数据源、回答的问题、允许解释和 caveat。

## 10. 追加实验：step1000 与 step16000

### 10.1 追加原因与协议一致性

原 Pilot 只有训练起点 `step0` 和终点 `step143000`，无法区分“平滑单调变化”和“中间发生动力学转折”。因此追加 `step1000`、`step16000`。除 checkpoint revision 外，token cohort、context、三组算子、768步轨迹、evaluation window、$epsilon=10^{-3}$、两条 Benettin probe、exact Jacobian 采样状态和四个投影向量完全不变。

四-checkpoint 新审计：

- 16/16 shard 完成；
- 320条轨迹、245,760个 trajectory row；
- 576个 exact `[512,512]` Jacobian；
- `max_causal_cross_gradient=0`；
- 232/320条 nearby trajectory 触及预注册 float32 floor；
- 共生成98张 PNG 及对应 sidecar。

新审计文件：`/home/luohaoming/model_feature_experiments/single_token_frequency_dynamics/processed/pilot_audit_all_checkpoints.json`。

### 10.2 四 checkpoint 联合结果

| checkpoint/group | tail relative delta 中位数 | Lyapunov 中位数 | t767 exact normalized Frobenius | 主标签分布 |
|---|---:|---:|---:|---|
| step0 isolated | 0.07645 | +0.01032 | 0.684 | 11 expanding，5 unresolved |
| step1000 isolated | 0.03147 | -0.00175 | 0.685 | 7 stable non-fixed，2 expanding，7 unresolved |
| step16000 isolated | $4.67\times10^{-6}$ | -0.01659 | 0.683 | 1 stable non-fixed，15 unresolved |
| step143000 isolated | $3.56\times10^{-8}$ | -0.25006 | 0.469 | 16 unresolved（全部 nearby floor） |
| step0 frozen | $1.36\times10^{-7}$ | -0.03738 | 0.676 | 1 stable non-fixed，31 unresolved |
| step1000 frozen | $1.27\times10^{-7}$ | -0.08885 | 0.692 | 32 unresolved（全部 nearby floor） |
| step16000 frozen | $5.16\times10^{-7}$ | -0.02338 | 0.617 | 2 fixed，7 stable non-fixed，2 expanding，21 unresolved |
| step143000 frozen | $1.78\times10^{-6}$ | -0.27113 | 0.459 | 2 fixed，16 stable non-fixed，14 unresolved |
| step0 dynamic | 0.06392 | -0.11512 | 0.634 | 32 unresolved |
| step1000 dynamic | 0.03077 | -0.11656 | 0.655 | 32 unresolved |
| step16000 dynamic | $4.33\times10^{-6}$ | **-0.51314** | **0.394** | 32 unresolved（全部 nearby floor） |
| step143000 dynamic | $4.69\times10^{-8}$ | -0.25919 | 0.463 | 32 unresolved（全部 nearby floor） |

![四 checkpoint 训练过渡](assets/single_token_frequency_dynamics/checkpoint_transition_summary.png)

### 10.3 step1000 与 step16000 的三投影和 Projected Poincaré

为与第7.6节的 `step0`、`step143000` 代表轨迹形成四-checkpoint 对照，这里补充同一 frequency-bin-2 token（token id `10692`，解码为 `lady`）在两个中间 checkpoint 的固定投影轨迹和 projected Poincaré section。四个 checkpoint 共用相同的随机投影方向；颜色表示 dynamics step，黑点/红叉分别表示轨迹起点/终点。Poincaré 图仍采用 evaluation window 内 `z0` 穿越自身中位数的向上 crossing，并对 crossing 坐标做仅用于显示的均值中心化。

`step1000 isolated` 是当前 isolated-token Lyapunov 最接近零的 checkpoint，因此选它观察临界过渡候选的几何形态：

![step1000 isolated 三投影轨迹](assets/single_token_frequency_dynamics/trajectory__step1000__isolated_token__bin2__token10692.png)

![step1000 isolated 2D/3D Projected Poincaré](assets/single_token_frequency_dynamics/poincare__step1000__isolated_token__bin2__token10692.png)

该轨迹从大幅瞬态进入持续的回环区域，但 evaluation window 中按预注册截面定义只有2次 upward crossing。两点截面没有足够的重复次数来区分周期、准周期或长瞬态；它只能与该组接近零但样本级正负混合的 Lyapunov 结果联合使用，不能单独证明 edge of chaos。

`step16000 dynamic` 是四个 checkpoint 中 target-conditional Lyapunov 最负、tail normalized Frobenius 最低的 dynamic-context checkpoint，因此选它展示中期强收缩候选的几何形态：

![step16000 dynamic 三投影轨迹](assets/single_token_frequency_dynamics/trajectory__step16000__dynamic_context__bin2__token10692.png)

![step16000 dynamic 2D/3D Projected Poincaré](assets/single_token_frequency_dynamics/poincare__step16000__dynamic_context__bin2__token10692.png)

投影轨迹在长瞬态后聚到很小区域，截面得到5个 crossing；但 centered crossing 坐标仅约 $10^{-4}$ 到 $10^{-3}$ 量级，且该组32/32 nearby trajectories 均进入预注册 float32 numerical floor。因而“5个点”不等于“5周期”，更保守的解释是收缩轨迹在有限精度和有限观察窗下留下的 projected residual motion。

这两组图分别回答“近零 Lyapunov 候选的轨迹几何是什么样”和“强负条件 Lyapunov 的轨迹几何是什么样”。它们不改变原有标签规则：是否收敛仍由 relative step delta、Benettin Lyapunov 与有效的 nearby separation 联合决定；周期/准周期类型还需要 recurrence、自相关/频谱和跨窗口重复性检验。

### 10.4 新结论

1. **isolated token 的零穿越发生得很早。** Lyapunov 中位数从 step0 的 `+0.0103` 到 step1000 的 `-0.00175`，step1000 仍同时包含2条扩张候选、7条稳定 non-fixed 和7条冲突样本，说明它更像临界附近的过渡 checkpoint，而不是整齐跨过单一阈值。
2. **训练变化明显非单调。** dynamic context 在 step16000 的条件 Lyapunov 达到约 `-0.513`，tail Frobenius降到约`0.394`；到 step143000 两者反而回升到约`-0.259/0.463`。因此不能把训练解释成持续增加收缩或持续靠近临界点。
3. **Frobenius 与 Lyapunov 不是同一指标。** isolated 在 step0、1000、16000 的 tail Frobenius 都约`0.68`，但 Lyapunov 从正变为近零再变负；这直接展示了局部 RMS singular gain 与沿轨迹主切向长期乘积的差别。
4. **frozen context 在 step16000 最具异质性。** 两条 Lyapunov probe多数为负，但样本分散范围跨过0；nearby 有11条有效扩张、11条有效收缩、10条触 floor，最终同时出现 fixed、stable non-fixed、expanding 和 unresolved。这里不宜只报告中位数。
5. **词频结论仍未增强。** 四个频率层在各 checkpoint/group 内没有稳定单调次序；扩展增加的是训练时间分辨率，并没有增加 token 数，因此 H2 仍是未决/暂无支持。

### 10.5 与核心论文问题的关系

四 checkpoint 结果更接近“训练导致动力学相变/跨越局部稳定边界”的 LLM 版本证据，但尚不能复现论文中“普遍自组织到临界点”的强结论：

- isolated 在 step1000 附近出现 Lyapunov 零穿越候选，值得加密 checkpoint；
- 其他算子同一 checkpoint 仍显著为负，说明临界位置依赖 operator definition；
- normalized Frobenius 没有普遍靠近1；
- 232/320 nearby floor 使稳定标签受到数值精度限制。

下一轮应优先在 `step0→step1000` 之间加密 checkpoint，并同时做 float64/epsilon 校准，而不是直接把当前零穿越写成训练临界点已被证明。
