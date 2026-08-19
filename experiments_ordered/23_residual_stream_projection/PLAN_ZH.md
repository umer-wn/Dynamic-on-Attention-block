# 实验23计划：单 token 动力学中的残差流投影

**状态：** 仅完成实验设计，尚未运行正式计算。  
**首要交付：** 与实验16/17上传格式兼容的 CSV；本阶段不修改可视化 HTML。

## 1. 研究问题

此前可视化研究的是单 token 状态轨迹

\[
x_t \longmapsto x_{t+1}=F(x_t).
\]

本实验不只观察状态 `x_t`，而是提取每次迭代中真正写入残差流的更新量，研究：

1. 残差更新的方向和大小是否随 dynamic step 收敛、振荡或形成周期；
2. 状态轨迹中看到的回归/周期结构，是否也存在于残差更新本身；
3. 不同 checkpoint、token 和投影方向上的残差流如何变化；
4. final LayerNorm 对“内部残差更新”与“最终状态差分”的区别有多大。

## 2. 必须明确的数据流

Pythia-70M 有6个 GPT-NeoX block，并使用 parallel residual。对第 `t` 次单 token 动力迭代：

\[
h_t^{(0)}=x_t.
\]

对第 `l=0,\ldots,5` 层：

\[
a_t^{(l)}=\operatorname{Attention}_l(\operatorname{LN}^{attn}_l(h_t^{(l)})),
\]

\[
m_t^{(l)}=\operatorname{MLP}_l(\operatorname{LN}^{mlp}_l(h_t^{(l)})),
\]

\[
r_t^{(l)}=a_t^{(l)}+m_t^{(l)},
\qquad
h_t^{(l+1)}=h_t^{(l)}+r_t^{(l)}.
\]

六层之后：

\[
u_t=h_t^{(6)}=x_t+\sum_{l=0}^{5}r_t^{(l)},
\]

\[
x_{t+1}=\operatorname{LN}_{final}(u_t).
\]

因此需要区分两个 `f(x_t)`：

- **内部残差分支（主分析对象）**

  \[
  f_{internal}(x_t)=\sum_{l=0}^{5}r_t^{(l)}=u_t-x_t.
  \]

  它表示六个 block 实际写入 residual stream 的总更新。

- **有效动力学增量（对照对象）**

  \[
  f_{effective}(x_t)=x_{t+1}-x_t.
  \]

  它严格满足 `x_{t+1}=x_t+f_effective(x_t)`，但包含 final LayerNorm 的影响，不能直接称为 Transformer block 的残差分支。

二者一般不相等：

\[
f_{effective}(x_t)-f_{internal}(x_t)
=\operatorname{LN}_{final}(u_t)-u_t.
\]

实验23将 `f_internal` 作为默认投影，并同时保存 `f_effective` 和 final-LN correction，避免概念混淆。

## 3. 实验范围

- 模型：`EleutherAI/pythia-70m`，冻结权重，eval 模式。
- checkpoint：沿用实验16的19个 checkpoint。
- token：严格沿用实验16的4-token manifest，而不是实验18扩展后的8-token manifest：`clones`（id 21825，WikiText-2 train count 2，bin 0）、`motive`（id 23778，count 8，bin 2）、`cabinet`（id 19211，count 33，bin 5）、`miles`（id 6574，count 404，bin 7）。四个token在全部checkpoint保持配对一致。
- 初始状态：每个 checkpoint 对应 token 的 input embedding。
- dynamic step：默认 `t=0...1023`，每一行 `t` 描述从 `x_t` 计算到 `x_{t+1}` 时产生的更新。
- 投影基：必须复用实验16的同一组4维正交投影基及 seed，不能重新随机生成另一组基。
- 数值精度：提取和投影使用 FP32；记录模型 revision、seed、投影基 checksum。

## 4. 每个 dynamic step 的采集流程

对每个 checkpoint–token：

1. 令 `x_0` 为 token embedding。
2. 输入 `x_t`，保持原模型前向不变；用只读 hook 捕获每层输入/输出，并捕获 final LayerNorm 的输入 `u_t`。
3. 计算 `f_internal=Σ_l r_t^(l)` 与 pre-LN 状态 `u_t=x_t+f_internal`。
4. 执行 final LayerNorm，得到 `x_(t+1)`。
5. 计算 `f_effective=x_(t+1)-x_t` 和 `final_ln_correction=x_(t+1)-u_t`。
6. 将上述向量投影到实验16的4个方向；写入CSV。
7. 令 `x_t←x_(t+1)`，进入下一步。

实现时采用 forward hook，而不重写 Transformer block：`f_internal` 直接由 final LayerNorm 输入 `u_t` 减去 `x_t` 得到；各层 `output-input` 只用于 smoke-test 验证。必须用逐项恒等式检查 hook 没有改变模型输出。

## 5. CSV设计

### 5.1 主文件：`processed/residual_projection_trajectory.csv`

此文件用于直接上传现有 HTML。保持现有可视化需要的核心字段名：

```text
checkpoint,dynamic_step,selection_index,token_id,token,
wikitext_train_count,frequency_bin,
projection_1,projection_2,projection_3,projection_4,
vector_kind,vector_l2,state_l2,relative_update_l2,
projection_seed,projection_sha256
```

其中：

- `projection_1...4` 默认存放 `f_internal(x_t)` 的四维投影；因此现有 HTML 无需改动即可把它当作一条投影轨迹上传。
- `vector_kind` 固定为 `residual_internal`，为以后扩展保留语义。
- `dynamic_step=t` 表示更新 `x_t→x_(t+1)`，不是状态 `x_(t+1)` 的编号。
- `vector_l2=||f_internal||₂`。
- `state_l2=||x_t||₂`。
- `relative_update_l2=||f_internal||₂/max(||x_t||₂,eps)`。

### 5.2 对照文件：`processed/residual_projection_components.csv`

使用长表保存三种量，字段与主文件相同，`vector_kind` 取：

- `residual_internal`：`u_t-x_t`；
- `effective_increment`：`x_(t+1)-x_t`；
- `final_ln_correction`：`x_(t+1)-u_t`。

这样未来 HTML 只需增加 `vector_kind` 过滤器即可扩展，而当前 HTML 仍可直接上传主文件。

### 5.3 可选诊断文件：`processed/residual_projection_by_layer.csv`

若完整性测试通过，再保存 layer-level 长表：

```text
checkpoint,dynamic_step,selection_index,token_id,token,
layer_index,branch,projection_1,projection_2,projection_3,projection_4,vector_l2
```

`branch∈{attention,mlp,total}`。该文件用于解释是哪一层、哪一分支造成总残差变化，不作为第一轮可视化的必需输入。

## 6. 正确性检查（运行正式实验前必须通过）

每个 smoke-test 样本检查：

1. `max_abs(u_t - x_t - Σ_l r_t^(l)) < 1e-5`；
2. `max_abs(f_effective - f_internal - final_ln_correction) < 1e-5`；
3. 显式数据流得到的 `x_(t+1)` 与原模型单 token forward 输出 `max_abs_error < 1e-5`；
4. 四维投影等于原512维向量乘以实验16投影基；
5. 同一 checkpoint–token 的CSV恰有1024个更新行，dynamic step 连续且无重复；
6. 主CSV只包含 `residual_internal`，能够被当前 HTML 成功解析。

## 7. 分阶段执行计划

### 阶段A：接口与投影基审计

- 定位实验16投影基的生成方式、seed和checksum；
- 固定 token manifest、checkpoint 列表和CSV schema；
- 写元数据说明 dynamic step 的语义。

### 阶段B：单 checkpoint smoke test

- 使用 `step10000`、`clones`、16个 dynamic steps；
- 保存完整512维中间量仅用于测试；
- 完成上述三条代数恒等式和原模型输出一致性测试。

### 阶段C：正式生成主CSV

- 对19 checkpoint × 4 token × 1024 updates 运行；
- 预计主文件恰有77,824行；
- 原始512维向量不长期保存，边运行边投影并分 checkpoint 原子写入，最后合并。

### 阶段D：对照CSV与质量报告

- 生成 `residual_projection_components.csv`；
- 汇总每个 checkpoint/token 的残差范数、相对更新量、final-LN correction比例；
- 写 `REPORT_ZH.md` / `REPORT_EN.md`，说明内部残差与有效增量的差异。

### 阶段E：可选逐层扩展

- 只有在需要解释具体层贡献时才生成逐层CSV；
- 不阻塞第一轮总残差投影交付。

## 8. 第一轮交付与非目标

第一轮交付：

- `PLAN_ZH.md` 与 `PLAN_EN.md`；
- 采集脚本；
- `residual_projection_trajectory.csv`；
- `residual_projection_components.csv`；
- smoke-test 一致性记录。

当前阶段非目标：

- 不修改实验17 HTML；
- 不重新定义投影基；
- 不计算残差流的 Jacobian/Floquet 指标；
- 不把 final LayerNorm 后的 `x_(t+1)-x_t` 错称为内部 residual branch。

## 9. 预期可视化解释

把主CSV上传现有 HTML 后，图中的点不再表示 `x_t`，而表示每一步的四维投影 `P f_internal(x_t)`。若形成闭合曲线，说明“网络每一步施加的更新向量”周期化；这不自动等价于状态轨道闭合。需要将其与原 `P x_t` 轨迹按相同 checkpoint、token、dynamic-step 区间对照解释。

## 10. 周期性 checkpoint 周边的细粒度扩展

针对当前观察到周期结构的位置，按1000 training-step间隔扩展：

- `step10000` 左侧：`step8000, step9000`（二者已经在基础19-checkpoint集合中）；
- `step29000` 左侧：新增 `step27000, step28000`；
- `step41000` 左侧：新增 `step39000, step40000`；
- `step57000` 右侧：新增 `step58000, step59000`。

因此需要新下载和计算的模型revision一共6个。每个新增checkpoint统一生成：

1. 4个实验16 token的1024步 `f_internal` 四维投影；
2. dynamic step `0,64,...,1024` 上的完整512×512 Jacobian；
3. `spectral radius = max|lambda_i(J)|`；
4. `normalized Frobenius = ||J||_F/sqrt(512)`，不是算子范数；
5. 与实验18完全相同的固定512样本 Proof-Pile-2 test loss。

细粒度结果独立写入 `fine_*` 文件，确认完整后再生成基础+细粒度的 combined CSV，避免覆盖原数据。
