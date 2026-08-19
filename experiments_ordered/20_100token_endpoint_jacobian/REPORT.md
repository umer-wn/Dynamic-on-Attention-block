# 实验20：100-token、dynamic step 1024 端点 Jacobian 归一化 Frobenius 范数

## 协议

- 模型：`EleutherAI/pythia-70m`，checkpoint 为 `step10000 / step41000 / step53000 / step57000`。
- 动力学：沿用实验15/16的 isolated single-token map。每个 token 从当前 checkpoint 的 input embedding 出发，冻结模型权重，迭代到 dynamic step 1024。
- token cohort：从 WikiText-2 train 中出现过的非特殊 token 类型按词频十分位分层，每层固定抽取10个，共100个唯一 token；随机种子 `190905176`。四个 checkpoint 使用完全相同的配对 cohort。
- Jacobian：在每个 token 的 `x_1024` 处构造完整 `512×512` exact Jacobian `J = Df(x_1024)`。
- 核心指标：
  - 谱半径 `rho(J) = max_i |lambda_i|`；
  - 谱横坐标 `alpha(J) = max_i Re(lambda_i)`；
  - 归一化 Frobenius 范数 `||J||_F / sqrt(N) = sqrt(sum_ij J_ij^2 / N)`，其中 `N=512`。
- 算子2-范数 `sigma_max(J)`仍保留在原始表中作为辅助诊断，但不作为本轮所称的“归一化范数”。
- 聚合：先对每个 token 独立计算指标，再对100个 token 的标量指标取算术平均；不是先平均 Jacobian 再求指标。
- 95% CI：`mean ± 1.96 × sample_std / sqrt(100)`，描述的是 token cohort 均值的不确定区间。

## 结果

| checkpoint | mean rho (95% CI) | mean alpha (95% CI) | mean normalized Frobenius (95% CI) |
|---|---:|---:|---:|
| step10000 | 0.997978 [0.992187, 1.003770] | 0.966090 [0.961271, 0.970910] | 0.681811 [0.679011, 0.684611] |
| step41000 | 1.054608 [1.040950, 1.068266] | 1.053066 [1.039162, 1.066969] | 0.684911 [0.680844, 0.688978] |
| step53000 | 1.074951 [1.063284, 1.086619] | 1.066178 [1.055161, 1.077196] | 0.669065 [0.665082, 0.673048] |
| step57000 | 0.981669 [0.967880, 0.995457] | 0.981658 [0.967869, 0.995447] | 0.604645 [0.601529, 0.607760] |

## 描述性观察

1. `step10000` 的平均谱半径最接近1，95%区间跨过1；其平均谱横坐标低于1。
2. `step41000` 和 `step53000` 的平均谱半径、谱横坐标均高于1；归一化 Frobenius 均值最高的是 `step41000`。
3. `step57000` 的两个特征值尺度均值低于1，但 token 级谱半径最大值为1.275848，说明均值不能替代 token 级异质性检查。
4. 归一化 Frobenius 范数从 `step41000` 的约0.685下降到 `step57000` 的约0.605；这些端点局部指标本身不等价于全局混沌或吸引子类型判定。

## 数据文件

- `manifests/tokens_100_frequency_deciles.csv`：固定100-token cohort。
- `processed/checkpoint_parts/step*.csv`：400行 token 级原始结果。
- `processed/normalized_frobenius_summary.csv`：归一化 Frobenius 的均值、标准差、中位数、范围、SEM和95% CI。
- `logs/step*.log`：逐 token 运行日志。
