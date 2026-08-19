# 实验15：256-step窗口Jacobian、收敛词与投影三元组

本实验沿用实验14的 isolated single-token dynamic map 和4个固定随机token。

## 正式协议

- checkpoints：`step5000, step7000, step9000, step13000, step21000,
  step29000, step37000, step53000, step61000`；
- dynamic trace：step0–2048，共2049个hidden states；
- 与实验14完全同协议的8个checkpoint复用其已验证state文件；新增step7000
  单独计算并保存在实验15数据根，加载时严格校验shape和4个token id；
- window：每256次状态转移一个窗口，共8个窗口；
- 每个窗口统一使用右端点 `h_t`（`t=256,512,...,2048`）：
  - 计算完整 `512×512` Jacobian `Df(h_t)`；
  - 报告最大模特征值（谱半径）的模、实部、虚部和第二大模；
  - 分别报告input embedding cosine最近词、欧式最近词、LM-head top1；
  - cosine使用top1-top2 similarity margin；
  - Euclidean使用绝对与相对距离margin；
  - LM head使用概率、概率margin、logit margin和归一化熵。

这里的“窗口Jacobian”明确指窗口右端点的局部Jacobian，不是对窗口内256个
Jacobian取最大值。这样Jacobian、最近token和投影坐标都对应同一个hidden state。

## 投影数据

使用与实验14相同的seed 1414 QR正交固定投影基。输出：

- `processed/projection_trajectory.csv`：扁平表；
- `processed/projection_triples.jsonl`：每行
  `[[checkpoint, dynamic_step], [projection_1, projection_2], token]`；
- `processed/projection_basis.pt`：实际使用的投影基；
- `figures/projection_window_step*.png`：8张3×3 checkpoint汇总图。

## 运行

```bash
python scripts/run_experiment15.py --stage dynamics
python scripts/run_experiment15.py --stage projection
python scripts/run_experiment15.py --stage jacobian
```

Jacobian阶段按checkpoint保存可恢复的part CSV。正式输出为
`processed/window_endpoint_jacobian_token_metrics.csv`，共
`9 checkpoints × 8 endpoints × 4 tokens = 288` 行；该主表同时合并对应
端点的 `projection_1/2`。谱半径汇总图为
`figures/jacobian_spectral_radius_by_window.png`。

三种最近词按 `checkpoint / dynamic step` 分开的完整Markdown表见
[`REPORT.md`](REPORT.md)。

## Jacobian三个尺度

补充阶段 `--stage jacobian_geometry` 对同一批288个端点Jacobian分别计算：

- 谱半径：`ρ(J)=max|λᵢ|`；
- “最大特征值”：明确采用谱横坐标 `α(J)=max Re(λᵢ)`，并保存对应复特征值
  的虚部和模；
- 算子范数：`||J||₂=σmax(J)`，即最大奇异值。

结果保存至 `processed/window_endpoint_jacobian_three_metrics.csv`，并输出三个
分指标图和一个联合图。
