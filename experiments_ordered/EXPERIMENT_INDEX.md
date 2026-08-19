# Ordered experiment index

| ID | 实验 | 状态 | 摘要 |
|---|---|---|---|
| `01_static_spectrum_archive` | 静态谱实验归档 | `archived_historical` | 早期 Jacobian/spectrum/static sensitivity 相关证据归档；不作为当前 LLM dynamics 主线证据。 |
| `02_paper_method_reconstruction` | 论文方法重构与早期动力学 | `archived_historical` | 围绕 1909.05176 的 embedding-feedback 复现、早期 dynamical-edge smoke/metrics/projection 诊断。 |
| `03_operator_exploration` | 跨模型和替代算子探索 | `archived_historical` | Pythia/GPT-2/Qwen normal dynamics、norm-matched、residual、embedding expectation、temperature sweep。 |
| `04_numeric_calibration` | 数值与方法校准 | `archived_historical` | output-gain、residual identity、epsilon sensitivity 等数值校准实验。 |
| `05_lyapunov_remeasurement` | 修正后的 Lyapunov 重测 | `full_complete` | paper-aligned Lyapunov remeasurement，建立后续 checkpoint 结论的方法基础。 |
| `06_checkpoint_criticality` | 训练 checkpoint 临界性 | `full_complete` | step0/1000/16000/143000、long asymptotic、visualization rerun 与 phase6 Poincare 审计。 |
| `07_single_token_frequency_dynamics` | 单 token 词频动力学 | `full_complete` | 按词频层选择 token，比较 isolated/frozen/dynamic context 的单 token dynamics 与 Jacobian。 |
| `08_rolling_next_token_dynamics` | Rolling next-token 动力学 | `full_complete` | 滑动窗口 next-token 使用场景、可视化补充、token-block Jacobian followup。 |
| `09_pythia_early_training_frobenius_scan` | Pythia early training Frobenius scan | `full_complete` | Pythia-70M early checkpoints 的 test loss、single-token trajectory、精确 token-level Jacobian/Frobenius 粗扫描。 |
| `10_validation_corpus_loss_rescan` | Validation corpus loss rescan | `full_complete` | The Pile/Paloma loss-only rescan merged with existing Frobenius. |
| `28_token_zipf_analysis` | 可扩展 Token Zipf 分析 | `implemented_snapshot_complete` | 支持选择数据集、文档与 token 范围；输出 unique token、词表覆盖率、逐 token-ID/rank CSV 与 Zipf 拟合。 |
| `29_checkpoint41000_initial_state_trajectories` | checkpoint 41000 不同初态轨迹 | `full_complete` | 复用 2 个 Experiment 19 token，并比较 2 个长尾 token、2 个词表模长匹配随机 embedding、2 个完全随机 embedding；输出 4096 步轨迹、候选周期、CSV、图像与中文报告。 |
