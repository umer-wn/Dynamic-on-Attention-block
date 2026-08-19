# 100-token endpoint metrics across 19 Pythia-70M checkpoints

## Protocol

- The same 100 unique WikiText-2 train token types are paired across every checkpoint; 10 tokens are sampled from each frequency decile with seed `190905176`.
- Each token starts from the checkpoint-specific input embedding and is propagated through the frozen isolated-token map for 1024 dynamic steps.
- At `x_1024`, an exact `512 x 512` Jacobian is constructed.
- Spectral radius: `rho(J) = max_i |lambda_i(J)|`.
- Normalized Frobenius: `||J||_F / sqrt(512)`.
- Lyapunov exponent: exact JVP/Benettin tangent propagation with per-step renormalization; the reported value is the mean log growth over dynamic steps 768-1024. The 0-1024 value is retained in token-level CSV files as an auxiliary column.
- Every checkpoint value below is the arithmetic mean over 100 per-token scalar metrics. CSV also stores standard deviation, median, min/max, SEM and normal-approximation 95% CI.

## Means over 100 paired tokens

| checkpoint | spectral radius | Lyapunov (768-1024) | normalized Frobenius |
|---|---:|---:|---:|
| step0 | 1.024472 | +0.009189 | 0.680538 |
| step2000 | 0.985608 | -0.014498 | 0.643691 |
| step3000 | 0.957951 | -0.042959 | 0.628343 |
| step4000 | 0.988868 | -0.011251 | 0.663474 |
| step5000 | 1.012245 | +0.007556 | 0.679212 |
| step7000 | 1.027762 | +0.011212 | 0.701312 |
| step8000 | 1.028802 | +0.011269 | 0.705073 |
| step9000 | 0.994399 | -0.005626 | 0.694679 |
| step10000 | 0.997978 | +0.000023 | 0.681811 |
| step13000 | 0.951643 | -0.049566 | 0.670617 |
| step21000 | 1.057036 | +0.021844 | 0.709699 |
| step25000 | 1.017047 | +0.008164 | 0.686957 |
| step29000 | 1.014271 | -0.000680 | 0.687454 |
| step33000 | 1.052190 | +0.016266 | 0.701251 |
| step37000 | 1.018137 | +0.007625 | 0.692648 |
| step41000 | 1.054608 | +0.000105 | 0.684911 |
| step53000 | 1.074951 | +0.000535 | 0.669065 |
| step57000 | 0.981669 | +0.003473 | 0.604645 |
| step61000 | 0.899905 | -0.105471 | 0.591459 |

## Descriptive checks

- Highest mean spectral radius: `step53000` = `1.074951`.
- Highest mean last-window Lyapunov: `step21000` = `+0.021844`; lowest: `step61000` = `-0.105471`.
- Highest normalized Frobenius mean: `step21000` = `0.709699`; lowest: `step61000` = `0.591459`.
- These local endpoint/window metrics do not by themselves prove a global attractor or chaos classification.

## Outputs

- `processed/checkpoint_metric_summary.csv`: 19-checkpoint summary with distribution statistics and 95% CI.
- `processed/checkpoint_parts/step*.csv`: 1900 token-level rows.
- `figures/spectral_radius_lyapunov_normalized_frobenius_by_checkpoint.png`: combined three-panel chart.
- `figures/spectral_radius_by_checkpoint.png`, `lyapunov_exponent_last_256_by_checkpoint.png`, `normalized_frobenius_norm_by_checkpoint.png`: individual charts.
