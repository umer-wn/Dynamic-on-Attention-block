# Multi-Step Jacobian Product Gain

- Figure: `qwen_dynamical_edge_normal_long__product_log_gain_by_window.png`
- Source data: `qwen_dynamical_edge_normal_long__product_jacobian_metrics.csv`
- X axis: product window
- Y axis: mean max log gain per step
- Meaning: Estimates expansion or contraction under the product of Jacobians across multiple feedback steps; values below zero indicate average contraction.
- Caution: This is a stochastic probe estimate, so probe count and window count affect stability.
