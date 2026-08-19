# Lagged State-Space Distance

- Figure: `qwen_dynamical_edge_normal__lag_distance_by_window.png`
- Source data: `qwen_dynamical_edge_normal__state_distance_metrics.csv`
- X axis: lag window
- Y axis: mean lag distance
- Meaning: Measures how far states separated by a fixed number of feedback steps are in hidden-state space.
- Caution: Averaging hides per-sample variation; inspect the CSV when the curve is flat or noisy.
