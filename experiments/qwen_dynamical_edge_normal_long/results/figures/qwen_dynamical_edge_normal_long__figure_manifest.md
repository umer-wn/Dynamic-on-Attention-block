# qwen_dynamical_edge_normal_long Figure Manifest

## Phase Projection

- Figure: `qwen_dynamical_edge_normal_long__phase_projection.png`
- Note: `qwen_dynamical_edge_normal_long__phase_projection.md`
- Source data: `qwen_dynamical_edge_normal_long__trajectory_summary.csv`
- Axes: state norm / step delta
- Meaning: Shows the feedback trajectory in a two-scalar phase view; movement toward low step delta indicates convergence of the iterated hidden state.
- Caution: This is a diagnostic projection, not a full-dimensional phase portrait.

## Approximate Poincare Section

- Figure: `qwen_dynamical_edge_normal_long__poincare_section.png`
- Note: `qwen_dynamical_edge_normal_long__poincare_section.md`
- Source data: `qwen_dynamical_edge_normal_long__poincare_points.csv`
- Axes: step delta at crossing / nearby distance at crossing
- Meaning: Samples crossings of an empirical section defined by state norm passing above its per-sample median.
- Caution: The section is approximate and scalar-defined; it is useful for recurrence diagnostics but is not identical to a hand-selected low-dimensional map in the paper.

## Fixed-Projection Return Map

- Figure: `qwen_dynamical_edge_normal_long__return_map_projection.png`
- Note: `qwen_dynamical_edge_normal_long__return_map_projection.md`
- Source data: `qwen_dynamical_edge_normal_long__return_map_points.csv`
- Axes: z_t / z_{t+1}
- Meaning: Plots consecutive iterates after projecting every state in a trajectory onto one fixed direction, matching the paper-style return-map idea more closely than changing projections over time.
- Caution: It remains a one-dimensional projection of the full hidden state; conclusions depend on the chosen fixed projection direction.

## PCA-Summary Return Map

- Figure: `qwen_dynamical_edge_normal_long__return_map_pca_summary.png`
- Note: `qwen_dynamical_edge_normal_long__return_map_pca_summary.md`
- Source data: `qwen_dynamical_edge_normal_long__pca_summary_return_map_points.csv`
- Axes: PCA-summary z_t / PCA-summary z_{t+1}
- Meaning: Uses the first principal component of scalar trajectory summaries as a post-hoc return-map coordinate.
- Caution: The PCA direction is fitted after observing the trajectory, so it is exploratory rather than a fixed physical coordinate.

## Nearby Trajectory Distance

- Figure: `qwen_dynamical_edge_normal_long__nearby_distance_by_step.png`
- Note: `qwen_dynamical_edge_normal_long__nearby_distance_by_step.md`
- Source data: `qwen_dynamical_edge_normal_long__trajectory_summary.csv`
- Axes: step / nearby distance
- Meaning: Tracks whether a small perturbed trajectory separates from or contracts toward the reference trajectory over feedback iterations.
- Caution: Distances are measured under the configured mask and perturbation scale; they should be compared with those settings fixed.

## Lagged State-Space Distance

- Figure: `qwen_dynamical_edge_normal_long__lag_distance_by_window.png`
- Note: `qwen_dynamical_edge_normal_long__lag_distance_by_window.md`
- Source data: `qwen_dynamical_edge_normal_long__state_distance_metrics.csv`
- Axes: lag window / mean lag distance
- Meaning: Measures how far states separated by a fixed number of feedback steps are in hidden-state space.
- Caution: Averaging hides per-sample variation; inspect the CSV when the curve is flat or noisy.

## Multi-Step Jacobian Product Gain

- Figure: `qwen_dynamical_edge_normal_long__product_log_gain_by_window.png`
- Note: `qwen_dynamical_edge_normal_long__product_log_gain_by_window.md`
- Source data: `qwen_dynamical_edge_normal_long__product_jacobian_metrics.csv`
- Axes: product window / mean max log gain per step
- Meaning: Estimates expansion or contraction under the product of Jacobians across multiple feedback steps; values below zero indicate average contraction.
- Caution: This is a stochastic probe estimate, so probe count and window count affect stability.
