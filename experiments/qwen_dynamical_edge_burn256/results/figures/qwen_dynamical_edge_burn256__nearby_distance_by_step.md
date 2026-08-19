# Nearby Trajectory Distance

- Figure: `qwen_dynamical_edge_burn256__nearby_distance_by_step.png`
- Source data: `qwen_dynamical_edge_burn256__trajectory_summary.csv`
- X axis: step
- Y axis: nearby distance
- Meaning: Tracks whether a small perturbed trajectory separates from or contracts toward the reference trajectory over feedback iterations.
- Caution: Distances are measured under the configured mask and perturbation scale; they should be compared with those settings fixed.
