# pythia_embedding_expectation_temp025 Figure Manifest

## Phase Projection

- Figure: `pythia_embedding_expectation_temp025__phase_projection.png`
- Note: `pythia_embedding_expectation_temp025__phase_projection.md`
- Source data: `pythia_embedding_expectation_temp025__trajectory_summary.csv`
- Axes: state norm / step delta
- Meaning: Shows the feedback trajectory in a two-scalar phase view; movement toward low step delta indicates convergence of the iterated hidden state.
- Caution: This is a diagnostic projection, not a full-dimensional phase portrait.

## Approximate Poincare Section

- Figure: `pythia_embedding_expectation_temp025__poincare_section.png`
- Note: `pythia_embedding_expectation_temp025__poincare_section.md`
- Source data: `pythia_embedding_expectation_temp025__poincare_points.csv`
- Axes: step delta at crossing / nearby distance at crossing
- Meaning: Samples crossings of an empirical section defined by state norm passing above its per-sample median.
- Caution: The section is approximate and scalar-defined; it is useful for recurrence diagnostics but is not identical to a hand-selected low-dimensional map in the paper.

## Fixed-Projection Return Map

- Figure: `pythia_embedding_expectation_temp025__return_map_projection.png`
- Note: `pythia_embedding_expectation_temp025__return_map_projection.md`
- Source data: `pythia_embedding_expectation_temp025__return_map_points.csv`
- Axes: z_t / z_{t+1}
- Meaning: Plots consecutive iterates after projecting every state in a trajectory onto one fixed direction, matching the paper-style return-map idea more closely than changing projections over time.
- Caution: It remains a one-dimensional projection of the full hidden state; conclusions depend on the chosen fixed projection direction.

## PCA-Summary Return Map

- Figure: `pythia_embedding_expectation_temp025__return_map_pca_summary.png`
- Note: `pythia_embedding_expectation_temp025__return_map_pca_summary.md`
- Source data: `pythia_embedding_expectation_temp025__pca_summary_return_map_points.csv`
- Axes: PCA-summary z_t / PCA-summary z_{t+1}
- Meaning: Uses the first principal component of scalar trajectory summaries as a post-hoc return-map coordinate.
- Caution: The PCA direction is fitted after observing the trajectory, so it is exploratory rather than a fixed physical coordinate.

## Nearby Trajectory Distance

- Figure: `pythia_embedding_expectation_temp025__nearby_distance_by_step.png`
- Note: `pythia_embedding_expectation_temp025__nearby_distance_by_step.md`
- Source data: `pythia_embedding_expectation_temp025__trajectory_summary.csv`
- Axes: step / nearby distance
- Meaning: Tracks whether a small perturbed trajectory separates from or contracts toward the reference trajectory over feedback iterations.
- Caution: Distances are measured under the configured mask and perturbation scale; they should be compared with those settings fixed.

## Lagged State-Space Distance

- Figure: `pythia_embedding_expectation_temp025__lag_distance_by_window.png`
- Note: `pythia_embedding_expectation_temp025__lag_distance_by_window.md`
- Source data: `pythia_embedding_expectation_temp025__state_distance_metrics.csv`
- Axes: lag window / mean lag distance
- Meaning: Measures how far states separated by a fixed number of feedback steps are in hidden-state space.
- Caution: Averaging hides per-sample variation; inspect the CSV when the curve is flat or noisy.
