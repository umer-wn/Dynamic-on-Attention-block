# Fixed-Projection Return Map

- Figure: `pythia_embedding_expectation_temp025__return_map_projection.png`
- Source data: `pythia_embedding_expectation_temp025__return_map_points.csv`
- X axis: z_t
- Y axis: z_{t+1}
- Meaning: Plots consecutive iterates after projecting every state in a trajectory onto one fixed direction, matching the paper-style return-map idea more closely than changing projections over time.
- Caution: It remains a one-dimensional projection of the full hidden state; conclusions depend on the chosen fixed projection direction.
