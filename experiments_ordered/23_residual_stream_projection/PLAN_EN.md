# Experiment 23 Plan: Residual-Stream Projection in Isolated-Token Dynamics

**Status:** design only; no formal computation has started.  
**First deliverable:** CSV files compatible with the Experiment 16/17 upload format. The existing HTML will not be modified in this phase.

## Question and state flow

For one dynamical update, set `h_t^(0)=x_t`. Pythia-70M uses six parallel-residual GPT-NeoX blocks:

```text
a_t^(l) = Attention_l(LN_attn_l(h_t^(l)))
m_t^(l) = MLP_l(LN_mlp_l(h_t^(l)))
r_t^(l) = a_t^(l) + m_t^(l)
h_t^(l+1) = h_t^(l) + r_t^(l),  l=0,...,5
u_t = h_t^(6) = x_t + sum_l r_t^(l)
x_(t+1) = FinalLayerNorm(u_t)
```

The primary quantity is the update written by the six blocks:

```text
f_internal(x_t) = sum_l r_t^(l) = u_t - x_t.
```

For comparison, also record

```text
f_effective(x_t) = x_(t+1) - x_t
final_ln_correction = x_(t+1) - u_t.
```

Only `f_effective` makes the identity `x_(t+1)=x_t+f(x_t)` exact after final LayerNorm. It is not identical to the internal residual branch, because

```text
f_effective - f_internal = final_ln_correction.
```

## Scope

- Frozen `EleutherAI/pythia-70m` in eval mode.
- The same 19 checkpoints and exactly the four-token Experiment 16 manifest—not the later eight-token Experiment 18 extension: `clones` (id 21825, count 2, bin 0), `motive` (id 23778, count 8, bin 2), `cabinet` (id 19211, count 33, bin 5), and `miles` (id 6574, count 404, bin 7). The four-dimensional orthonormal basis is also reused from Experiment 16.
- Start each trajectory from the checkpoint-specific token embedding.
- Record updates for `t=0,...,1023`; row `t` describes the update `x_t -> x_(t+1)`.
- FP32 extraction and projection, with model revision, seeds, and projection checksum recorded.

## CSV products

`processed/residual_projection_trajectory.csv` is directly uploadable to the current HTML. It retains the required columns

```text
checkpoint,dynamic_step,selection_index,token_id,token,
wikitext_train_count,frequency_bin,
projection_1,projection_2,projection_3,projection_4
```

and adds

```text
vector_kind,vector_l2,state_l2,relative_update_l2,
projection_seed,projection_sha256
```

The four projection columns contain `P f_internal(x_t)` and `vector_kind=residual_internal`.

`processed/residual_projection_components.csv` uses the same long-table schema with `vector_kind` in `{residual_internal,effective_increment,final_ln_correction}`. An optional `residual_projection_by_layer.csv` may later store attention, MLP, and total updates for every layer.

## Correctness gates

Before the full run, a `step10000`, `clones`, 16-step smoke test must verify:

1. `max_abs(u_t-x_t-sum_l r_t^(l)) < 1e-5`;
2. `max_abs(f_effective-f_internal-final_ln_correction) < 1e-5`;
3. the explicit block flow matches the original isolated-token forward with max absolute error `<1e-5`;
4. projections use exactly the Experiment 16 basis;
5. dynamic-step rows are continuous and unique;
6. the primary CSV loads successfully in the existing HTML.

## Execution stages

1. Audit the Experiment 16 basis, token manifest, checkpoint list, and CSV schema.
2. Implement extraction and pass the one-token smoke test.
3. Generate the primary CSV for 19×4×1024 = 77,824 updates, writing checkpoint parts atomically and merging them.
4. Generate the component comparison CSV and summarize residual magnitude, relative update, and final-LN correction by checkpoint/token.
5. Optionally produce the layer-level file if layer attribution is needed.

The first run does not modify the HTML, redefine the projection basis, or calculate residual Jacobian/Floquet metrics. A closed curve in residual projection means that the network's update vector has become periodic; it does not by itself prove that the state trajectory is closed.

## Fine-grained checkpoint extension

Use a 1,000-training-step spacing around the observed periodic regions:

- left of `step10000`: `step8000, step9000` (already in the base set);
- left of `step29000`: add `step27000, step28000`;
- left of `step41000`: add `step39000, step40000`;
- right of `step57000`: add `step58000, step59000`.

The six genuinely new revisions receive the same four-token 1024-step residual projection, plus exact 512×512 Jacobians at dynamic steps `0,64,...,1024`. The reported features are spectral radius and normalized Frobenius `||J||_F/sqrt(512)`—not operator norm—together with the identical fixed-512-sample Proof-Pile-2 test loss protocol from Experiment 18. Fine outputs remain separate until validation, then merge into combined visualization CSVs.
