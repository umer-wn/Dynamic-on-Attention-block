# The Pile / Paloma Loss Re-evaluation 报告

状态：`complete`

数据根：`/home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan`

图表数：4

## step16000 vs step100000

source_id,step16000_loss,step100000_loss,delta_100000_minus_16000,label
the_pile_test,3.57926635325978,3.5510813123075886,-0.0281850409521915,no_rebound_detected_on_this_source
the_pile_validation,3.568465662568335,3.5186532720195327,-0.0498123905488023,no_rebound_detected_on_this_source

## Coarse-only completion update

The canonical scope is now the coarse-grained checkpoint scan, not the dense all-100 scan.

Audited checkpoint target:

```text
coarse: step1000, step5000, ..., step97000
sentinel: step0, step10000, step16000, step100000, step143000
```

Direct HF cache inspection found:

- model weights: 30/30 complete;
- The Pile validation loss: 30/30 complete;
- The Pile test loss: 30/30 complete.

The earlier suspicion that about 10 coarse weights remained missing was not supported by the `refs/snapshots` audit. The only missing coarse loss was `the_pile_validation/step10000`, and it has now been computed.

Status:

```text
/home/luohaoming/model_feature_experiments/pythia_validation_corpus_loss_rescan/status/coarse_checkpoint_loss_completion.json
```

## Final checkpoint check: step143000

To test whether overfitting appears only at the last checkpoint, `step143000` was evaluated on the same fixed The Pile validation/test manifests.

```text
source_id,step16000_loss,step100000_loss,step143000_loss,delta_143000_minus_100000,label_143000_vs_100000,delta_143000_minus_16000
the_pile_validation,3.568465662568335,3.5186532720195327,3.526936224659186,0.008282952639653285,final_rebound_candidate,-0.04152943790914909
the_pile_test,3.57926635325978,3.5510813123075886,3.5515952413652396,0.0005139290576510191,final_rebound_candidate,-0.027671111894540523
```

Interpretation: `step143000` is slightly worse than `step100000` on both sampled The Pile validation and test. The validation rebound is visible but modest; the test rebound is extremely small. Relative to `step16000`, the final checkpoint remains better.

Outputs:

```text
processed/step143000_final_checkpoint_comparison_by_source.csv
figures/step143000_final_checkpoint_loss_check.png
status/final_checkpoint_step143000_complete.json
```

## Paloma mirror probe

Paloma was retried through `hf-mirror.com`.

Observed behavior:

- `https://hf-mirror.com/api/datasets/allenai/paloma`: reachable, returns metadata and indicates gated access.
- `https://hf-mirror.com/datasets/allenai/paloma/resolve/main/README.md`: reachable.
- `https://hf-mirror.com/datasets/allenai/paloma/raw/main/README.md`: returns 401 with an authentication-required message.
- `datasets` with `HF_ENDPOINT=https://hf-mirror.com` still reports `allenai/paloma` as gated and requiring authentication.

Conclusion: the mirror does not bypass Paloma gating. Paloma remains blocked until `myserver` has authenticated HuggingFace access accepted for `allenai/paloma`.

Evidence:

```text
logs/paloma_mirror_probe.log
logs/prepare_paloma_mirror_retry.log
status/paloma_access_blocked.json
```
