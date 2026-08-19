# Pythia Residual-Operator Calibration Findings

## Outcome

The exact identity anchor passed.

| alpha | normalized Frobenius | nearby log growth/step | product gain window 2 | product gain window 4 | phase |
|---:|---:|---:|---:|---:|---|
| 0.0 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | stable fixed-like |
| 0.5 | about 0.72945 | about -0.00024 | about 0.5356 | about 0.2945 | bounded nonfixed-like |

All four identity samples returned local Frobenius values exactly equal to one, constant perturbation distance, and zero state-step differences. No run diverged or collapsed. The half-residual operator moved coherently into a contractive regime.

## Review

This validates the implementation of the local Frobenius estimator, nearby-trajectory propagation, and multi-step JVP product on an exact control. It does not validate the scientific choice of `inputs_embeds -> final_hidden` as the LLM analogue of the paper's image-model operator.

The `alpha=0.5` trajectory approaches a state norm near 1193 while its absolute step delta drops toward `2e-5`. Convergence should therefore use a relative delta (`||x[t+1]-x[t]|| / ||x[t]||`) in future work. Nearby distances around `1e-4`, despite an initial perturbation of `1e-5`, also motivate an epsilon/precision sweep.

## Data and Logs

- Raw: `/data1/luohaoming/model_feature/results/raw/pythia_residual_alpha*`
- Logs: `/data1/luohaoming/model_feature/logs/pythia_residual_calibration`
- Configs: `/data1/luohaoming/model_feature/configs/pythia_residual_alpha*.yaml`

The experiment is complete. The next experiment is numerical-sensitivity calibration across perturbation epsilon values.
