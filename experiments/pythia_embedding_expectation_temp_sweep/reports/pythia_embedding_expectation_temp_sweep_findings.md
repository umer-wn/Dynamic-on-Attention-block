# Pythia Embedding-Expectation Temperature Sweep Findings

## Run

- Date: 2026-07-11
- Plan: `experiments/pythia_embedding_expectation_temp_sweep/plan/pythia_embedding_expectation_temp_sweep_plan.md`
- Summary CSV: `experiments/pythia_embedding_expectation_temp_sweep/results/temperature_sweep_summary.csv`
- Model: `EleutherAI/pythia-70m`
- Operator: `inputs_embeds -> logits -> softmax(logits / T) -> expected input embedding`

## Results

| temperature | mean normalized Frobenius | nearby log growth mean | settled fraction | max final step delta |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.110450 | -0.128754 | 0.5 | 1.678900 |
| 0.5 | 0.343072 | 0.100781 | 0.5 | 3.062361 |
| 1.0 | 0.244026 | -0.051410 | 0.0 | 1.296017 |
| 2.0 | 0.135963 | 0.066774 | 0.5 | 0.037793 |

## Interpretation

Temperature changes the dynamics, but none of the tested values recover the paper's edge criterion. `T=0.5` is the closest smoke setting, but it is still far below 1 and has unstable/unsettled trajectory behavior. `T=2.0` is more settled but more contractive.

The pretrained LM-head embedding-expectation map is a better same-space operator than final-hidden feedback, but it still does not reproduce the paper. The next credible attempt should train an explicit same-space reconstruction/denoising map, then evaluate the asymptotic Jacobian after training.
