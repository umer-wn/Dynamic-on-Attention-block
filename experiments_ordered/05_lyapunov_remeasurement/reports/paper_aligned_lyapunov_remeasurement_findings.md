# Paper-Aligned Lyapunov Remeasurement Findings

## Status

Complete. This experiment corrects two non-equivalent historical diagnostics: random-direction product gain and post-burn-in distance ratio. It adds a Benettin-style maximal finite-time Lyapunov estimate, final asymptotic separation from the original perturbation, and relative step delta.

## Results

| residual alpha | normalized Frobenius | maximal Lyapunov | across-sample std | final/initial separation | final relative step delta | phase |
|---:|---:|---:|---:|---:|---:|---|
| 0.0 | 1.000000 | approximately 0 | approximately 0 | 1.0000 | 0 | 4/4 fixed-like |
| 0.5 | 0.729223 | -0.138064 | 0.000932 | 0.1628 | 4.9e-9 | 4/4 fixed-like |
| 1.0 | 0.464398 | -0.265742 | 0.002433 | 0.1564 | 5.2e-8 | 4/4 fixed-like |

The identity control passes all analytical expectations. The native constructed operator (`alpha=1`) has a negative maximal finite-time Lyapunov estimate on every tested sample and converges by the relative-step criterion. The agreement between negative Lyapunov exponent, bounded trajectories, reduced final perturbation separation, and very small relative step delta is substantially stronger evidence of local asymptotic contraction than previous reports provided.

## What This Result Supports

For Pythia-70M `main`, sequence length 64, and these four WikiText inputs, the constructed deterministic map `inputs_embeds -> final_hidden -> inputs_embeds` approaches fixed-like attractors and has a negative measured leading finite-time Lyapunov exponent. It is on the stable side of the paper's dynamical boundary under this protocol.

## What It Does Not Support

- It does not show that native autoregressive token generation is contractive.
- It does not show that all prompts, sequence lengths, models, or checkpoints share this phase.
- It does not prove that the Transformer Jacobian satisfies the random-matrix assumptions behind Frobenius equivalence.
- It does not prove or falsify "optimal intelligence at the edge of chaos," because no model-quality curve across training checkpoints has been compared with the stability metric.
- It does not yet match the paper's 100 samples and 500 iterations.

## Historical Corrections

The earlier random-direction `product_log_gain` must not be called a maximal Lyapunov exponent. The earlier nearby growth ratio must not be called the paper's final asymptotic separation. Old reports should be interpreted through the correction audit in `reports/core_paper_alignment_audit_20260712.md`.

## Reproducibility

- Plan: `/data1/luohaoming/model_feature/plan/paper_aligned_lyapunov_remeasurement_plan.md`
- Configs: `/data1/luohaoming/model_feature/configs/pythia_lyapunov_alpha*.yaml`
- Raw and processed data: `/home/luohaoming/model_feature_experiments/paper_aligned_lyapunov_remeasurement`
- Logs: `/home/luohaoming/model_feature_experiments/paper_aligned_lyapunov_remeasurement/logs`
- Analysis: `/data1/luohaoming/model_feature/scripts/analyze_paper_aligned_lyapunov.py`

## Next Scientific Experiment

The core hypothesis requires Pythia training checkpoints with validation loss/perplexity and paper-aligned Lyapunov/Frobenius metrics. Recommended revisions are `step0`, `step1000`, `step16000`, and `step143000`, followed by a larger 100-sample/500-iteration confirmation only if a checkpoint trend is visible.
