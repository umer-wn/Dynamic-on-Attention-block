# Pythia Residual-Operator Calibration Plan

## Purpose

Validate the complete dynamics pipeline against an exact mathematical anchor before interpreting LLM results. Define `F_alpha(x) = (1-alpha)x + alpha f(x)`.

At `alpha=0`, `F` is exactly identity. The expected normalized Frobenius, two/four-step tangent gain, and nearby-trajectory ratio are all exactly one. At `alpha=0.5`, the result should move away from the identity anchor toward the native `alpha=1` operator.

## Setup

- Model: cached `EleutherAI/pythia-70m`, revision `main`.
- Four WikiText validation samples, sequence length 64.
- `alpha=0` on GPU 5; `alpha=0.5` on GPU 6.
- Rademacher Frobenius probes and Jacobian-product windows 2 and 4.

## Pass Criteria

The identity run must return Frobenius `1`, product gains `1`, nearby log growth `0`, zero step delta, and no divergence/collapse. The `alpha=0.5` run must produce finite values and a coherent departure from the identity anchor.

## Storage Note

This calibration was executed before the storage-policy update. Its raw outputs and logs are under `/data1/luohaoming/model_feature/results` and `/data1/luohaoming/model_feature/logs`. All subsequent experiment data will be stored under `/home/luohaoming/model_feature_experiments`.
