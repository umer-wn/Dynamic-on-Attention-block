# Model Download Prep

## Requested Model Types

The target is base/pretrained language models, not instruction-tuned or aligned chat models.

## Download Status

| model | status | note |
|---|---|---|
| `gpt2` | downloaded | GPT-2 base pretrained model; no instruction tuning variant selected |
| `Qwen/Qwen2.5-0.5B` | downloaded | Base Qwen2.5 0.5B model; not the `-Instruct` variant |
| `google/gemma-3-270m` | failed | Hugging Face gated repo; requires authentication and accepted access terms |

Cache path:

```text
/public/luohaoming/model_feature/hf_cache
```

## Current Recommendation

Use these for the next comparable base-model experiments:

```text
EleutherAI/pythia-70m
gpt2
Qwen/Qwen2.5-0.5B
```

Skip Gemma until authenticated access is available, or provide a Hugging Face token with accepted Gemma terms.

## Why These Are Suitable

- `EleutherAI/pythia-70m`: controlled pretrained checkpoint from the Pythia suite.
- `gpt2`: classic GPT-style pretrained causal language model.
- `Qwen/Qwen2.5-0.5B`: base model name without `Instruct`; suitable as a modern small pretrained comparator.

Do not use model IDs containing:

```text
Instruct
Chat
RLHF
SFT
```

unless the experiment explicitly studies post-training/alignment effects.
