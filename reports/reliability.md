# Reliability: re-scoring with a new shuffle seed and with a different scorer

<!-- Generated from templates/reports/reliability.md; numbers come from results/main/numbers.json. -->

**Question.** How much do the scores depend on noise in the scorer LLM, and on which LLM is the scorer?

## Setup

- A random 50-latent subset of the main sample (seed 2), with all Exp 1 and Exp 2 conditions. The explanations and test contexts are identical to the main run.
- **Retest:** same scorer (Llama-3.1-70B, DeepInfra), delphi shuffle seed 43 instead of 42. This changes which 5 contexts share a prompt and their order.
- **Alternative scorer:** Llama-3.3-70B-Instruct (OpenRouter default routing), seed 42.
- Cost of both re-scorings: US$2.03 for 10040 calls.

## Results

| re-scoring | latents | explanations | Spearman J | Spearman S |
|---|---:|---:|---:|---:|
| same scorer, new shuffle seed (43) | 50 | 502 | 0.961 | 0.957 |
| Llama-3.3-70B as scorer | 50 | 502 | 0.971 | 0.968 |

Spearman correlations are computed over individual explanations, each scored on only 50 contexts, so they include sampling noise from the scorer's per-batch answers.

**Does H3's direction hold under the alternative scorer?** Mean S (H = 20) per budget with Llama-3.3-70B as the scorer, on the 50-latent subset:

| ~2 words | ~5 | ~10 | ~20 | ~40 |
|---:|---:|---:|---:|---:|
| 0.424 | 0.338 | 0.352 | 0.250 | 0.108 |

Retest with Llama-3.1-70B and seed 43, same subset:

| ~2 words | ~5 | ~10 | ~20 | ~40 |
|---:|---:|---:|---:|---:|
| 0.423 | 0.357 | 0.344 | 0.245 | 0.107 |

## Interpretation

Per-explanation J and S rankings are highly reproducible: Spearman ≥ 0.957, both under a new batch shuffle and with a different scorer model.

Under both re-scorings, the ~2-word budget keeps the highest mean S, so the H3 conclusion (no interior optimum at H = 20) does not depend on the batching or on the scorer. With Llama-3.3-70B, ~10 words scores slightly above ~5 words; both remain below ~2 words.
