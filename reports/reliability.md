# Reliability: re-scoring with a new shuffle seed and with a different scorer

<!-- Generated from templates/reports/reliability.md; numbers come from results/main/numbers.json. -->

**Question.** How much do the scores depend on noise in the scorer LLM, on which LLM is the scorer, and on which contexts were drawn?

## Setup

- A random 50-latent subset of the main sample (seed 2), with all Exp 1 and Exp 2 conditions. The explanations and test contexts are identical to the main run, so the two re-scorings below measure the scorer's own noise only.
- **Retest:** same scorer (Llama-3.1-70B, DeepInfra), delphi shuffle seed 43 instead of 42. This changes which 5 contexts share a prompt and their order.
- **Alternative scorer:** Llama-3.3-70B-Instruct (OpenRouter default routing), seed 42. Unlike the pinned main scorer, its calls were served by several providers, which may differ in quantization (table below).
- **Split halves (no new calls):** every non-random explanation of all 200 main latents, scored on the two class-stratified halves of its 50 contexts used for H4's cross-fitting (seed 0). The halves share no contexts, so this adds the noise from which contexts were drawn. Each half has 25 contexts, so it is a lower bound on the agreement at 50.
- Cost of both re-scorings: US$2.03 for 10040 calls.

Providers that served each run, counted from the logs:

| phase | output | provider: calls |
|---|---|---|
| score | `explain` | DeepInfra: 1200 |
| score | `scores` | DeepInfra: 20020 |
| reliability | `scores_retest` | DeepInfra: 5020 |
| reliability | `scores_altscorer` | AkashML: 4056, Parasail: 907, DeepInfra: 48, Novita: 4, SambaNova: 3, Groq: 2 |

## Results

| second scoring (Spearman) | contexts | latents | expl. | J pooled | S pooled | J within | S within |
|------------------------|--------------|-------:|-------:|-------:|-------:|-------:|-------:|
| new shuffle seed (43) | same 50 | 50 | 502 | 0.961 | 0.957 | 0.909 | 0.945 |
| Llama-3.3-70B scorer | same 50 | 50 | 502 | 0.971 | 0.968 | 0.919 | 0.957 |
| other half of contexts | disjoint 25/25 | 200 | 1802 | 0.824 | 0.849 | 0.663 | 0.829 |

Spearman correlations are computed over individual explanations. *Pooled* correlates all explanations at once, so it also rewards agreeing on which latents are easy to detect. *Within latent* first subtracts each latent's mean over its explanations, so it measures agreement on which explanation of a latent is better, the comparison H2–H4 rely on.

**Does H3's direction hold under the re-scorings?** Mean S (H = 20) per budget on the 50-latent subset:

| scoring | ~2 words | ~5 | ~10 | ~20 | ~40 |
|---|---:|---:|---:|---:|---:|
| main run (Llama-3.1-70B, seed 42) | 0.409 | 0.353 | 0.336 | 0.239 | 0.105 |
| retest (Llama-3.1-70B, seed 43) | 0.423 | 0.357 | 0.344 | 0.245 | 0.107 |
| Llama-3.3-70B as scorer (seed 42) | 0.424 | 0.338 | 0.352 | 0.250 | 0.108 |

## Interpretation

On the same contexts, the scorer is consistent: pooled Spearman ≥ 0.957 under a new batch shuffle and with a different scorer model, and within latent 0.909 and 0.919 for J.

The contexts matter more than the scorer. On disjoint halves of the contexts, the within-latent Spearman of J falls to 0.663 (pooled 0.824). A single explanation's J, and any per-latent comparison between two explanations, is therefore noisy. The confirmatory claims are about means over 200 latents. Their bootstrap CIs resample latents, and each latent's scores come from its own sampled contexts, so the CIs include this noise.

Under both re-scorings, the ~2-word budget keeps the highest mean S, so the H3 conclusion (no interior optimum at H = 20) holds for one other shuffle seed and one other scorer on these 50 latents. These subset means are descriptive; no test was run on them. With Llama-3.3-70B, ~10 words scores slightly above ~5 words; both remain below ~2 words.
