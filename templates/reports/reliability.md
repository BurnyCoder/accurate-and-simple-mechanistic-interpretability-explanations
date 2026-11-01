# Reliability: re-scoring with a new shuffle seed and with a different scorer

<!-- Generated from templates/reports/reliability.md; numbers come from results/main/numbers.json. -->

**Question.** How much do the scores depend on noise in the scorer LLM, on which LLM is the scorer, and on which contexts were drawn?

## Setup

- A random 50-latent subset of the main sample (seed 2), with all Exp 1 and Exp 2 conditions. The explanations and test contexts are identical to the main run, so the two re-scorings below measure the scorer's own noise only.
- **Retest:** same scorer (Llama-3.1-70B, DeepInfra), delphi shuffle seed 43 instead of 42. This changes which 5 contexts share a prompt and their order.
- **Alternative scorer:** Llama-3.3-70B-Instruct (OpenRouter default routing), seed 42. Unlike the pinned main scorer, its calls were served by several providers, which may differ in quantization (table below).
- **Split halves (no new calls):** every non-random explanation of all {{n_latents}} main latents, scored on the two class-stratified halves of its 50 contexts used for H4's cross-fitting (seed 0). The halves share no contexts, so this adds the noise from which contexts were drawn. Each half has 25 contexts, so it is a lower bound on the agreement at 50.
- Cost of both re-scorings: US${{api_usage.reliability.cost_usd:.2f}} for {{api_usage.reliability.calls}} calls.

Providers that served each run, counted from the logs:

{{table:providers}}

## Results

{{table:reliability}}

Spearman correlations are computed over individual explanations. *Pooled* correlates all explanations at once, so it also rewards agreeing on which latents are easy to detect. *Within latent* first subtracts each latent's mean over its explanations, so it measures agreement on which explanation of a latent is better, the comparison H2–H4 rely on.

**Does H3's direction hold under the re-scorings?** Mean S (H = 20) per budget on the 50-latent subset:

| scoring | ~2 words | ~5 | ~10 | ~20 | ~40 |
|---|---:|---:|---:|---:|---:|
| main run (Llama-3.1-70B, seed 42) | {{reliability_retest.mean_s_by_budget_main_subset.gen_b2:.3f}} | {{reliability_retest.mean_s_by_budget_main_subset.gen_b5:.3f}} | {{reliability_retest.mean_s_by_budget_main_subset.gen_b10:.3f}} | {{reliability_retest.mean_s_by_budget_main_subset.gen_b20:.3f}} | {{reliability_retest.mean_s_by_budget_main_subset.gen_b40:.3f}} |
| retest (Llama-3.1-70B, seed 43) | {{reliability_retest.mean_s_by_budget_other.gen_b2:.3f}} | {{reliability_retest.mean_s_by_budget_other.gen_b5:.3f}} | {{reliability_retest.mean_s_by_budget_other.gen_b10:.3f}} | {{reliability_retest.mean_s_by_budget_other.gen_b20:.3f}} | {{reliability_retest.mean_s_by_budget_other.gen_b40:.3f}} |
| Llama-3.3-70B as scorer (seed 42) | {{reliability_altscorer.mean_s_by_budget_other.gen_b2:.3f}} | {{reliability_altscorer.mean_s_by_budget_other.gen_b5:.3f}} | {{reliability_altscorer.mean_s_by_budget_other.gen_b10:.3f}} | {{reliability_altscorer.mean_s_by_budget_other.gen_b20:.3f}} | {{reliability_altscorer.mean_s_by_budget_other.gen_b40:.3f}} |

## Interpretation

On the same contexts, the scorer is consistent: pooled Spearman ≥ {{reliability_min_spearman:.3f}} under a new batch shuffle and with a different scorer model, and within latent {{reliability_retest.spearman_j_within:.3f}} and {{reliability_altscorer.spearman_j_within:.3f}} for J.

The contexts matter more than the scorer. On disjoint halves of the contexts, the within-latent Spearman of J falls to {{reliability_split_half.spearman_j_within:.3f}} (pooled {{reliability_split_half.spearman_j:.3f}}). A single explanation's J, and any per-latent comparison between two explanations, is therefore noisy. The confirmatory claims are about means over {{n_latents}} latents. Their bootstrap CIs resample latents, and each latent's scores come from its own sampled contexts, so the CIs include this noise.

Under both re-scorings, the ~2-word budget keeps the highest mean S, so the H3 conclusion (no interior optimum at H = 20) holds for one other shuffle seed and one other scorer on these 50 latents. These subset means are descriptive; no test was run on them. With Llama-3.3-70B, ~10 words scores slightly above ~5 words; both remain below ~2 words.
