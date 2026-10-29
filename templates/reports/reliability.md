# Reliability: re-scoring with a new shuffle seed and with a different scorer

<!-- Generated from templates/reports/reliability.md; numbers come from results/main/numbers.json. -->

**Question.** How much do the scores depend on noise in the scorer LLM, and on which LLM is the scorer?

## Setup

- A random 50-latent subset of the main sample (seed 2), with all Exp 1 and Exp 2 conditions. The explanations and test contexts are identical to the main run.
- **Retest:** same scorer (Llama-3.1-70B, DeepInfra), delphi shuffle seed 43 instead of 42. This changes which 5 contexts share a prompt and their order.
- **Alternative scorer:** Llama-3.3-70B-Instruct (OpenRouter default routing), seed 42.
- Cost of both re-scorings: US${{api_usage.reliability.cost_usd:.2f}} for {{api_usage.reliability.calls}} calls.

## Results

{{table:reliability}}

Spearman correlations are computed over individual explanations, each scored on only 50 contexts, so they include sampling noise from the scorer's per-batch answers.

**Does H3's direction hold under the alternative scorer?** Mean S (H = 20) per budget with Llama-3.3-70B as the scorer, on the 50-latent subset:

| ~2 words | ~5 | ~10 | ~20 | ~40 |
|---:|---:|---:|---:|---:|
| {{reliability_altscorer.mean_s_by_budget_other.gen_b2:.3f}} | {{reliability_altscorer.mean_s_by_budget_other.gen_b5:.3f}} | {{reliability_altscorer.mean_s_by_budget_other.gen_b10:.3f}} | {{reliability_altscorer.mean_s_by_budget_other.gen_b20:.3f}} | {{reliability_altscorer.mean_s_by_budget_other.gen_b40:.3f}} |

Retest with Llama-3.1-70B and seed 43, same subset:

| ~2 words | ~5 | ~10 | ~20 | ~40 |
|---:|---:|---:|---:|---:|
| {{reliability_retest.mean_s_by_budget_other.gen_b2:.3f}} | {{reliability_retest.mean_s_by_budget_other.gen_b5:.3f}} | {{reliability_retest.mean_s_by_budget_other.gen_b10:.3f}} | {{reliability_retest.mean_s_by_budget_other.gen_b20:.3f}} | {{reliability_retest.mean_s_by_budget_other.gen_b40:.3f}} |

## Interpretation

Per-explanation J and S rankings are highly reproducible: Spearman ≥ {{reliability_min_spearman:.3f}}, both under a new batch shuffle and with a different scorer model.

Under both re-scorings, the ~2-word budget keeps the highest mean S, so the H3 conclusion (no interior optimum at H = 20) does not depend on the batching or on the scorer. With Llama-3.3-70B, ~10 words scores slightly above ~5 words; both remain below ~2 words.
