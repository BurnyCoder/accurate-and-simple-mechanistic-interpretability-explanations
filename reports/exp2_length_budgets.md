# Experiment 2: does the length factor reward accurate *and* short explanations?

<!-- Generated from templates/reports/exp2_length_budgets.md; numbers come from results/main/numbers.json. -->

**Question.** Hold the latent fixed and vary mainly the requested length of the explanation. Does accuracy rise with length (H2)? Does S peak at an intermediate length at the pre-registered H = 20 (H3, primary)? Does choosing explanations by S, instead of by plain J, give much shorter explanations at a small cost in accuracy (H4)?

## Setup

- **Explainer:** Llama-3.1-70B-Instruct (OpenRouter, DeepInfra, temperature 0) with delphi's `DefaultExplainer` prompt: its system prompt, 3 few-shot examples, and the latent's 20 strongest contexts with `<<highlighted>>` tokens and activation values.
- **Budgets:** one sentence was appended: "Your final explanation (the [EXPLANATION]: line) must be approximately B words long", for B ∈ {2, 5, 10, 20, 40}. One unconstrained explanation used delphi's prompt unchanged.
- **Volume:** 1200 explanations (6 per latent × 200 latents), of which 1200 parsed.
- **Test set:** the test contexts are disjoint from the 20 the explainer saw, so these explanations are fully held out.
- **Budget choice:** pre-registered (PREREG section 5), spaced roughly geometrically from 2 to 40 words. The observed medians run from 2 to 45 tokens, which brackets Neuronpedia's explanations (median 8 tokens) and delphi's unconstrained output (median 30 tokens).

**Manipulation check (median tokens):**

| ~2 words | ~5 | ~10 | ~20 | ~40 | unconstrained |
|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 11 | 20 | 45 | 30 |

The median length rises with every budget step, so the explainer followed the budgets roughly. (For scale: across all 2002 scored explanations there are 0.77 words per o200k token.)

## H2: accuracy rises with length, with diminishing returns (True)

| | ~2 words | ~5 | ~10 | ~20 | ~40 | unconstrained |
|---|---:|---:|---:|---:|---:|---:|
| mean J | 0.464 | 0.493 | 0.548 | 0.552 | 0.577 | 0.563 |

- J(~40) − J(~2) = +0.113 [+0.079, +0.147], Wilcoxon p = 7.1e-09.
- The J gain per nominal word from 2→10 words, minus the gain per word from 10→40, is +0.0095 [+0.0056, +0.0136] > 0. That is diminishing returns.
- Across all 2002 scored explanations, J and length are only weakly related (Spearman 0.13).

## H3 (primary): an interior optimum at H = 20? **False — falsified**

| | ~2 words | ~5 | ~10 | ~20 | ~40 | unconstrained |
|---|---:|---:|---:|---:|---:|---:|
| mean S (H=20) | 0.430 | 0.395 | 0.374 | 0.278 | 0.123 | 0.204 |

| comparison (S at H=20) | mean difference [95% CI] | Wilcoxon p (one-sided) | Holm p |
|---|---:|---:|---:|
| ~5 words − ~2 words | -0.035 [-0.064, -0.006] | 0.99 | 1 |
| ~5 words − ~40 words | +0.272 [+0.246, +0.297] | 6.6e-33 | 3.3e-32 |
| ~10 words − ~2 words | -0.056 [-0.081, -0.031] | 1 | 1 |
| ~10 words − ~40 words | +0.251 [+0.230, +0.271] | 1.9e-33 | 1.1e-32 |
| ~20 words − ~2 words | -0.152 [-0.181, -0.123] | 1 | 1 |
| ~20 words − ~40 words | +0.155 [+0.141, +0.169] | 6.8e-33 | 3.3e-32 |

Every intermediate budget beats ~40 words, but none beats ~2 words. On average, the accuracy gained by writing more is smaller than the length cost at H = 20.

From ~2 to ~10 words, mean J rises by a factor of 1.18. Over the same stretch, the length factor at the median lengths (2 → 11 tokens) falls by a factor of 0.73. The pre-registered prediction is therefore falsified.

![J and S vs length](../figures/fig3_budget_curves.png)

### Exploratory (post hoc): the H sweep

| H (tokens) | random-explanation control | top-token baseline | gpt-3.5-turbo (Neuronpedia) | gpt-4o-mini (Neuronpedia) | Llama-3.1-70B, ~2 words | Llama-3.1-70B, ~5 words | Llama-3.1-70B, ~10 words | Llama-3.1-70B, ~20 words | Llama-3.1-70B, ~40 words | Llama-3.1-70B, delphi default prompt | best budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.009 | 0.216 | 0.161 | 0.163 | 0.332 | 0.203 | 0.119 | 0.037 | 0.001 | 0.014 | ~2 words |
| 10 | 0.016 | 0.306 | 0.262 | 0.280 | 0.394 | 0.316 | 0.255 | 0.140 | 0.027 | 0.079 | ~2 words |
| 20 | 0.021 | 0.365 | 0.341 | 0.374 | 0.430 | 0.395 | 0.374 | 0.278 | 0.123 | 0.204 | ~2 words |
| 40 | 0.025 | 0.398 | 0.391 | 0.434 | 0.449 | 0.442 | 0.454 | 0.392 | 0.266 | 0.337 | ~10 words |
| 80 | 0.027 | 0.416 | 0.419 | 0.468 | 0.459 | 0.467 | 0.500 | 0.467 | 0.392 | 0.435 | ~10 words |
| ∞ (no length factor: max(J, 0)) | 0.029 | 0.435 | 0.450 | 0.505 | 0.469 | 0.494 | 0.551 | 0.556 | 0.578 | 0.565 | ~40 words |

At H = 40 the budget with the highest mean S is ~10 words:

- vs ~2 words: +0.005 [-0.022, +0.032], one-sided Wilcoxon p = 0.80 (a tie);
- vs ~40 words: +0.188 [+0.167, +0.208], p = 6.1e-29.

At H = 80 it is ~10 words:

- vs ~2 words: +0.041 [+0.013, +0.069], p = 0.050 (the bootstrap CI excludes 0, the Wilcoxon test is borderline);
- vs ~40 words: +0.108 [+0.087, +0.128], p = 4.8e-18.

These comparisons were chosen after seeing the data and are not corrected for multiplicity, so they are hypotheses for a future pre-registered test, not confirmed findings.

![H sweep](../figures/fig4_h_sweep.png)

## H4: choosing explanations by S (True)

- **Candidates:** for each latent, the 6 Exp 2 explanations plus Neuronpedia's gpt-3.5-turbo and gpt-4o-mini explanations.
- **Cross-fitting:** the test contexts are split into halves, stratified by class. We choose on one half, measure J on the other, then swap. This avoids rewarding explanations that won by noise (the winner's curse).

| selection rule | held-out mean J [95% CI] | median tokens |
|---|---:|---:|
| pick highest J (plain detection) | 0.644 [0.605, 0.682] | 9.0 |
| pick highest S (this method) | 0.613 [0.574, 0.649] | 5.5 |
| pick shortest | 0.467 [0.425, 0.509] | 2.0 |

- **Length:** argmax-S picks are 39% shorter than argmax-J picks (criterion: ≥ 20%).
- **Accuracy loss:** J drops by 0.031 [0.018, 0.046] (criterion: ≤ 0.05).
- **Versus shortest:** argmax-S beats argmin-L by 0.146 [0.115, 0.179] J (criterion: CI > 0).

All three pre-registered criteria hold. Which candidates got picked (counts over 400 selections: 2 halves × 200 latents):

- argmax J: ~2 words 80, ~5 51, ~10 43, ~20 32, ~40 28, unconstrained 29, gpt-3.5 63, gpt-4o-mini 74.
- argmax S: ~2 words 145, ~5 78, ~10 27, ~20 8, ~40 2, unconstrained 1, gpt-3.5 67, gpt-4o-mini 72.

Plain J often picks short explanations too, likely because ties are broken toward the shorter one and J is noisy on 25 contexts (we did not measure how often ties occur). Even so, S moves picks away from the long budgets almost completely.

![Selection](../figures/fig5_selection.png)

## Takeaways

1. The length factor does what it is designed to do at the level of choosing an explanation per latent: much shorter picks for little accuracy (H4).
2. Averaged over latents, the pre-registered H = 20 is too harsh to make anything but the shortest budget optimal (H3 falsified). H is a value judgement. In the post-hoc sweep, ~10 words ties ~2 words at H = 40 and is nominally ahead at H = 80 (one-sided p = 0.050).
3. For these GPT-2 layer-6 latents, a ~2-word explanation from a strong explainer already reaches most of the J of a ~40-word one: 0.464 vs 0.577.
