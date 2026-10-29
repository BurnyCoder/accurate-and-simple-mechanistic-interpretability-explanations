# Experiment 2: does the length factor reward accurate *and* short explanations?

<!-- Generated from templates/reports/exp2_length_budgets.md; numbers come from results/main/numbers.json. -->

**Question.** Hold the latent fixed and vary only how long the explanation is. Does accuracy rise with length (H2)? Does S peak at an intermediate length at the pre-registered H = 20 (H3, primary)? Does choosing explanations by S, instead of by plain J, give much shorter explanations at a small cost in accuracy (H4)?

## Setup

- **Explainer:** Llama-3.1-70B-Instruct (OpenRouter, DeepInfra, temperature 0) with delphi's `DefaultExplainer` prompt: its system prompt, 3 few-shot examples, and the latent's 20 strongest contexts with `<<highlighted>>` tokens and activation values.
- **Budgets:** one sentence was appended: "Your final explanation (the [EXPLANATION]: line) must be approximately B words long", for B ∈ {2, 5, 10, 20, 40}. One unconstrained explanation used delphi's prompt unchanged.
- **Volume:** 6 explanations × {{n_latents}} latents, all parsed.
- **Test set:** the test contexts are disjoint from the 20 the explainer saw, so these explanations are fully held out.
- **Why these budgets:** they span the range of existing explanations (Neuronpedia's median is {{by_condition.gpt4omini.tokens_median:.0f}} tokens) up to delphi's unconstrained output ({{by_condition.gen_unconstrained.tokens_median:.0f}} tokens median). They are spaced roughly geometrically.

**Manipulation check (median tokens):**

| ~2 words | ~5 | ~10 | ~20 | ~40 | unconstrained |
|---:|---:|---:|---:|---:|---:|
| {{by_condition.gen_b2.tokens_median:.0f}} | {{by_condition.gen_b5.tokens_median:.0f}} | {{by_condition.gen_b10.tokens_median:.0f}} | {{by_condition.gen_b20.tokens_median:.0f}} | {{by_condition.gen_b40.tokens_median:.0f}} | {{by_condition.gen_unconstrained.tokens_median:.0f}} |

The explainer followed the budgets (about {{exploratory.unit_robustness_spearman.words_per_token:.2f}} words per token).

## H2: accuracy rises with length, with diminishing returns ({{H2.supported}})

| | ~2 words | ~5 | ~10 | ~20 | ~40 | unconstrained |
|---|---:|---:|---:|---:|---:|---:|
| mean J | {{H2.j_by_budget.gen_b2.mean:.3f}} | {{H2.j_by_budget.gen_b5.mean:.3f}} | {{H2.j_by_budget.gen_b10.mean:.3f}} | {{H2.j_by_budget.gen_b20.mean:.3f}} | {{H2.j_by_budget.gen_b40.mean:.3f}} | {{H2.j_by_budget.gen_unconstrained.mean:.3f}} |

- J(~40) − J(~2) = {{H2.rise_b40_minus_b2.mean:+.3f}} [{{H2.rise_b40_minus_b2.lo:+.3f}}, {{H2.rise_b40_minus_b2.hi:+.3f}}], Wilcoxon p = {{H2.rise_b40_minus_b2.p_greater:.1e}}.
- The J gain per nominal word from 2→10 words, minus the gain per word from 10→40, is {{H2.diminishing_gain_per_word.mean:+.4f}} [{{H2.diminishing_gain_per_word.lo:+.4f}}, {{H2.diminishing_gain_per_word.hi:+.4f}}] > 0. That is diminishing returns.
- Across all {{n_explanations}} scored explanations, J and length are only weakly related (Spearman {{exploratory.spearman_j_vs_tokens_all:.2f}}).

## H3 (primary): an interior optimum at H = 20? **{{H3.supported}} — falsified**

| | ~2 words | ~5 | ~10 | ~20 | ~40 | unconstrained |
|---|---:|---:|---:|---:|---:|---:|
| mean S (H=20) | {{H3.s_by_budget.gen_b2.mean:.3f}} | {{H3.s_by_budget.gen_b5.mean:.3f}} | {{H3.s_by_budget.gen_b10.mean:.3f}} | {{H3.s_by_budget.gen_b20.mean:.3f}} | {{H3.s_by_budget.gen_b40.mean:.3f}} | {{H3.s_by_budget.gen_unconstrained.mean:.3f}} |

{{table:h3}}

Every intermediate budget beats ~40 words, but none beats ~2 words. On average, the accuracy gained by writing more is smaller than the length cost at H = 20.

From ~2 to ~10 words, mean J rises by a factor of {{H3.illustration.j_ratio_b10_over_b2:.2f}}. Over the same stretch, the length factor at the median lengths ({{by_condition.gen_b2.tokens_median:.0f}} → {{by_condition.gen_b10.tokens_median:.0f}} tokens) falls by a factor of {{H3.illustration.factor_ratio_b10_over_b2:.2f}}. The pre-registered prediction is therefore falsified.

![J and S vs length](../figures/fig3_budget_curves.png)

### Exploratory (post hoc): the H sweep

{{table:h_sweep}}

At H = 40 the best budget is {{exploratory.h_sweep.40.best_budget}}:

- vs ~2 words: {{exploratory.h_sweep.40.tests.best_minus_gen_b2.mean:+.3f}} [{{exploratory.h_sweep.40.tests.best_minus_gen_b2.lo:+.3f}}, {{exploratory.h_sweep.40.tests.best_minus_gen_b2.hi:+.3f}}];
- vs ~40 words: {{exploratory.h_sweep.40.tests.best_minus_gen_b40.mean:+.3f}} [{{exploratory.h_sweep.40.tests.best_minus_gen_b40.lo:+.3f}}, {{exploratory.h_sweep.40.tests.best_minus_gen_b40.hi:+.3f}}].

At H = 80 the best budget is {{exploratory.h_sweep.80.best_budget}}:

- vs ~2 words: {{exploratory.h_sweep.80.tests.best_minus_gen_b2.mean:+.3f}} [{{exploratory.h_sweep.80.tests.best_minus_gen_b2.lo:+.3f}}, {{exploratory.h_sweep.80.tests.best_minus_gen_b2.hi:+.3f}}];
- vs ~40 words: {{exploratory.h_sweep.80.tests.best_minus_gen_b40.mean:+.3f}} [{{exploratory.h_sweep.80.tests.best_minus_gen_b40.lo:+.3f}}, {{exploratory.h_sweep.80.tests.best_minus_gen_b40.hi:+.3f}}].

These comparisons were chosen after seeing the data, so they are hypotheses for a future pre-registered test, not confirmed findings.

![H sweep](../figures/fig4_h_sweep.png)

## H4: choosing explanations by S ({{H4.supported}})

- **Candidates:** for each latent, the 6 Exp 2 explanations plus Neuronpedia's gpt-3.5-turbo and gpt-4o-mini explanations.
- **Cross-fitting:** the test contexts are split into halves, stratified by class. We choose on one half, measure J on the other, then swap. This avoids rewarding explanations that won by noise (the winner's curse).

{{table:h4}}

- **Length:** argmax-S picks are {{H4.length_reduction:.0%}} shorter than argmax-J picks (criterion: ≥ 20%).
- **Accuracy loss:** J drops by {{H4.j_loss_argmaxJ_minus_argmaxS.mean:.3f}} [{{H4.j_loss_argmaxJ_minus_argmaxS.lo:.3f}}, {{H4.j_loss_argmaxJ_minus_argmaxS.hi:.3f}}] (criterion: ≤ 0.05).
- **Versus shortest:** argmax-S beats argmin-L by {{H4.argmaxS_minus_argminL.mean:.3f}} [{{H4.argmaxS_minus_argminL.lo:.3f}}, {{H4.argmaxS_minus_argminL.hi:.3f}}] J (criterion: CI > 0).

All three pre-registered criteria hold. Which candidates got picked (counts over {{H4.n_selections}} selections: 2 halves × {{n_latents}} latents):

- argmax J: ~2 words {{H4.picked_counts.argmax_j.gen_b2}}, ~5 {{H4.picked_counts.argmax_j.gen_b5}}, ~10 {{H4.picked_counts.argmax_j.gen_b10}}, ~20 {{H4.picked_counts.argmax_j.gen_b20}}, ~40 {{H4.picked_counts.argmax_j.gen_b40}}, unconstrained {{H4.picked_counts.argmax_j.gen_unconstrained}}, gpt-3.5 {{H4.picked_counts.argmax_j.gpt35}}, gpt-4o-mini {{H4.picked_counts.argmax_j.gpt4omini}}.
- argmax S: ~2 words {{H4.picked_counts.argmax_s.gen_b2}}, ~5 {{H4.picked_counts.argmax_s.gen_b5}}, ~10 {{H4.picked_counts.argmax_s.gen_b10}}, ~20 {{H4.picked_counts.argmax_s.gen_b20}}, ~40 {{H4.picked_counts.argmax_s.gen_b40}}, unconstrained {{H4.picked_counts.argmax_s.gen_unconstrained}}, gpt-3.5 {{H4.picked_counts.argmax_s.gpt35}}, gpt-4o-mini {{H4.picked_counts.argmax_s.gpt4omini}}.

Plain J often picks short explanations too, because ties are broken toward the shorter one and J is noisy on 25 contexts. Even so, S moves picks away from the long budgets almost completely.

![Selection](../figures/fig5_selection.png)

## Takeaways

1. The length factor does what it is designed to do at the level of choosing an explanation per latent: much shorter picks for little accuracy (H4).
2. Averaged over latents, the pre-registered H = 20 is too harsh to make anything but the shortest budget optimal (H3 falsified). H is a value judgement. In the post-hoc sweep, ~10 words ties ~2 words at H = 40 and wins at H = 80.
3. A 2-word explanation from a strong explainer already captures most of the detectable signal for these GPT-2 layer-6 latents: J {{H2.j_by_budget.gen_b2.mean:.3f}} vs {{H2.j_by_budget.gen_b40.mean:.3f}} at ~40 words.
