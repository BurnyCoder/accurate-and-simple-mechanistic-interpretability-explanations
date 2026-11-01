---
title: "Accurate and Simple: Length-Penalized Detection Scoring for Sparse Autoencoder Feature Explanations"
author: "Libor \"Burny\" Burian"
date: "2026"
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
# Latin Modern (pandoc's default xelatex font) has no glyphs for these; without the mapping they vanish from the PDF.
header-includes:
  - \usepackage{newunicodechar}
  - \newunicodechar{≥}{\ensuremath{\geq}}
  - \newunicodechar{≤}{\ensuremath{\leq}}
  - \newunicodechar{≈}{\ensuremath{\approx}}
  - \newunicodechar{∈}{\ensuremath{\in}}
# Figures stay where they are written: a figure left floating across a longtable made LaTeX drop
# the text after it ("Overfull \vbox while \output is active"), which lost the Discussion heading.
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

<!-- Generated from templates/paper/paper.md by `run.py --phase analyze --sample main`.
     Every number is filled from results/main/numbers.json. -->

# Abstract

Detection scoring ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)) is a widely used way to grade natural-language explanations of sparse autoencoder (SAE) latents, and SAEBench's autointerp metric is similar to it. It rewards accuracy only. We propose the smallest change that also rewards brevity, $S = \max(J, 0)\cdot 2^{-L/H}$: Youden's $J = 2\,\mathrm{BA} - 1$ from the detection scorer, times a factor that halves every $H$ tokens of explanation length $L$.

We test it with no training and minimal custom code:

- **Data:** Neuronpedia's public explanations and activations for {{n_latents}} randomly sampled latents of a GPT-2 small residual SAE, plus random negatives verified with the same SAE.
- **Code:** delphi's unmodified detection scorer and explainer, with Llama-3.1-70B-Instruct.

We pre-registered four hypotheses. Results:

- The scorer is valid. Random explanations score $J$ = {{H1.random.mean:.3f}}; real ones {{H1.gpt35.mean:.3f}}–{{H1.gpt4omini.mean:.3f}}.
- Asking the same explainer for ~2 to ~40 words raises $J$ from {{H2.j_by_budget.gen_b2.mean:.3f}} to {{H2.j_by_budget.gen_b40.mean:.3f}}, with diminishing returns.
- Our primary hypothesis, an interior optimum of $S$ at the pre-registered $H = 20$, is **falsified**. Mean $S$ decreases monotonically with length, so on average the shortest explanations win. In an exploratory, post-hoc sweep, ~10-word explanations only tie the shortest at $H = 40$; at $H = 80$ their mean is higher, but not significantly (one-sided Wilcoxon $p$ = {{exploratory.h_sweep.80.tests.best_minus_gen_b2.p_greater:.4f}}, uncorrected, for the budget picked as the best mean).
- Used to choose among 8 candidate explanations per latent, with cross-fitted held-out evaluation, $S$ selects explanations {{H4.length_reduction:.0%}} shorter than plain detection scoring does, at a cost of {{H4.j_loss_argmaxJ_minus_argmaxS.mean:.3f}} $J$ [{{H4.j_loss_argmaxJ_minus_argmaxS.lo:.3f}}, {{H4.j_loss_argmaxJ_minus_argmaxS.hi:.3f}}]. That is {{H4.argmaxS_minus_argminL.mean:.3f}} $J$ better than simply picking the shortest.

Under $S$, a top-token baseline (`the token "X"`) is not distinguishable from Neuronpedia's gpt-3.5-turbo or gpt-4o-mini explanations (post hoc; both paired 95% CIs include 0). Code, data, logs and pre-registration: <https://github.com/BurnyCoder/accurate-and-simple-mechanistic-interpretability-explanations>.

{{include:templates_static/paper_intro_method.md}}

**Data.**

- **SAE and latents.** We use Joseph Bloom's residual-stream SAE for GPT-2 small ([Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) at layer 6 (`blocks.6.hook_resid_pre`, 24,576 latents with a ReLU encoder, sae_lens's `standard` architecture), as exported by Neuronpedia. A latent is eligible if it has explanations from both gpt-3.5-turbo and gpt-4o-mini, at least 45 activating 17-token contexts and at least 25 zero-activation rows in the export, and is not one of delphi's few-shot latents. Alone, these filters keep {{data.filters.both_llm_explanations.pct:.1f}}%, {{data.filters.min_activating.pct:.1f}}% and {{data.filters.min_zero_rows.pct:.1f}}% of latents; together they leave {{data.n_eligible}} of {{data.n_latents_total}} ({{data.eligible_pct:.1f}}%). We sample {{n_latents}} at random (seed 0).
- **Contexts per latent.**
  - The 20 strongest contexts are reserved for the explainer.
  - 25 test positives are drawn, 5 from each of 5 equal-count activation-quantile bins of the remaining contexts. These are quantiles of the export's rows, which hold only fairly strong activations: every test positive reaches at least {{checks.activation_ratio.min:.0%}} of its latent's top exported activation (bin medians, strongest to weakest: {{checks.activation_ratio.median_by_bin.0:.2f}}, {{checks.activation_ratio.median_by_bin.1:.2f}}, {{checks.activation_ratio.median_by_bin.2:.2f}}, {{checks.activation_ratio.median_by_bin.3:.2f}}, {{checks.activation_ratio.median_by_bin.4:.2f}}). Paulo et al. sample deciles of the full activation distribution, so our detection task is easier and absolute $J$ is not comparable with theirs.
  - 25 negatives are drawn from {{data.negatives.pool}} random OpenWebText windows on which the latent's activation is exactly 0.
- **Verifying the negatives.** We compute those activations with [sae_lens](https://github.com/jbloomAus/SAELens) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) on GPT-2 small. Re-running {{data.negatives.reproduce_neuronpedia.n}} of Neuronpedia's top-activating windows (BOS + the 17 exported tokens) reproduces the exported maxima (Pearson r = {{data.negatives.reproduce_neuronpedia.pearson_max:.3f}}); the maximum falls on the same token in {{data.negatives.reproduce_neuronpedia.frac_argmax_same_token:.1%}} of them. Negatives are displayed in Neuronpedia's exact token format ({{data.negatives.token_format_roundtrip:.0%}} round-trip), so the format does not reveal the label. One content difference remains: {{checks.eot.pct_positives:.1f}}% of test positives contain GPT-2's document separator `<|endoftext|>` (Neuronpedia's windows can cross documents), against {{checks.eot.pct_negatives:.0f}}% of negatives (each is one window inside one document). The scorer does not treat it as a cue: given the random control's unrelated explanations, it says "yes" to {{checks.eot.random_positive_rate.listed:.1%}} of those positives and {{checks.eot.random_positive_rate.other:.1%}} of the others (§4.5 re-runs every hypothesis without them).
- **Why not Neuronpedia's own zero rows.** Neuronpedia's zero-activation rows were our pre-registered negatives. A pilot showed they are near-duplicate sliding windows: {{data.neuronpedia_zero_rows.rows}} rows but {{data.neuronpedia_zero_rows.distinct_texts}} distinct texts. Their replacement was recorded as a deviation before any main results, but after the pilot's own results had been computed (Appendix A).

**Conditions.**

- **Experiment 1:**
  - Neuronpedia's gpt-3.5-turbo and gpt-4o-mini explanations;
  - a top-token baseline (`the token "X"`, X the most frequent peak token among the train contexts);
  - a random control (another sampled latent's explanation).
- **Experiment 2:** Llama-3.1-70B-Instruct ([Llama Team 2024](https://arxiv.org/abs/2407.21783)) writes six explanations per latent with delphi's default explainer prompt, shown the 20 train contexts. One sentence is appended asking for approximately B ∈ {2, 5, 10, 20, 40} words, plus one unconstrained explanation.

**Scoring.** delphi `DetectionScorer`, Llama-3.1-70B-Instruct via OpenRouter, temperature 0, 5 contexts per prompt. Requests were pinned to the DeepInfra provider, which served all {{api_usage.score.calls}} main-run calls (fp8 according to [OpenRouter's endpoint list](https://openrouter.ai/api/v1/models/meta-llama/llama-3.1-70b-instruct/endpoints), checked 2026-10-31). All conditions of a latent are scored on the same 50 contexts in the same shuffled batches. The main run generated {{generated.n}} explanations and scored {{n_explanations}}, in {{api_usage.score.calls}} calls (US${{api_usage.score.cost_usd:.2f}}). {{sample_failure_rate:.3%}} of samples were lost to unparseable answers.

**Statistics.** Means over latents with 95% percentile bootstrap CIs ([Efron 1979](https://doi.org/10.1214/aos/1176344552); 10,000 resamples), paired one-sided Wilcoxon signed-rank tests ([Wilcoxon 1945](https://doi.org/10.2307/3001968)), and Holm correction within each hypothesis ([Holm 1979](https://www.jstor.org/stable/4615733)).

For H4, each latent's 50 test contexts are split into class-stratified halves. A candidate is selected on one half and its $J$ measured on the other, then the halves are swapped. Selection and evaluation therefore never share contexts (cross-fitting; [Chernozhukov et al. 2018](https://doi.org/10.1111/ectj.12097)).

# 4. Results

## 4.1 The detection scorer is valid (H1)

{{table:conditions}}

Table 1: All conditions. BA = balanced accuracy. 95% CIs in brackets.

The random control scores $J$ = {{H1.random.mean:.3f}} [{{H1.random.lo:.3f}}, {{H1.random.hi:.3f}}]. That is within the pre-registered $|J| < 0.05$, but its CI lies slightly below zero. Neuronpedia's explanations score well above chance. gpt-4o-mini's are nominally more accurate than gpt-3.5-turbo's, by {{exploratory.gpt4omini_minus_gpt35_J.mean:.3f}} [{{exploratory.gpt4omini_minus_gpt35_J.lo:.3f}}, {{exploratory.gpt4omini_minus_gpt35_J.hi:.3f}}]: the bootstrap CI excludes 0, but the two-sided Wilcoxon test gives $p$ = {{exploratory.gpt4omini_minus_gpt35_J.two_sided_p:.3f}} (exploratory). Both have a median of {{by_condition.gpt4omini.tokens_median:.0f}} tokens, and $S$ keeps their order ({{exploratory.gpt4omini_minus_gpt35_S.mean:+.3f}} [{{exploratory.gpt4omini_minus_gpt35_S.lo:+.3f}}, {{exploratory.gpt4omini_minus_gpt35_S.hi:+.3f}}]).

The top-token baseline (median {{by_condition.top_token.tokens_median:.0f}} tokens) reaches $J$ = {{by_condition.top_token.j_mean:.3f}}, below gpt-4o-mini ({{exploratory.top_token_vs_existing.minus_gpt4omini_J.mean:+.3f}} [{{exploratory.top_token_vs_existing.minus_gpt4omini_J.lo:+.3f}}, {{exploratory.top_token_vs_existing.minus_gpt4omini_J.hi:+.3f}}]) and not distinguishable from gpt-3.5-turbo ({{exploratory.top_token_vs_existing.minus_gpt35_J.mean:+.3f}} [{{exploratory.top_token_vs_existing.minus_gpt35_J.lo:+.3f}}, {{exploratory.top_token_vs_existing.minus_gpt35_J.hi:+.3f}}]). Under $S$ it scores {{by_condition.top_token.s_mean:.3f}}, not distinguishable from either: {{exploratory.top_token_vs_existing.minus_gpt4omini_S.mean:+.3f}} [{{exploratory.top_token_vs_existing.minus_gpt4omini_S.lo:+.3f}}, {{exploratory.top_token_vs_existing.minus_gpt4omini_S.hi:+.3f}}] against gpt-4o-mini and {{exploratory.top_token_vs_existing.minus_gpt35_S.mean:+.3f}} [{{exploratory.top_token_vs_existing.minus_gpt35_S.lo:+.3f}}, {{exploratory.top_token_vs_existing.minus_gpt35_S.hi:+.3f}}] against gpt-3.5-turbo (paired, post hoc).

![Mean $J$ (left) and $S$ at $H = 20$ (right) per explanation source, with 95% CIs.](../figures/fig1_conditions.png)

## 4.2 Accuracy rises with length, with diminishing returns (H2)

Asked for ~2, ~5, ~10, ~20 and ~40 words, the explainer produced median lengths of {{by_condition.gen_b2.tokens_median:.0f}}, {{by_condition.gen_b5.tokens_median:.0f}}, {{by_condition.gen_b10.tokens_median:.0f}}, {{by_condition.gen_b20.tokens_median:.0f}} and {{by_condition.gen_b40.tokens_median:.0f}} tokens. Mean $J$ was {{H2.j_by_budget.gen_b2.mean:.3f}}, {{H2.j_by_budget.gen_b5.mean:.3f}}, {{H2.j_by_budget.gen_b10.mean:.3f}}, {{H2.j_by_budget.gen_b20.mean:.3f}} and {{H2.j_by_budget.gen_b40.mean:.3f}}.

- **Rise:** $J(40) - J(2)$ = {{H2.rise_b40_minus_b2.mean:.3f}} [{{H2.rise_b40_minus_b2.lo:.3f}}, {{H2.rise_b40_minus_b2.hi:.3f}}], $p$ = {{H2.rise_b40_minus_b2.p_greater:.1e}}.
- **Diminishing returns:** the gain per word from 2→10 words exceeds that from 10→40 by {{H2.diminishing_gain_per_word.mean:.4f}} [{{H2.diminishing_gain_per_word.lo:.4f}}, {{H2.diminishing_gain_per_word.hi:.4f}}].

H2 is supported.

## 4.3 No interior optimum at $H = 20$ (H3, primary): falsified

{{table:h3}}

Table 2: Paired differences in $S$ ($H = 20$) between budgets.

Mean $S$ is {{H3.s_by_budget.gen_b2.mean:.3f}}, {{H3.s_by_budget.gen_b5.mean:.3f}}, {{H3.s_by_budget.gen_b10.mean:.3f}}, {{H3.s_by_budget.gen_b20.mean:.3f}} and {{H3.s_by_budget.gen_b40.mean:.3f}} for ~2 to ~40 words. Every intermediate budget beats ~40 words, but none beats ~2 words, so H3 is falsified.

The reason is quantitative. From ~2 to ~10 words mean $J$ grows by a factor of {{H3.illustration.j_ratio_b10_over_b2:.2f}}, while the length factor at the median lengths ({{by_condition.gen_b2.tokens_median:.0f}} → {{by_condition.gen_b10.tokens_median:.0f}} tokens) falls to {{H3.illustration.factor_ratio_b10_over_b2:.2f}} of its value. The product is below 1, so the extra accuracy does not pay for the extra tokens (an approximation: mean $S$ is not exactly mean $J$ times the factor at the median length).

In an exploratory, post-hoc sweep (Table 3, Fig. 4) the best budget is ~2 words for $H \leq 20$. At $H = 40$, ~10 words and ~2 words tie: {{exploratory.h_sweep.40.tests.best_minus_gen_b2.mean:+.3f}} [{{exploratory.h_sweep.40.tests.best_minus_gen_b2.lo:+.3f}}, {{exploratory.h_sweep.40.tests.best_minus_gen_b2.hi:+.3f}}]. At $H = 80$, ~10 words is ahead of ~2 words by {{exploratory.h_sweep.80.tests.best_minus_gen_b2.mean:.3f}} [{{exploratory.h_sweep.80.tests.best_minus_gen_b2.lo:.3f}}, {{exploratory.h_sweep.80.tests.best_minus_gen_b2.hi:.3f}}] in mean: the bootstrap CI excludes 0, but the one-sided Wilcoxon test does not reach 0.05 ($p$ = {{exploratory.h_sweep.80.tests.best_minus_gen_b2.p_greater:.4f}}). It is ahead of ~40 words by {{exploratory.h_sweep.80.tests.best_minus_gen_b40.mean:.3f}} [{{exploratory.h_sweep.80.tests.best_minus_gen_b40.lo:.3f}}, {{exploratory.h_sweep.80.tests.best_minus_gen_b40.hi:.3f}}] ($p$ = {{exploratory.h_sweep.80.tests.best_minus_gen_b40.p_greater:.1e}}). These comparisons were chosen after seeing the data, and the budget tested at each $H$ is the one with the highest mean, which favours it; they are not corrected for multiplicity.

Per latent, the picture is mixed (exploratory). At $H = 20$ the ~2-word explanation has the highest $S$ for {{exploratory.per_latent_best_budget.gen_b2}} of {{n_latents}} latents, an intermediate budget for {{exploratory.per_latent_best_intermediate}} (~5 words: {{exploratory.per_latent_best_budget.gen_b5}}, ~10: {{exploratory.per_latent_best_budget.gen_b10}}, ~20: {{exploratory.per_latent_best_budget.gen_b20}}) and ~40 words for {{exploratory.per_latent_best_budget.gen_b40}}. Single-explanation scores are noisy (§4.5), so part of this spread is measurement noise; H3 is a claim about the mean.

![Mean $J$ and $S$ ($H = 20$) against median explanation length in Experiment 2.](../figures/fig3_budget_curves.png)

{{table:h_sweep}}

Table 3: Mean $S$ per condition for each half-life $H$ (exploratory).

![Mean $S$ per length budget for each $H$.](../figures/fig4_h_sweep.png)

## 4.4 Selecting by $S$ gives short and accurate explanations (H4)

{{table:h4}}

Table 4: Cross-fitted selection among 8 candidates per latent.

- Selecting by $S$ instead of $J$ shortens the chosen explanation by {{H4.length_reduction:.0%}} (median {{H4.median_tokens.argmax_s:.1f}} vs {{H4.median_tokens.argmax_j:.1f}} tokens).
- It loses {{H4.j_loss_argmaxJ_minus_argmaxS.mean:.3f}} held-out $J$.
- It beats choosing the shortest candidate by {{H4.argmaxS_minus_argminL.mean:.3f}} [{{H4.argmaxS_minus_argminL.lo:.3f}}, {{H4.argmaxS_minus_argminL.hi:.3f}}].

All three pre-registered criteria hold. Selection by $S$ almost never picks the ~40-word or unconstrained explanations: {{H4.picked_counts.argmax_s.gen_b40}} and {{H4.picked_counts.argmax_s.gen_unconstrained}} picks out of {{H4.n_selections}}, against {{H4.picked_counts.argmax_j.gen_b40}} and {{H4.picked_counts.argmax_j.gen_unconstrained}} under $J$.

The pre-registered analysis uses one random half-split per latent (seed 0). Repeating it with {{exploratory.h4_split_sensitivity.n_splits}} different splits (exploratory), all three criteria hold in {{exploratory.h4_split_sensitivity.n_supported}} of them: the $J$ loss ranges from {{exploratory.h4_split_sensitivity.j_loss.min:.3f}} to {{exploratory.h4_split_sensitivity.j_loss.max:.3f}}, the length reduction from {{exploratory.h4_split_sensitivity.length_reduction.min:.0%}} to {{exploratory.h4_split_sensitivity.length_reduction.max:.0%}} (mean {{exploratory.h4_split_sensitivity.length_reduction.mean:.0%}}; the pre-registered split's {{H4.length_reduction:.0%}} is near the top of that range), and the margin over picking the shortest from {{exploratory.h4_split_sensitivity.s_minus_shortest.min:.3f}} to {{exploratory.h4_split_sensitivity.s_minus_shortest.max:.3f}}.

![Held-out $J$ and median length of the explanation chosen by each rule.](../figures/fig5_selection.png)

## 4.5 Reliability and robustness

{{table:reliability}}

Table 5: Agreement between two scorings of the same explanations. The two re-scorings (50 latents, all conditions) reuse the same 50 contexts, so they measure only the scorer's own noise. The split-half row compares each explanation's scores on the two disjoint halves of its contexts in the main run (random control excluded), which adds the noise from which contexts were drawn. *Within latent* = Spearman after subtracting each latent's mean, which removes the spread between latents that inflates the pooled values.

The scorer is consistent with itself: on the same contexts, a new batch order or a different scorer (Llama-3.3-70B) mostly preserves which explanation of a latent is better (within-latent Spearman of $J$: {{reliability_retest.spearman_j_within:.2f}} and {{reliability_altscorer.spearman_j_within:.2f}}). The choice of contexts matters more: across disjoint halves of 25 contexts, the within-latent Spearman of $J$ is {{reliability_split_half.spearman_j_within:.2f}}. Single-explanation scores, and so per-latent comparisons, are noisy. Our hypotheses are about means over {{n_latents}} latents, and their bootstrap CIs include this noise, because each latent's scores come from its own sampled contexts.

Dropping the {{checks.eot.pct_positives:.1f}}% of test positives that contain `<|endoftext|>` (§3) and re-running H1–H4 gives the same verdicts (supported: H1 {{checks.without_eot_positives.supported.H1}}, H2 {{checks.without_eot_positives.supported.H2}}, H3 {{checks.without_eot_positives.supported.H3}}, H4 {{checks.without_eot_positives.supported.H4}}). Mean $S$ at $H = 20$ for ~2, ~10 and ~40 words becomes {{checks.without_eot_positives.s_by_budget.gen_b2:.3f}}, {{checks.without_eot_positives.s_by_budget.gen_b10:.3f}} and {{checks.without_eot_positives.s_by_budget.gen_b40:.3f}}.

Measuring $L$ in words or characters instead of tokens leaves the ranking of explanations by $S$ almost unchanged: Spearman {{exploratory.unit_robustness_spearman.tokens_vs_words:.3f}} and {{exploratory.unit_robustness_spearman.tokens_vs_chars:.3f}}, with $H$ rescaled by the corpus words-per-token and characters-per-token ratios.

![Every scored explanation: $J$ against length; dashed curves have equal $S$ at $H = 20$.](../figures/fig2_j_vs_length.png)

# 5. Discussion

**What the length factor does well.** As a *selection* criterion over candidate explanations of the same latent, $S$ achieves the intended trade-off: much shorter explanations at a small, measured loss in accuracy. It is also robust:

- Unrelated explanations score near zero: the random control (median {{by_condition.random.tokens_median:.0f}} tokens) has $S$ = {{by_condition.random.s_mean:.3f}}, and by construction any explanation at or below chance gets $S = 0$, however short it is.
- Selecting by $S$ is far better than simply preferring brevity.

**What it does not do at $H = 20$.** Averaged over latents, the pre-registered penalty is harsh enough that the shortest explanations win. For GPT-2 layer-6 latents, a ~2-word label already reaches most of the $J$ that ~40 words reach: {{H2.j_by_budget.gen_b2.mean:.3f}} vs {{H2.j_by_budget.gen_b40.mean:.3f}}. Two readings fit this result; we did not test which one holds:

- many of these latents are close to single-token detectors, in line with the strong top-token baseline;
- detection on 17-token contexts cannot reward nuance beyond what a short label carries.

**Choosing $H$.** $H$ is the exchange rate between accuracy and brevity, the price in tokens of halving the score. It is a value judgement and should be set for the use case. In our post-hoc sweep, ~2-word labels win for $H \leq 20$, tie with ~10-word explanations at $H = 40$, and trail them in mean at $H = 80$, though not significantly. Because $\log_2 S = \log_2 J - L/H$, $H$ also has an MDL reading: each token costs $1/H$ bits of $\log_2 J$.

# 6. Limitations

- **One SAE and one scorer family.** GPT-2 small, layer 6, with Llama 70B scorers. The reliability checks cover a second scorer on 50 latents only, and OpenRouter routed that scorer's calls across several providers (see the reliability report).
- **Short contexts and a small test set.** Contexts are 17 tokens, and each explanation is scored on 50 contexts rather than Paulo et al.'s 100 + 100 (§4.2). Single-explanation scores are therefore noisy: across disjoint halves of the contexts, the within-latent Spearman of $J$ is only {{reliability_split_half.spearman_j_within:.2f}} (Table 5). Our claims are about means over {{n_latents}} latents.
- **Strong activations only.** Every test positive reaches at least {{checks.activation_ratio.min:.0%}} of its latent's top exported activation (§3). Weak activations, which Paulo et al.'s decile sampling includes, are not tested, so absolute $J$ overstates how well explanations cover a latent's full range.
- **Possible overlap.** Neuronpedia's explanations were written from top activations. Holding out the 20 strongest contexts reduces but does not remove overlap with our test positives. The Exp 2 explainer saw only those 20 train contexts, none of which is a test context, but {{checks.train_overlap.n}} of {{checks.n_test_positives}} test positives ({{checks.train_overlap.pct:.1f}}%, in {{checks.train_overlap.n_latents}} latents) share a run of at least {{checks.train_overlap.min_shared_tokens}} consecutive tokens with a train context, i.e. come from the same passage.
- **Positives and negatives come from different sources.** Positives are Neuronpedia's windows and negatives are our OpenWebText windows. One known difference: {{checks.eot.pct_positives:.1f}}% of positives contain `<|endoftext|>`, against {{checks.eot.pct_negatives:.0f}}% of negatives. The scorer does not use it as a cue, and the verdicts hold without those positives (§4.5), but other, unmeasured differences between the two sources may remain.
- **The same model explains and scores.** Llama-3.1-70B writes the Exp 2 explanations and also scores every explanation, and it may favour its own phrasing. Comparisons between Exp 2 and Neuronpedia's explanations (Table 1, and H4's candidate pool) are confounded with this, and with the explainers having seen different contexts. H2 and H3 compare Exp 2 explanations with each other and are not affected.
- **Sample bias.** The eligibility filters keep {{data.eligible_pct:.1f}}% of latents. The ≥ 45 activating contexts filter alone keeps {{data.filters.min_activating.pct:.1f}}% (the pre-registration estimated about 81%) and likely favours denser latents; the ≥ 25 zero-row filter alone keeps {{data.filters.min_zero_rows.pct:.1f}}%, and we did not study what it selects for.
- **Tokenizer dependence.** $L$ depends on the tokenizer, but rankings barely change with words or characters.
- **Untested properties.** We did not test whether shorter explanations are more *useful* to humans or downstream tools. We assume it, as the premise of the project.

# 7. Conclusion

Multiplying the above-chance detection score by $2^{-L/H}$ is a one-line, library-compatible way to make SAE explanation scores reward brevity. Used to choose explanations, it gives much shorter ones at a small cost in accuracy. The pre-registered strength $H = 20$ is, however, strong enough that on average the shortest explanations win, so $H$ should be chosen deliberately; in our post-hoc sweep, ~10-word explanations only draw level at $H = 40$, and at $H = 80$ their higher mean is not significant.

{{include:templates_static/paper_related.md}}

# Appendix A: Pre-registration and deviations

Hypotheses, metrics, tests and falsification criteria were committed in `PREREG.md` before any results. Two deviations were made before any main-sample results.

1. The Exp 2 budget sentence asks for "approximately B words" instead of "at most B words", because delphi's explainer prompt already demands concision.
2. The negatives changed from Neuronpedia's zero rows to verified random OpenWebText windows, after the 10-latent debugging pilot showed the zero rows were near-duplicates.

The second change was not blind to the pilot's results. The first pilot run, with the old negatives, had already been scored and analysed. Its random control scored $J$ = {{pilot_discarded.H1.random.mean:+.3f}}, and the pre-registered verdicts on its 10 latents were H1 {{pilot_discarded.H1.supported}}, H2 {{pilot_discarded.H2.supported}}, H3 {{pilot_discarded.H3.supported}} and H4 {{pilot_discarded.H4.supported}}. The re-run with the new negatives gave $J$ = {{pilot.H1.random.mean:+.3f}} for the random control and H1 {{pilot.H1.supported}}, H2 {{pilot.H2.supported}}, H3 {{pilot.H3.supported}}, H4 {{pilot.H4.supported}}. The change was prompted by the duplicate negatives in the logged prompts, and it follows delphi's default and Paulo et al.'s random negatives. A random control above chance is itself a sign of flawed negatives. Both pilot runs are kept in `results/`.

The analysis plan was not changed. The $H$ sweep tests are labelled exploratory.

# Appendix B: Prompts

- **Scorer:** delphi's detection prompt (`delphi/scorers/classifier/prompts/detection_prompt.py`, commit `4fea06e`). It has a system prompt, three few-shot examples and the test batch. Two of the few-shot examples are latents 6048 and 9396 of this SAE, which we exclude; the third is latent 12654 of the layer-8 SAE (`8-res-jb`).
- **Explainer:** delphi's `DefaultExplainer` prompt (`delphi/explainers/default/prompts.py`), with "Your final explanation (the [EXPLANATION]: line) must be approximately B words long." appended to the last user message.

Every prompt and response is in the run logs (published with the dataset).

# AI statement

This codebase and paper were written by Claude Code (Anthropic's Claude, model Claude Opus 5.5) from an exact specification provided by Libor "Burny" Burian, who directed the work. Human cleanup and rewriting are in progress.
