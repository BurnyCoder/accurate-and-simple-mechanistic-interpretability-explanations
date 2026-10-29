---
title: "Accurate and Simple: Length-Penalized Detection Scoring for Sparse Autoencoder Feature Explanations"
author: "Libor \"Burny\" Burian"
date: "2026"
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
---

<!-- Generated from templates/paper/paper.md by `run.py --phase analyze --sample main`.
     Every number is filled from results/main/numbers.json. -->

# Abstract

Detection scoring is the standard way to grade natural-language explanations of sparse autoencoder (SAE) latents ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928); SAEBench). It rewards accuracy only. We propose the smallest change that also rewards brevity, $S = \max(J, 0)\cdot 2^{-L/H}$: Youden's $J = 2\,\mathrm{BA} - 1$ from the detection scorer, times a factor that halves every $H$ tokens of explanation length $L$.

We test it with no training and minimal custom code:

- **Data:** Neuronpedia's public explanations and activations for {{n_latents}} randomly sampled latents of a GPT-2 small residual SAE, plus random negatives verified with the same SAE.
- **Code:** delphi's unmodified detection scorer and explainer, with Llama-3.1-70B-Instruct.

We pre-registered four hypotheses. Results:

- The scorer is valid. Random explanations score $J$ = {{H1.random.mean:.3f}}; real ones {{H1.gpt35.mean:.3f}}–{{H1.gpt4omini.mean:.3f}}.
- Asking the same explainer for ~2 to ~40 words raises $J$ from {{H2.j_by_budget.gen_b2.mean:.3f}} to {{H2.j_by_budget.gen_b40.mean:.3f}}, with diminishing returns.
- Our primary hypothesis, an interior optimum of $S$ at the pre-registered $H = 20$, is **falsified**. Mean $S$ decreases monotonically with length, so on average the shortest explanations win. In an exploratory sweep, ~10-word explanations win only at a much weaker penalty ($H = 80$).
- Used to choose among 8 candidate explanations per latent, with cross-fitted held-out evaluation, $S$ selects explanations {{H4.length_reduction:.0%}} shorter than plain detection scoring does, at a cost of {{H4.j_loss_argmaxJ_minus_argmaxS.mean:.3f}} $J$ [{{H4.j_loss_argmaxJ_minus_argmaxS.lo:.3f}}, {{H4.j_loss_argmaxJ_minus_argmaxS.hi:.3f}}]. That is {{H4.argmaxS_minus_argminL.mean:.3f}} $J$ better than simply picking the shortest.

A one-token baseline (`the token "X"`) is as good under $S$ as gpt-4o-mini's explanations. Code, data, logs and pre-registration: <https://github.com/BurnyCoder/accurate-and-simple-mechanistic-interpretability-explanations>.

{{include:templates_static/paper_intro_method.md}}

**Data.**

- **SAE and latents.** We use Joseph Bloom's residual-stream SAE for GPT-2 small at layer 6 (`blocks.6.hook_resid_pre`, 24,576 ReLU latents), as exported by Neuronpedia. A latent is eligible if it has explanations from both gpt-3.5-turbo and gpt-4o-mini, at least 45 activating 17-token contexts, and is not one of delphi's few-shot latents. That leaves {{data.n_eligible}} of {{data.n_latents_total}} ({{data.eligible_pct:.1f}}%). We sample {{n_latents}} at random (seed 0).
- **Contexts per latent.**
  - The 20 strongest contexts are reserved for the explainer.
  - 25 test positives are drawn, 5 from each of 5 equal-count activation-quantile bins of the remaining contexts.
  - 25 negatives are drawn from {{data.negatives.pool}} random OpenWebText windows on which the latent's activation is exactly 0.
- **Verifying the negatives.** We compute those activations with sae_lens and TransformerLens on GPT-2 small. Re-running Neuronpedia's top windows reproduces its exported maxima (Pearson r = {{data.negatives.reproduce_neuronpedia.pearson_max:.3f}}, n = {{data.negatives.reproduce_neuronpedia.n}}). Negatives are displayed in Neuronpedia's exact token format ({{data.negatives.token_format_roundtrip:.0%}} round-trip), so the format cannot reveal the label.
- **Why not Neuronpedia's own zero rows.** Neuronpedia's zero-activation rows were our pre-registered negatives. A pilot showed they are near-duplicate sliding windows: {{data.neuronpedia_zero_rows.rows}} rows but {{data.neuronpedia_zero_rows.distinct_texts}} distinct texts. Their replacement was recorded as a deviation before any main results (Appendix A).

**Conditions.**

- **Experiment 1:**
  - Neuronpedia's gpt-3.5-turbo and gpt-4o-mini explanations;
  - a top-token baseline (`the token "X"`, X the most frequent peak token among the train contexts);
  - a random control (another sampled latent's explanation).
- **Experiment 2:** Llama-3.1-70B-Instruct writes six explanations per latent with delphi's default explainer prompt, shown the 20 train contexts. One sentence is appended asking for approximately B ∈ {2, 5, 10, 20, 40} words, plus one unconstrained explanation.

**Scoring.** delphi `DetectionScorer`, Llama-3.1-70B-Instruct via OpenRouter (pinned to DeepInfra fp8), temperature 0, 5 contexts per prompt. All conditions of a latent are scored on the same 50 contexts in the same shuffled batches. {{n_explanations}} explanations were scored with {{api_usage.score.calls}} calls (US${{api_usage.score.cost_usd:.2f}}). {{sample_failure_rate:.3%}} of samples were lost to unparseable answers.

**Statistics.** Means over latents with 95% percentile bootstrap CIs (10,000 resamples), paired one-sided Wilcoxon signed-rank tests, and Holm correction within each hypothesis.

For H4, each latent's 50 test contexts are split into class-stratified halves. A candidate is selected on one half and its $J$ measured on the other, then the halves are swapped. Selection and evaluation therefore never share contexts (cross-fitting).

# 4. Results

## 4.1 The detection scorer is valid (H1)

{{table:conditions}}

Table 1: All conditions. BA = balanced accuracy. 95% CIs in brackets.

The random control scores $J$ = {{H1.random.mean:.3f}} [{{H1.random.lo:.3f}}, {{H1.random.hi:.3f}}]. That is within the pre-registered $|J| < 0.05$, but its CI lies slightly below zero. Neuronpedia's explanations score well above chance. gpt-4o-mini's are more accurate than gpt-3.5-turbo's by {{exploratory.gpt4omini_minus_gpt35_J.mean:.3f}} [{{exploratory.gpt4omini_minus_gpt35_J.lo:.3f}}, {{exploratory.gpt4omini_minus_gpt35_J.hi:.3f}}] (two-sided $p$ = {{exploratory.gpt4omini_minus_gpt35_J.two_sided_p:.3f}}, exploratory). Both have a median of {{by_condition.gpt4omini.tokens_median:.0f}} tokens, so $S$ keeps their order.

The one-token baseline reaches $J$ = {{by_condition.top_token.j_mean:.3f}}. Under $S$ it scores {{by_condition.top_token.s_mean:.3f}}, level with gpt-4o-mini ({{by_condition.gpt4omini.s_mean:.3f}}) and above gpt-3.5-turbo ({{by_condition.gpt35.s_mean:.3f}}).

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

The reason is quantitative. From ~2 to ~10 words mean $J$ grows by a factor of {{H3.illustration.j_ratio_b10_over_b2:.2f}}, while the length factor shrinks by {{H3.illustration.factor_ratio_b10_over_b2:.2f}}.

In an exploratory, post-hoc sweep (Table 3, Fig. 4) the best budget is ~2 words for $H \leq 20$. At $H = 40$, ~10 words and ~2 words tie: {{exploratory.h_sweep.40.tests.best_minus_gen_b2.mean:+.3f}} [{{exploratory.h_sweep.40.tests.best_minus_gen_b2.lo:+.3f}}, {{exploratory.h_sweep.40.tests.best_minus_gen_b2.hi:+.3f}}]. At $H = 80$, ~10 words beats ~2 words by {{exploratory.h_sweep.80.tests.best_minus_gen_b2.mean:.3f}} [{{exploratory.h_sweep.80.tests.best_minus_gen_b2.lo:.3f}}, {{exploratory.h_sweep.80.tests.best_minus_gen_b2.hi:.3f}}] and ~40 words by {{exploratory.h_sweep.80.tests.best_minus_gen_b40.mean:.3f}} [{{exploratory.h_sweep.80.tests.best_minus_gen_b40.lo:.3f}}, {{exploratory.h_sweep.80.tests.best_minus_gen_b40.hi:.3f}}]. These comparisons were chosen after seeing the data.

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

![Held-out $J$ and median length of the explanation chosen by each rule.](../figures/fig5_selection.png)

## 4.5 Reliability and robustness

{{table:reliability}}

Table 5: Re-scoring 50 latents (all conditions).

Measuring $L$ in words or characters instead of tokens leaves the ranking of explanations by $S$ almost unchanged: Spearman {{exploratory.unit_robustness_spearman.tokens_vs_words:.3f}} and {{exploratory.unit_robustness_spearman.tokens_vs_chars:.3f}}, with $H$ rescaled by the corpus words-per-token and characters-per-token ratios.

![Every scored explanation: $J$ against length; dashed curves have equal $S$.](../figures/fig2_j_vs_length.png)

# 5. Discussion

**What the length factor does well.** As a *selection* criterion over candidate explanations of the same latent, $S$ achieves the intended trade-off: much shorter explanations at a small, measured loss in accuracy. It is also robust:

- Short-but-wrong explanations cannot win: the random control has $S \approx 0$.
- Selecting by $S$ is far better than simply preferring brevity.

**What it does not do at $H = 20$.** Averaged over latents, the pre-registered penalty is harsh enough that the shortest explanations win. For GPT-2 layer-6 latents, a 2-word label already captures most of the signal detection scoring can measure: $J$ {{H2.j_by_budget.gen_b2.mean:.3f}} vs {{H2.j_by_budget.gen_b40.mean:.3f}} at ~40 words. Two readings fit this result:

- many of these latents are close to single-token detectors, in line with the strong top-token baseline;
- detection on 17-token contexts cannot reward nuance beyond what a short label carries.

**Choosing $H$.** $H$ is the exchange rate between accuracy and brevity, the price in tokens of halving the score. It is a value judgement and should be set for the use case. In our sweep, labels win for $H \leq 40$, and ~10-word explanations need $H \approx 80$. Because $\log_2 S = \log_2 J - L/H$, $H$ also has an MDL reading: each token costs $1/H$ bits of $\log_2 J$.

# 6. Limitations

- **One SAE and one scorer family.** GPT-2 small, layer 6, with Llama 70B scorers. The reliability checks cover a second scorer on 50 latents only.
- **Short contexts and a small test set.** Contexts are 17 tokens, and each explanation is scored on 50 contexts rather than Paulo et al.'s 200, so single-explanation scores are noisy (Table 5). Our claims are about means over {{n_latents}} latents.
- **Possible overlap.** Neuronpedia's explanations were written from top activations. Holding out the 20 strongest contexts reduces but does not remove overlap with our test positives. The Exp 2 explanations are fully held out.
- **Sample bias.** The eligibility filter (≥ 45 activating contexts) biases the sample toward denser latents: {{data.eligible_pct:.1f}}% of latents are eligible.
- **Tokenizer dependence.** $L$ depends on the tokenizer, but rankings barely change with words or characters.
- **Untested properties.** We did not test whether shorter explanations are more *useful* to humans or downstream tools. We assume it, as the premise of the project.

# 7. Conclusion

Multiplying the above-chance detection score by $2^{-L/H}$ is a one-line, library-compatible way to make SAE explanation scores reward brevity. Used to choose explanations, it gives much shorter ones at a small cost in accuracy. The pre-registered strength $H = 20$ is, however, strong enough that on average the shortest explanations win, so $H$ should be chosen deliberately; in our post-hoc sweep, ~10-word explanations win only at $H \approx 80$.

{{include:templates_static/paper_related.md}}

# Appendix A: Pre-registration and deviations

Hypotheses, metrics, tests and falsification criteria were committed in `PREREG.md` before any results. Two deviations were made before any main-sample results.

1. The Exp 2 budget sentence asks for "approximately B words" instead of "at most B words", because delphi's explainer prompt already demands concision.
2. The negatives changed from Neuronpedia's zero rows to verified random OpenWebText windows, after the 10-latent debugging pilot showed the zero rows were near-duplicates.

The analysis plan was not changed. The $H$ sweep tests are labelled exploratory.

# Appendix B: Prompts

- **Scorer:** delphi's detection prompt (`delphi/scorers/classifier/prompts/detection_prompt.py`, commit `4fea06e`). It has a system prompt, three few-shot examples from this SAE (two of them latents 6048 and 9396, which we exclude), and the test batch.
- **Explainer:** delphi's `DefaultExplainer` prompt (`delphi/explainers/default/prompts.py`), with "Your final explanation (the [EXPLANATION]: line) must be approximately B words long." appended to the last user message.

Every prompt and response is in the run logs (published with the dataset).

# AI statement

This codebase and paper were written by Claude Code (Anthropic's Claude, model Claude Opus 5.5) from an exact specification provided by Libor "Burny" Burian, who directed the work. Human cleanup and rewriting are in progress.
