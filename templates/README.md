# Accurate and simple mechanistic-interpretability explanations

<!-- Generated from templates/README.md by `uv run python run.py --phase analyze --sample main`.
     Every number below is filled from results/main/numbers.json; edit the template, not this file. -->

**Length-penalized detection scoring for sparse-autoencoder (SAE) feature explanations: a score that rewards explanations for being both accurate and short.**

Detection scoring ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)) is implemented in EleutherAI's [delphi](https://github.com/EleutherAI/delphi), and SAEBench's autointerp metric is built on it ([Karvonen et al. 2025](https://arxiv.org/abs/2503.09532)). It is the standard way to score SAE feature explanations. An LLM gets only the explanation and must say which of 5 text snippets the feature fires on. That score rewards accuracy only; Paulo et al. note that "current scoring methods do not account for interpretation length". This repo adds one factor:

```
S = max(J, 0) · 2^(−L / H)
J = 2·balanced_accuracy − 1      (0 = chance, 1 = perfect; Youden's J)
L = explanation length in tokens (tiktoken o200k_base)
H = half-life: the score halves every H tokens (pre-registered: 20)
```

In log space, `log S = log J − (ln 2 / H)·L`: a constant cost per token, the minimum-description-length trade-off between fit and description length. The data takes no model runs and no training:

- Explanations and activating contexts come from [Neuronpedia's public export](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html) for GPT-2 small (`6-res-jb`, [Bloom 2024](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream)).
- Non-activating contexts are random OpenWebText windows on which the same SAE is verified to be 0.
- delphi's scorer and explainer run unmodified, with Llama-3.1-70B via OpenRouter, the scorer family of Paulo et al.

Everything was pre-registered in [PREREG.md](PREREG.md) before any results existed. **[Paper (PDF)](paper/paper.pdf)** · [Reports](reports/) · [Interactive explorer](https://burnycoder.github.io/accurate-and-simple-mechanistic-interpretability-explanations/) · [Methodology](docs/methodology.md) · [Scored data on Hugging Face](https://huggingface.co/datasets/BurnyCoder/length-penalized-autointerp-gpt2-small) ([collection](https://huggingface.co/collections/BurnyCoder/accurate-and-simple-interpretability-explanations-6ae2bbf0e413a206073deb4e))

## Highlights

{{data.n_latents_total}} latents in the SAE; {{n_latents}} sampled at random; {{n_explanations}} explanations scored; {{api_usage.score.calls}} LLM calls (US${{api_usage.score.cost_usd:.2f}}).

- ✅ **The base scorer is valid.**
  - Random explanations (another latent's explanation) score J = {{H1.random.mean:.3f}} [{{H1.random.lo:.3f}}, {{H1.random.hi:.3f}}], about chance.
  - Neuronpedia's gpt-4o-mini explanations score J = {{H1.gpt4omini.mean:.3f}} [{{H1.gpt4omini.lo:.3f}}, {{H1.gpt4omini.hi:.3f}}], and its gpt-3.5-turbo explanations {{H1.gpt35.mean:.3f}}.
- ✅ **Longer explanations are more accurate, with diminishing returns.**
  - When Llama-3.1-70B is asked for ~2 vs ~40 words, J goes from {{H2.j_by_budget.gen_b2.mean:.3f}} to {{H2.j_by_budget.gen_b40.mean:.3f}} (+{{H2.rise_b40_minus_b2.mean:.3f}} [{{H2.rise_b40_minus_b2.lo:.3f}}, {{H2.rise_b40_minus_b2.hi:.3f}}]).
  - The gain per word is larger from 2→10 words than from 10→40.
- ❌ **Primary hypothesis falsified: at the pre-registered H = 20 there is no interior optimum.**
  - Mean S *decreases* steadily with length: {{H3.s_by_budget.gen_b2.mean:.3f}} (~2 words) > {{H3.s_by_budget.gen_b5.mean:.3f}} > {{H3.s_by_budget.gen_b10.mean:.3f}} > {{H3.s_by_budget.gen_b20.mean:.3f}} > {{H3.s_by_budget.gen_b40.mean:.3f}} (~40 words).
  - At H = 20, the accuracy gained per extra token is smaller than the length cost, so on average the shortest explanation wins.
  - In an exploratory (not pre-registered) sweep, ~10 words ties ~2 words at H = 40 and clearly beats both ends at H = 80: +{{exploratory.h_sweep.80.tests.best_minus_gen_b2.mean:.3f}} [{{exploratory.h_sweep.80.tests.best_minus_gen_b2.lo:.3f}}, {{exploratory.h_sweep.80.tests.best_minus_gen_b2.hi:.3f}}] over ~2 words.
- ✅ **The length factor works as a selection rule.**
  - Among 8 candidate explanations per latent, picking by S instead of plain detection J gives explanations {{H4.length_reduction:.0%}} shorter (median {{H4.median_tokens.argmax_s:.1f}} vs {{H4.median_tokens.argmax_j:.1f}} tokens).
  - The held-out J loss is only {{H4.j_loss_argmaxJ_minus_argmaxS.mean:.3f}} [{{H4.j_loss_argmaxJ_minus_argmaxS.lo:.3f}}, {{H4.j_loss_argmaxJ_minus_argmaxS.hi:.3f}}]: about {{H4.j.argmax_s.mean:.3f}} vs {{H4.j.argmax_j.mean:.3f}}.
  - Picking by S beats "pick the shortest" by {{H4.argmaxS_minus_argminL.mean:.3f}} J. Selection and evaluation used disjoint halves of the test snippets.
- 🔎 **The laziest baseline is competitive.**
  - `the token "X"` (median {{by_condition.top_token.tokens_median:.0f}} tokens) gets J = {{by_condition.top_token.j_mean:.3f}}, close to gpt-3.5-turbo's {{by_condition.gpt35.j_mean:.3f}}.
  - Under S it scores {{by_condition.top_token.s_mean:.3f}}, above gpt-3.5-turbo ({{by_condition.gpt35.s_mean:.3f}}) and level with gpt-4o-mini ({{by_condition.gpt4omini.s_mean:.3f}}).
- 🔁 **The conclusions survive re-scoring.**
  - Per-explanation rankings are stable under a new batch shuffle: Spearman J = {{reliability_retest.spearman_j:.2f}}, S = {{reliability_retest.spearman_s:.2f}}.
  - They are also stable under a different scorer (Llama-3.3-70B): J = {{reliability_altscorer.spearman_j:.2f}}, S = {{reliability_altscorer.spearman_s:.2f}}.
  - Measuring length in words or characters instead of tokens changes S rankings negligibly (Spearman ≥ {{exploratory.unit_robustness_spearman.tokens_vs_chars:.3f}}).

![Exp 2: J rises with length, S falls at H = 20](figures/fig3_budget_curves.png)

## Results

### Every explanation source, under plain detection (J) and the length-penalized score (S)

{{table:conditions}}

BA = balanced accuracy; J = 2·BA − 1; 95% bootstrap CIs over latents. *existing* = Neuronpedia's explanations; *Exp 2* = generated by Llama-3.1-70B with delphi's explainer prompt, asked for ~B words.

![J and S per explanation source](figures/fig1_conditions.png)

### H3 (primary): is there an intermediate length that beats both extremes at H = 20?

{{table:h3}}

No. The ~2-word explanations have the highest mean S, and every longer budget is significantly lower. H3 is **falsified** for H = 20.

### Where the optimum sits depends on H (exploratory)

{{table:h_sweep}}

![Mean S per length budget for each half-life H](figures/fig4_h_sweep.png)

H is an exchange rate between accuracy and brevity, not a measured quantity. For these GPT-2 latents, H = 20 is strong enough that the shortest explanations win on average. At H = 40, ~10-word and ~2-word explanations tie ({{exploratory.h_sweep.40.tests.best_minus_gen_b2.mean:+.3f}} [{{exploratory.h_sweep.40.tests.best_minus_gen_b2.lo:+.3f}}, {{exploratory.h_sweep.40.tests.best_minus_gen_b2.hi:+.3f}}]). Only around H = 80 do ~10-word explanations win, at the length where accuracy gains flatten out (see the Exp 2 curve above). These sweep comparisons are post hoc.

### H4: does selecting by S give short and accurate explanations?

{{table:h4}}

![Selection rules](figures/fig5_selection.png)

### Reliability

{{table:reliability}}

![Every scored explanation, J vs length](figures/fig2_j_vs_length.png)

### In plain words

- Compared with **plain detection scoring**, the new score prefers explanations that are {{H4.length_reduction:.0%}} shorter and gives up only {{H4.j_loss_argmaxJ_minus_argmaxS.mean:.3f}} accuracy (J) when picking the best explanation per feature. That is the intended trade.
- Compared with the **"just pick the shortest" baseline**, it keeps much more accuracy ({{H4.argmaxS_minus_argminL.mean:.3f}} higher J). Brevity alone is not what it rewards.
- The **random-explanation control** scores about zero under both J and S ({{by_condition.random.s_mean:.3f}}), so being short cannot rescue an unrelated explanation.
- **Where it did not work as hoped:** at the pre-registered strength (H = 20) the score pushes, on average, all the way to ~2-word labels. If descriptive ~10-word explanations are wanted, a much weaker penalty (H ≈ 80) is needed.
- **Ablation:** setting H = ∞ removes the length factor and recovers plain detection scoring, under which the longest (~40-word) explanations are best on average.

## Method in brief

- **Subject model and SAE:** GPT-2 small, residual stream before block 6, Joseph Bloom's 24,576-latent SAE (`gpt2-small-res-jb`).
- **Latents:** {{n_latents}} drawn at random (seed 0) from the {{data.n_eligible}} ({{data.eligible_pct:.1f}}%) that have both Neuronpedia LLM explanations and ≥ 45 activating contexts. delphi's two few-shot latents are excluded.
- **Contexts per latent:**
  - the 20 strongest activating contexts go to the explainer only;
  - 25 activating test contexts are stratified over 5 activation-quantile bins;
  - 25 negatives are random 17-token OpenWebText windows on which the latent's activation is exactly 0.
- **Checking the negatives:** we run GPT-2 small + the SAE through sae_lens. Our activations reproduce Neuronpedia's exported maxima with Pearson r = {{data.negatives.reproduce_neuronpedia.pearson_max:.3f}}.
- **Scorer:** delphi `DetectionScorer`, 5 contexts per prompt, temperature 0, Llama-3.1-70B-Instruct (OpenRouter, pinned to DeepInfra).
- **Statistics:** paired bootstrap 95% CIs (10,000 resamples over latents), one-sided Wilcoxon signed-rank tests, Holm correction.
  - H4 selects on one half of the test contexts and evaluates on the other, to avoid the winner's curse.
  - Parse failures affected {{sample_failure_rate:.3%}} of samples.

Details: [docs/methodology.md](docs/methodology.md), [PREREG.md](PREREG.md) (including two deviations made before any main results), [reports/](reports/).

{{include:templates_static/README_body.md}}
