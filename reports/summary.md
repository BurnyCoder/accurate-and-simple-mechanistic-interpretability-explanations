# Summary of all experiments

<!-- Generated from templates/reports/summary.md; numbers come from results/main/numbers.json. -->

**Project.** We add a text-length factor to a widely used SAE auto-interpretability score (detection scoring, [Paulo et al. 2024](https://arxiv.org/abs/2410.13928), via [delphi](https://github.com/EleutherAI/delphi)), so that explanations are rewarded for being both accurate and short: `S = max(J, 0) · 2^(−L/H)`. All experiments use GPT-2 small's layer-6 residual SAE (`6-res-jb`) and existing Neuronpedia data. Hypotheses were pre-registered in [PREREG.md](../PREREG.md).

**For a non-specialist.** Neural networks contain internal "detectors", and researchers attach a short text label to each one (e.g. "the word *notice*"). To check a label, another AI is shown only the label and asked to pick out the snippets of text where the detector fires. The usual grade counts only how well the label works. We also make long labels pay a price, so that "accurate and short" wins. We then tested whether that price actually changes which labels win, and by how much.

## Experiments at a glance

| # | Experiment | Question | Outcome | Report |
|---|---|---|---|---|
| 0 | Pilot + verified negatives | Does the pipeline work? Are the non-activating examples good? | Pipeline works (0.0% parse failures). Neuronpedia's zero rows were near-duplicates, so they were replaced by verified random negatives (SAE reproduces Neuronpedia, r = 0.988). The switch came after the first pilot's results had been seen. | [exp0](exp0_pilot_and_negatives.md) |
| 1 | Existing explanations | H1: is the scorer valid? How do sources compare under J and S? | H1 True: random J = -0.022; gpt-4o-mini 0.499, gpt-3.5 0.444. Under S, the top-token baseline is not distinguishable from either Neuronpedia source (post hoc). | [exp1](exp1_existing_explanations.md) |
| 2a | Length budgets | H2: does J rise with length, with diminishing returns? | True: J 0.464 (~2 words) → 0.577 (~40 words). | [exp2](exp2_length_budgets.md) |
| 2b | Length budgets | **H3 (primary):** is there an interior S optimum at H = 20? | **Falsified** (False): S decreases monotonically. Exploratory: ~10 words ties ~2 words at H = 40; at H = 80 its higher mean is not significant (one-sided p = 0.0505, post hoc). | [exp2](exp2_length_budgets.md) |
| 2c | Selection | H4: does choosing by S give short and accurate picks? | True: 39% shorter, −0.031 J; +0.146 J over the shortest. | [exp2](exp2_length_budgets.md) |
| 3 | Reliability | Do scores reproduce under a new shuffle, a different scorer, and different contexts? | Same contexts: within-latent Spearman J 0.91 (retest), 0.92 (alternative scorer). Disjoint context halves: 0.66, so single scores are noisy and the claims rest on means. | [reliability](reliability.md) |

All experiments in one file: [all_experiments.md](all_experiments.md).

## Main results table (method, baseline, control, ablation)

| explanation source | role | n | mean BA | mean J [95% CI] | median tokens | mean S (H=20) [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| random-explanation control | control | 200 | 0.489 | -0.022 [-0.038, -0.006] | 8 | 0.021 [0.015, 0.029] |
| top-token baseline | baseline | 200 | 0.715 | 0.430 [0.382, 0.477] | 5 | 0.365 [0.326, 0.404] |
| gpt-3.5-turbo (Neuronpedia) | existing | 200 | 0.722 | 0.444 [0.400, 0.490] | 8 | 0.341 [0.307, 0.377] |
| gpt-4o-mini (Neuronpedia) | existing | 200 | 0.750 | 0.499 [0.453, 0.547] | 8 | 0.374 [0.339, 0.408] |
| Llama-3.1-70B, ~2 words | Exp 2 | 200 | 0.732 | 0.464 [0.423, 0.506] | 2 | 0.430 [0.393, 0.467] |
| Llama-3.1-70B, ~5 words | Exp 2 | 200 | 0.747 | 0.493 [0.455, 0.530] | 6 | 0.395 [0.365, 0.424] |
| Llama-3.1-70B, ~10 words | Exp 2 | 200 | 0.774 | 0.548 [0.508, 0.587] | 11 | 0.374 [0.347, 0.400] |
| Llama-3.1-70B, ~20 words | Exp 2 | 200 | 0.776 | 0.552 [0.512, 0.591] | 20 | 0.278 [0.258, 0.297] |
| Llama-3.1-70B, ~40 words | Exp 2 | 200 | 0.789 | 0.577 [0.540, 0.613] | 45 | 0.123 [0.115, 0.131] |
| Llama-3.1-70B, delphi default prompt | Exp 2 | 200 | 0.781 | 0.563 [0.524, 0.601] | 30 | 0.204 [0.189, 0.220] |

- **Ablation:** H = ∞ removes the length factor and leaves max(J, 0), plain detection clipped at 0 (sweep table in [exp2](exp2_length_budgets.md#exploratory-post-hoc-the-h-sweep)); the J column is plain detection without the clip.
- **H sweep:** see [exp2](exp2_length_budgets.md#exploratory-post-hoc-the-h-sweep).
- **Length unit:** S rankings with words or characters instead of tokens correlate with the token version at Spearman 0.998 and 0.996.

## Models and data used

No network was trained in this project. These are the models it uses.

| Model | Role | Architecture | Training / objective | Data | Notes |
|---|---|---|---|---|---|
| GPT-2 small ([Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) | Subject model whose activations are explained | 12-layer decoder-only transformer, d_model 768, ~124M parameters (117M in Radford et al., Table 2) | Next-token prediction | WebText | Run with TransformerLens to compute verified negatives |
| SAE `gpt2-small-res-jb`, layer 6 ([Bloom 2024](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream)) | Produces the latents being explained | 768 → 24,576 latents → 768 (expansion 32); ReLU encoder (sae_lens `standard` architecture) | Reconstruction MSE + L1 sparsity (coefficient 8e-5), 300M tokens ([cfg.json](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted/blob/main/blocks.6.hook_resid_pre/cfg.json)) | OpenWebText, 128-token contexts | Loaded via sae_lens; reproduces Neuronpedia activations (r = 0.988) |
| gpt-3.5-turbo, gpt-4o-mini (OpenAI) | Neuronpedia's explainers (existing explanations) | Proprietary | Proprietary | — | OpenAI token–activation-pair explainer method |
| Llama-3.1-70B-Instruct ([Llama Team 2024](https://arxiv.org/abs/2407.21783)) | Detection scorer and Exp 2 explainer | 70B decoder-only transformer | Pretraining + instruction tuning | — | Via OpenRouter, pinned to DeepInfra (fp8 per [OpenRouter](https://openrouter.ai/api/v1/models/meta-llama/llama-3.1-70b-instruct/endpoints), checked 2026-10-31), temperature 0; the explainer and scorer model of Paulo et al. (who ran it at 4 bits) |
| Llama-3.3-70B-Instruct | Alternative scorer (reliability) | 70B decoder-only transformer | Pretraining + instruction tuning | — | OpenRouter default routing, so several providers served it ([reliability](reliability.md)) |

## Key terms

| Term | Meaning | In plain words |
|---|---|---|
| SAE latent / feature | One unit of a sparse autoencoder trained on a model's internal activations. | A "detector" inside the model that switches on for some pattern in text. |
| Explanation | A short text describing when a latent fires, e.g. `the token "notice"`. | A label that says what the detector detects. |
| Detection scoring | An LLM sees the explanation and 5 text snippets at a time, and says which snippets the latent fires on ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928), §3.3.1, App. D.1). | A quiz: given only the label, can someone pick out the right snippets? |
| BA (balanced accuracy) | (TPR + TNR) / 2: the mean of the hit rate on activating snippets and the correct-rejection rate on non-activating ones. | The quiz score, fair even if the classes are unbalanced. |
| J (Youden's J) | 2·BA − 1 = TPR − FPR. 0 = chance, 1 = perfect ([Youden 1950](https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3)). | How far above guessing the label gets you. |
| L | Length of the explanation in tokens (o200k_base tokenizer). | How long the label is; in these explanations one token is 0.77 words on average. |
| H (half-life) | The length at which the length factor halves the score. Pre-registered as 20 tokens. | How strongly length is penalized. |
| S | max(J, 0) · 2^(−L/H): **the proposed score**. | Accurate *and* short labels score high. |
| Top-token baseline | `the token "X"`, where X is the most common peak token in the strongest examples. | The laziest possible label. |
| Random control | Another latent's explanation. | A wrong label; it should score ~0. |
| Verified negatives | Random OpenWebText snippets on which the latent's activation is exactly 0 (checked by running GPT-2 small + the SAE). | Snippets where the detector is known to be off. |
| Cross-fitting | Choose an explanation using half of the test snippets, and measure it on the other half, then swap ([Chernozhukov et al. 2018](https://doi.org/10.1111/ectj.12097)). | Don't grade yourself on the questions you used to choose your answer. |
| Bootstrap CI | 95% percentile interval from resampling latents 10,000 times ([Efron 1979](https://doi.org/10.1214/aos/1176344552)). | The range the average would plausibly fall in with a different set of latents. |
| Wilcoxon / Holm | Paired signed-rank test ([Wilcoxon 1945](https://doi.org/10.2307/3001968)); Holm corrects p-values for running several tests ([Holm 1979](https://www.jstor.org/stable/4615733)). | Is the difference real, after accounting for running many tests? |


## Learnings

- **Detection scoring is easy to set up from public data alone,** with one exception: the negatives. Neuronpedia exports top and quantile activations with exact per-token values, which are ideal positives. Its zero-activation rows, however, are near-duplicate windows. Verified random negatives need a forward pass of the subject model and SAE (minutes on a laptop GPU).
- **Read the scorer prompts.** The negative-example problem was found in the logged prompts. In the summary numbers it showed only as an above-chance random control (J = +0.084 on 10 latents), which did not say why.
- **Reliability checks must vary the contexts.** Re-scoring the same contexts with a new seed or scorer agreed closely; scoring disjoint halves of the contexts agreed much less. Most of the noise in a single score comes from which contexts were drawn.
- **H is a value judgement.** At H = 20, the accuracy gained per extra token was smaller than the length cost for these latents. Pre-registering H prevented tuning it to make H3 come out true. The sweep is reported as exploratory.
- **Selection is where the score earns its keep.** Averaged curves (H3) and per-latent choices (H4) can disagree. A length-aware score is most useful for choosing among candidate explanations of the same latent.
- **Engineering:**
  - delphi on Windows needs WSL2: it hard-depends on vLLM, which has no native Windows support, and `delphi.clients` imports it at import time.
  - Keep the venv off `/mnt/c`.
  - Pin delphi to a commit, not to the PyPI release (bugs fixed on main).
  - See [exp0](exp0_pilot_and_negatives.md#other-issues-fixed-during-setup-learnings).

## Cost

Main run: 21220 LLM calls, 17122288 tokens, US$6.78. Reliability: US$2.03.
