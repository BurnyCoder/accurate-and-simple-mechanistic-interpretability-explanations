# Summary of all experiments

<!-- Generated from templates/reports/summary.md; numbers come from results/main/numbers.json. -->

**Project.** We add a text-length factor to the standard SAE auto-interpretability score (detection scoring, [Paulo et al. 2024](https://arxiv.org/abs/2410.13928), via [delphi](https://github.com/EleutherAI/delphi)), so that explanations are rewarded for being both accurate and short: `S = max(J, 0) · 2^(−L/H)`. All experiments use GPT-2 small's layer-6 residual SAE (`6-res-jb`) and existing Neuronpedia data. Hypotheses were pre-registered in [PREREG.md](../PREREG.md).

**For a non-specialist.** Neural networks contain internal "detectors", and researchers attach a short text label to each one (e.g. "the word *notice*"). To check a label, another AI is shown only the label and asked to pick out the snippets of text where the detector fires. The usual grade counts only how well the label works. We also make long labels pay a price, so that "accurate and short" wins. We then tested whether that price actually changes which labels win, and by how much.

## Experiments at a glance

| # | Experiment | Question | Outcome | Report |
|---|---|---|---|---|
| 0 | Pilot + verified negatives | Does the pipeline work? Are the non-activating examples good? | Pipeline works ({{pilot.sample_failure_rate:.1%}} parse failures). Neuronpedia's zero rows were near-duplicates, so they were replaced by verified random negatives (SAE reproduces Neuronpedia, r = {{data.negatives.reproduce_neuronpedia.pearson_max:.3f}}). | [exp0](exp0_pilot_and_negatives.md) |
| 1 | Existing explanations | H1: is the scorer valid? How do sources compare under J and S? | H1 {{H1.supported}}: random J = {{H1.random.mean:.3f}}; gpt-4o-mini {{H1.gpt4omini.mean:.3f}}, gpt-3.5 {{H1.gpt35.mean:.3f}}. The top-token baseline matches gpt-4o-mini under S. | [exp1](exp1_existing_explanations.md) |
| 2a | Length budgets | H2: does J rise with length, with diminishing returns? | {{H2.supported}}: J {{H2.j_by_budget.gen_b2.mean:.3f}} (~2 words) → {{H2.j_by_budget.gen_b40.mean:.3f}} (~40 words). | [exp2](exp2_length_budgets.md) |
| 2b | Length budgets | **H3 (primary):** is there an interior S optimum at H = 20? | **Falsified** ({{H3.supported}}): S decreases monotonically. Exploratory: ~10 words ties ~2 words at H = 40 and wins at H = 80. | [exp2](exp2_length_budgets.md) |
| 2c | Selection | H4: does choosing by S give short and accurate picks? | {{H4.supported}}: {{H4.length_reduction:.0%}} shorter, −{{H4.j_loss_argmaxJ_minus_argmaxS.mean:.3f}} J; +{{H4.argmaxS_minus_argminL.mean:.3f}} J over the shortest. | [exp2](exp2_length_budgets.md) |
| 3 | Reliability | Do scores reproduce under a new shuffle and a different scorer? | Spearman J: retest {{reliability_retest.spearman_j:.2f}}, alternative scorer {{reliability_altscorer.spearman_j:.2f}}. | [reliability](reliability.md) |

All experiments in one file: [all_experiments.md](all_experiments.md).

## Main results table (method, baseline, control, ablation)

{{table:conditions}}

- **Ablation:** H = ∞ (plain J) is the J column.
- **H sweep:** see [exp2](exp2_length_budgets.md#exploratory-post-hoc-the-h-sweep).
- **Length unit:** S rankings with words or characters instead of tokens correlate with the token version at Spearman {{exploratory.unit_robustness_spearman.tokens_vs_words:.3f}} and {{exploratory.unit_robustness_spearman.tokens_vs_chars:.3f}}.

## Models and data used

No network was trained in this project. These are the models it uses.

| Model | Role | Architecture | Training / objective | Data | Notes |
|---|---|---|---|---|---|
| GPT-2 small ([Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) | Subject model whose activations are explained | 12-layer decoder-only transformer, d_model 768, 124M parameters | Next-token prediction | WebText | Run with TransformerLens to compute verified negatives |
| SAE `gpt2-small-res-jb`, layer 6 ([Bloom 2024](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream)) | Produces the latents being explained | 768 → 24,576 ReLU latents → 768 (expansion 32) | Reconstruction MSE + L1 sparsity (coefficient 8e-5), 300M tokens | OpenWebText, 128-token contexts | Loaded via sae_lens; reproduces Neuronpedia activations (r = {{data.negatives.reproduce_neuronpedia.pearson_max:.3f}}) |
| gpt-3.5-turbo, gpt-4o-mini (OpenAI) | Neuronpedia's explainers (existing explanations) | Proprietary | Proprietary | — | OpenAI token–activation-pair explainer method |
| Llama-3.1-70B-Instruct ([Meta 2024](https://arxiv.org/abs/2407.21783)) | Detection scorer and Exp 2 explainer | 70B decoder-only transformer | Pretraining + instruction tuning | — | Via OpenRouter, DeepInfra fp8, temperature 0; the scorer family used by Paulo et al. |
| Llama-3.3-70B-Instruct | Alternative scorer (reliability) | 70B decoder-only transformer | Pretraining + instruction tuning | — | OpenRouter default routing |

{{include:templates_static/glossary.md}}

## Learnings

- **Detection scoring is easy to set up from public data alone,** with one exception: the negatives. Neuronpedia exports top and quantile activations with exact per-token values, which are ideal positives. Its zero-activation rows, however, are near-duplicate windows. Verified random negatives need a forward pass of the subject model and SAE (minutes on a laptop GPU).
- **Read the scorer prompts.** The negative-example problem was invisible in the summary numbers and obvious in the logged prompts.
- **H is a value judgement.** At H = 20, the accuracy gained per extra token was smaller than the length cost for these latents. Pre-registering H prevented tuning it to make H3 come out true. The sweep is reported as exploratory.
- **Selection is where the score earns its keep.** Averaged curves (H3) and per-latent choices (H4) can disagree. A length-aware score is most useful for choosing among candidate explanations of the same latent.
- **Engineering:**
  - delphi on Windows needs WSL2, because vLLM is imported eagerly.
  - Keep the venv off `/mnt/c`.
  - Pin delphi to a commit, not to the PyPI release (bugs fixed on main).
  - See [exp0](exp0_pilot_and_negatives.md#other-issues-fixed-during-setup-learnings).

## Cost

Main run: {{api_usage.score.calls}} LLM calls, {{api_usage.score.total_tokens}} tokens, US${{api_usage.score.cost_usd:.2f}}. Reliability: US${{api_usage.reliability.cost_usd:.2f}}.
