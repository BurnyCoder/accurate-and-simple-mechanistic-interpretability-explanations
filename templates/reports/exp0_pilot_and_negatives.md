# Experiment 0: pilot run and verified negatives

<!-- Generated from templates/reports/exp0_pilot_and_negatives.md; numbers come from results/*/numbers.json. -->

**Goal.** Debug the pipeline end-to-end on 10 latents before the pre-registered main run. Pilot results are never used for confirmatory claims (PREREG section 3).

## Setup

- 10 eligible latents disjoint from the main sample (seed 1).
- The full pipeline: Exp 2 explanations (6 per latent) followed by detection scoring of every condition.
- Scorer and explainer: Llama-3.1-70B-Instruct via OpenRouter, pinned to DeepInfra.

## What the pilot checked, and what it found

| Check | Result |
|---|---|
| Prompts and responses logged in full | Yes (`logs/*_score_pilot.log`; uploaded with the dataset) |
| Scorer answers parse as `[0,1,…]` lists | Failure rate {{pilot.sample_failure_rate:.2%}} of samples in the final pilot |
| Explainer follows the word budgets | Median tokens ~2 words: {{pilot.by_condition.gen_b2.tokens_median:.0f}}, ~10 words: {{pilot.by_condition.gen_b10.tokens_median:.0f}}, ~40 words: {{pilot.by_condition.gen_b40.tokens_median:.1f}} |
| API cost of both pilot runs | US${{pilot.api_usage.score.cost_usd:.2f}} for {{pilot.api_usage.score.calls}} calls, summed over the discarded first run and the final run (both logged as `*_score_pilot.log`) |
| **Negatives are diverse** | **No.** Neuronpedia's zero-activation rows are sliding windows over a few documents (see below) |

## Issue found: Neuronpedia's zero-activation rows are not random negatives

Reading the logged scorer prompts showed the same sentence again and again among the non-activating examples, each time shifted by one token ("…pretrial conference in Jefferson County District Court next week…"). Across the {{data.neuronpedia_zero_rows.n_latents}} sampled latents (main + pilot), the export's {{data.neuronpedia_zero_rows.rows}} zero-activation rows contain only {{data.neuronpedia_zero_rows.distinct_texts}} distinct texts. Within a latent, the rows are overlapping windows of a few documents, and the same documents recur across latents.

Detection scoring measures precision only through its negatives. With nearly duplicate negatives, broad explanations would be penalized too little, and broad explanations are the very ones a length factor interacts with.

**Fix (PREREG deviation 2, recorded before any main results, after the first pilot's results; see below).** We switched to verified random negatives, as in delphi's default (`non_activating_source = "random"`) and in Paulo et al. (App. D.4):

1. GPT-2 small and the *same* SAE (sae_lens `gpt2-small-res-jb`, `blocks.6.hook_resid_pre`) run on {{data.negatives.pool}} random windows from Skylion007/openwebtext. Each window is one per document, 64 tokens of context, of which the last 17 are shown.
2. A window counts as a negative for a latent if the latent's activation is exactly 0 at all 17 displayed tokens. Each latent had at least {{data.negatives.zero_windows_min}} such windows (median {{data.negatives.zero_windows_median:.0f}}), and 25 are sampled per latent.

**Validation of this setup:**

- **Token format.** {{data.negatives.token_format_roundtrip:.0%}} of Neuronpedia's token strings map back to GPT-2 token ids when the format is assumed to be raw byte-level BPE with `Ġ` shown as a space. So negatives are rendered in exactly the format of the positives, and the scorer cannot tell them apart by formatting (one content difference remains; see below).
- **Same SAE.** Re-running {{data.negatives.reproduce_neuronpedia.n}} of Neuronpedia's top activating windows through our model+SAE (BOS + 17 tokens; the export lacks the longer original context) reproduces the exported maximum activations with Pearson r = {{data.negatives.reproduce_neuronpedia.pearson_max:.3f}}. The maximum falls on the same token in {{data.negatives.reproduce_neuronpedia.frac_argmax_same_token:.1%}} of windows, and the latent is active in {{data.negatives.reproduce_neuronpedia.frac_ours_positive:.0%}} of them.

With verified negatives, the final pilot's random control scored J = {{pilot.by_condition.random.j_mean:.3f}}, about chance. The first pilot run (with Neuronpedia's zero rows) is kept, unused, in `results/pilot_discarded_old_negatives/`.

**What was seen before the switch.** The fix was not blind to results. The first pilot run had already been scored and analysed when the switch was decided:

| pilot run (10 latents) | random-control J | H1 | H2 | H3 | H4 |
|---|---:|---|---|---|---|
| first, Neuronpedia zero rows (discarded) | {{pilot_discarded.H1.random.mean:+.3f}} | {{pilot_discarded.H1.supported}} | {{pilot_discarded.H2.supported}} | {{pilot_discarded.H3.supported}} | {{pilot_discarded.H4.supported}} |
| final, verified random negatives | {{pilot.H1.random.mean:+.3f}} | {{pilot.H1.supported}} | {{pilot.H2.supported}} | {{pilot.H3.supported}} | {{pilot.H4.supported}} |

The switch was prompted by the duplicate sentences in the logged prompts, and it matches delphi's default and Paulo et al.'s random negatives. A random control above chance means the scorer separates positives from negatives without a relevant explanation, which points to the negatives. Still, a reader should know that H1 and H4 changed from failing to holding on these 10 latents, and that the main run started only after the switch: the deviation's commit (`393e156`) predates the earliest `time` field in `results/main/scores.jsonl`.

**Remaining difference between positives and negatives.** Neuronpedia's windows can cross document boundaries, so {{checks.eot.pct_positives:.1f}}% of the main run's test positives contain `<|endoftext|>`. Our negatives are single windows inside one document and contain it {{checks.eot.pct_negatives:.0f}}% of the time. The scorer does not use it as a cue: given the random control's unrelated explanations, it says "yes" to {{checks.eot.random_positive_rate.listed:.1%}} of those positives and to {{checks.eot.random_positive_rate.other:.1%}} of the others. Re-running H1–H4 without them gives the same verdicts (paper §4.5).

## Other issues fixed during setup (learnings)

| Issue | Cause | Fix and why |
|---|---|---|
| `delphi` would not install on Windows | It hard-depends on `vllm>=0.10.2`, which has [no native Windows support](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html), and `delphi/clients/__init__.py` imports `offline.py`, which imports vLLM at module level | Run under WSL2 Ubuntu. Not vendoring delphi's code keeps the base method exactly as published. |
| Every Python start took minutes | The venv was on `/mnt/c`, and WSL reads the Windows drive over the slow 9p protocol; torch and vLLM are thousands of files | Keep the venv on ext4 (`UV_PROJECT_ENVIRONMENT=~/.venvs/concise-interp`); the project stays on the Windows drive. |
| delphi 0.1.3 on PyPI misreports failures | `compute_confusion` returns `failed_count = len(df_valid) - total` (always 0); its explanation parser grabs everything after the first `[EXPLANATION]:` | Pinned delphi to GitHub main commit `4fea06e`, which fixes both. |
| `delphi.log.result_analysis` import error | It imports plotly at module level (the `visualize` extra) | Depend on `eai-delphi[visualize]`. |
| Download phase crashed | `activations/v2/` holds the same rows in a sparse format with different keys | Use top-level batches only. |
| Exp 2 "at most B words" would collapse | delphi's explainer prompt already says "Keep your explanations short and concise" | Changed to "approximately B words" before any results (PREREG deviation 1). The analysis uses the actual length. |
| tiktoken raised on some explanations | An explanation can quote the literal `<\|endoftext\|>`, which tiktoken treats as a disallowed special token | `encode(..., disallowed_special=())` counts it as text. |
| Python crash message at exit of the negatives phase | Hugging Face streaming threads during interpreter shutdown | Harmless: it happens after the output file is written; noted. |
