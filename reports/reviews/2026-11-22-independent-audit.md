# Independent audit, 2026-11-22

Author of the project: Libor "Burny" Burian. This audit was done by Claude Code without using earlier audits. No model was run: no LLM calls, no GPT-2/SAE forward passes, no training. Everything below comes from the committed code and `results/`, the installed libraries, and the cited sources.

## Verdict

No error found invalidates the paper's conclusions. The pre-registered verdicts stand: H1 supported, H2 supported, **H3 (primary) falsified**, H4 supported.

The audit found and fixed:

- two defects in the H4 selection code;
- one statistics detail in the paired Wilcoxon tests;
- about twenty claims that were mis-attributed, imprecise or stated too strongly.

The code fixes moved H4's values in the third decimal and some p-values slightly. No verdict and no qualitative statement changed.

## 1. How the audit was done

| Check | Method | Result |
|---|---|---|
| Every headline number | `scripts/independent_check.py` (new). It re-implements BA, J, L, S, the bootstrap, Wilcoxon, Holm and the H4 cross-fit from the PREREG text, without importing the project's package. | 43 of 43 checks match `results/main/numbers.json` exactly: Table 1 (J, S, CIs, median tokens), H2, all six H3 tests, H4. |
| Base-method usage | Read delphi's installed source, the pinned commit `4fea06e`. | `compute_classification_metrics()["accuracy"]` is balanced accuracy, `(tp/pos + tn/neg)/2`. `DetectionScorer` scores `record.test` plus `record.not_active`, shuffled with `random.Random(seed)` and batched by `n_examples_shown`. Its default is 1, and the project sets 5, as Paulo et al. App. D.1 do. `seed` and `max_retries` reach `Classifier(seed=…)` and the client as the code says. `DefaultExplainer` defaults to `activations=True`, and its prompt contains "Keep your explanations short and concise". The few-shot latents are `6-res-jb` 6048 and 9396 and `8-res-jb` 12654, as the paper says. |
| Data provenance | Local data files, logs and git history. | 24,576 latents; 17,226 eligible (70.1%); the filter shares are as reported. 73 raw batches (201 MB). All 210 sampled latents have exactly 25 zero rows. Every scored latent has 50 sample ids, and all conditions of a latent share the same sample order, so the batches are identical. The reproduction check's 152 windows are all among their latent's top-20 activating rows. Commit `393e156` (deviation 2) is at 02:38:12 UTC on 2026-10-29. The main run's log starts at 02:38:56 UTC and its first scored row is at 02:41:16 UTC. The `negatives` phase took about 3 minutes and `analyze` 2.5–4 minutes, matching the README. |
| SAE and dataset facts | [cfg.json](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted/blob/main/blocks.6.hook_resid_pre/cfg.json); Neuronpedia's `v1/gpt2-small/6-res-jb/source.jsonl`; the sae_lens `pretrained_saes.yaml`. | d_sae 24,576, expansion 32, L1 coefficient 8e-5, 300M tokens, 128-token contexts, `Skylion007/openwebtext`, `standard` architecture with ReLU. Neuronpedia's source lists `Skylion007/openwebtext` with 128-token prompts. |
| Citations | Web: arXiv HTML of each paper, Crossref, JSTOR (via search). | See §3. |
| Hypothesis evaluation | Each PREREG criterion against `numbers.json`. | See §4. |

## 2. Defects in the code (fixed)

**2.1 H4 broke exact ties by API completion order.**

- PREREG H4 says "Ties go to the shorter explanation". The code used `max()`/`min()`, which return the first of equal keys, over the rows of `results/main/scores.jsonl`.
- Those rows are in the order in which API calls finished, so a candidate tied in both score and length (for example two 8-token explanations with equal J on 25 contexts) was chosen by timing.
- Evidence: re-implementing H4 with a fixed candidate order instead of file order changed the J loss from 0.0314 to 0.0326.
- Fix: canonical order, then a seeded per-latent shuffle (`hypotheses.crossfit_selection`). The test `test_crossfit_exact_ties_do_not_depend_on_row_order` fails on the old code.

**2.2 Equal J values could differ in their last floating-point bits.**

- delphi computes `BA = (tp/pos + tn/neg)/2`. Mathematically equal J from different confusion counts do not always round to the same float. With 12 contexts per class, 4/12 − 0/12 gives 0.33333333333333326, but 6/12 − 2/12 gives 0.3333333333333335.
- Such ties were therefore decided by rounding noise, not by length, which breaks the pre-registered tie rule.
- The same noise turned zero paired differences into ±1e-17 and split tied ranks in the Wilcoxon tests.
- Fix: J is compared after rounding to 12 decimals (`_rankable`), and paired differences are rounded the same way before `scipy.stats.wilcoxon` (`stats._differences`).
- Tests: `test_crossfit_j_ties_hidden_by_float_noise_still_go_to_the_shorter_explanation` and `test_paired_treats_a_float_noise_difference_as_zero`. Both fail on the old code.

**Effect of both fixes.** Old values are from `numbers.json` at commit `81fbcf4`; new values are from the current `numbers.json`. Unlisted values did not change.

| quantity | before | after |
|---|---|---|
| H4 held-out J, pick highest J | 0.644 [0.605, 0.682] | 0.642 [0.604, 0.679] |
| H4 held-out J, pick highest S | 0.613 [0.574, 0.649] | 0.612 [0.574, 0.648] |
| H4 held-out J, pick shortest | 0.467 [0.425, 0.509] | 0.466 [0.425, 0.508] |
| H4 J loss (criterion ≤ 0.05) | 0.031 [0.018, 0.046] | 0.030 [0.017, 0.045] |
| H4 S minus shortest (criterion CI > 0) | 0.146 [0.115, 0.179] | 0.145 [0.114, 0.179] |
| H4 length reduction (criterion ≥ 20%) | 39% | 39% |
| H4 over 20 splits: criteria hold / mean length reduction | 20 of 20 / 33% | 20 of 20 / 32% |
| H2 rise, one-sided Wilcoxon p | 7.1e-09 | 9.0e-09 |
| gpt-4o-mini − gpt-3.5 J, two-sided p (exploratory) | 0.065 | 0.059 |

The two 10-latent pilot runs, cited only in the paper's Appendix A and in exp0, were recomputed with the fixed code. Their H1–H4 verdicts, random-control J and H4 J loss are unchanged, so their `numbers.json` files were left as they are.

New output: `H4.tie_rate`. In 53% of argmax-J selections the best J is shared by at least two candidates, against 5% for argmax-S. This replaces the earlier admission "we did not measure how often ties occur" in the Exp 2 report.

## 3. Claims corrected

| Where | Was | Problem (source) | Now |
|---|---|---|---|
| Paper §2, `negatives.py`, exp0 | Paulo et al. use "100 random non-activating contexts (§4.2, App. D.4, D.6)"; "random negatives … (App. D.4)" | In [arXiv:2410.13928 v3](https://arxiv.org/html/2410.13928), App. D.4 is *Embedding details*. For detection, §4.2 and D.6 give only the counts. That the negatives are random is implied by App. D.5 (*Adversarial examples*), which discusses random non-activating contexts as a limitation. | Counts cite §4.2 and D.6; randomness cites D.5. |
| Paper §1, related work, methodology | Ayonrinde et al. apply MDL "to the SAE's latent code and decoder"; "the same fit-plus-length trade-off" | In [arXiv:2410.11179](https://arxiv.org/abs/2410.11179) (§2, Def. 2.3), the decoder is an O(1) term they neglect. The working objective counts the bits of the latent code, subject to a reconstruction-error tolerance (a constraint, not an additive fit term). | "use MDL to compare SAEs, counting the bits of the latent code under a bound on reconstruction error"; our score is a multiplicative trade-off, not a constraint. |
| Paper, README, glossary | "percentile bootstrap CIs (Efron 1979)" | Efron 1979 introduced the bootstrap. The percentile interval is usually cited from Efron & Tibshirani (1993, ch. 13; [Wikipedia's citation](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))). | Both cited, [doi:10.1201/9780429246593](https://doi.org/10.1201/9780429246593). |
| Paper §1, related work | "GPT-4 as a judge prefers longer answers more than human raters do" | [Saito et al.](https://arxiv.org/abs/2310.10076) say this "in our problem setting" (HH-RLHF agreement analysis). | Scoped to their setting. |
| Paper §1 | "leaderboards have had to correct for length" | [Dubois et al.](https://arxiv.org/abs/2404.04475) show this for AlpacaEval only. | "the AlpacaEval leaderboard added a length-controlled win rate". |
| Paper §1, related work, methodology | van der Ben et al. found detection "the most stable"; "pipeline choices dominate" | [arXiv:2607.19386](https://arxiv.org/abs/2607.19386): the scorer was held fixed (Gemini Flash). The varied pipeline choices were corpus, example draw, explainer and paraphrase. "Most stable" is the abstract's wording; reliability there is ICC-based. | Scorer-held-fixed and the four metrics stated. Their App. E.8 ten-word cap, which bears on our H2, is added: the text reports lower detection scores for Apertus-8B and little effect for Pythia-160M, and the figure caption there disagrees. |
| Abstract, §1, README | "delphi's unmodified … explainer" | Five of the six Exp 2 prompts append a budget sentence (`explain.py`). | "its explainer prompt (plus one length sentence)". |
| Abstract, §4.1, README | "The scorer is valid" | H1 is a sanity check (random ≈ 0, real > 0), not a validity study. | "passes the pre-registered validity check". |
| Paper §3, limitations, exp0 | "The scorer does not treat `<|endoftext|>` as a cue" | Evidence: a "yes" rate of 13.5% on 414 EOT positives vs 12.4% on the others, under random explanations. This cannot exclude a small effect. The stronger evidence is the robustness re-run. | "We see no sign that the scorer uses it as a cue". |
| Paper §3 | "our detection task is easier" | An inference. Paulo et al. (Fig. 2 caption) report lower accuracy on lower-activating deciles. | "likely easier", with that citation. |
| Paper limitations | H2 and H3 "are not affected" by self-preference | Only a self-preference that is constant across budgets cancels. | Stated that way. |
| Reliability report | split-half is "a lower bound on the agreement at 50" | The halves were scored in shared 5-context prompts, which may correlate their answers. | Hedged in both directions. |
| Paper §3 (H4) | — | The cross-fit halves share prompts (same batches), so selection and evaluation answers are not fully independent. | Disclosed. |
| Paper §5 | "H also has an MDL reading" | Contradicts §1's "analogous to, not an instance of". | "an MDL-style reading, by analogy only". |
| Abstract, §1 | "the smallest change" | Not demonstrable. | "a minimal change". |
| README, exp2, paper §4.4 | "20 other / different splits … instead of the pre-registered one" | The 20 seeds (0–19) include the pre-registered seed 0 (`h4_split_sensitivity`). | "consecutive seeds from 0, including the pre-registered one". |
| exp2 | "The pre-registered prediction is therefore falsified" (after the ratio illustration) | The falsification comes from the tests; the ratio explains it. | "This is why the pre-registered prediction fails." |
| `docs/methodology.md` | implementation map lists `analysis.py` for metrics, CIs, tests | Out of date after the split into `metrics.py`, `stats.py` and `hypotheses.py`. | Updated. Also "25 zero rows" → "up to 25 (exactly 25 for every sampled latent)". |
| README, paper, `config.py` | DeepInfra serves the scorer in fp8 (checked 2026-10-31) | [OpenRouter's endpoint list](https://openrouter.ai/api/v1/models/meta-llama/llama-3.1-70b-instruct/endpoints) on 2026-11-22 shows only Amazon Bedrock (quantization "unknown"). | Dated note plus a limitation: re-scoring needs another provider; the stored predictions reproduce the analysis. |

## 4. Hypothesis evaluation against the pre-registration

| H | PREREG criterion | Observed (current `numbers.json`) | Verdict |
|---|---|---|---|
| H1 | random: CI contains 0 **or** \|mean\| < 0.05; gpt35 and gpt4omini lower CI > 0 | random −0.022 [−0.038, −0.006]: the CI excludes 0, but \|mean\| < 0.05; gpt35 0.444 [0.400, 0.490]; gpt4omini 0.499 [0.453, 0.547] | supported (via the tolerance arm; the papers say the CI lies just below 0) |
| H2 | one-sided Wilcoxon J(40) > J(2), p < 0.05; gain per nominal word 2→10 minus 10→40, CI > 0 | +0.113 [0.079, 0.147], p = 9.0e-09; +0.0095 [0.0056, 0.0136] | supported |
| H3 | some B ∈ {5, 10, 20} beats both endpoints, Holm over 6 one-sided tests | no budget beats ~2 words (Holm p = 1 for all three) | **falsified** |
| H4 | median L ratio ≤ 0.8; mean J loss ≤ 0.05; argmax-S − argmin-L CI > 0 | 0.61 (5.5 / 9.0); 0.030; 0.145 [0.114, 0.179] | supported |

The code implements each criterion as written (`hypotheses.py`). Holm is applied within H3, the only hypothesis with several tests. One-sided tests match the directional predictions.

The Wilcoxon signed-rank test is a test of location, not of the mean, while H3 is phrased in means. Here the mean differences and the tests agree in sign everywhere (Table 2), so this does not matter.

## 5. Claims verified as correct (sample)

- **Paulo et al.**
  - The quote "current scoring methods do not account for interpretation length" (§5).
  - 5 shuffled examples per detection prompt (App. D.1).
  - 100 + 100 contexts and 10 per decile (§4.2, D.6).
  - Llama 3.1 70B Instruct as explainer (§3.2) and scorer (D.6).
  - 4-bit quantization (App. F.1 onward).
  - Precision and recall (§3.3.1).
  - Detection at least 5× cheaper than all-at-once simulation, by token-count estimate (§4.1, Table 1: $588 vs $3.6k for Llama 70B).
- **SAEBench.** §3.2.2; gpt-4o-mini judge; 14 sequences (10 random, 2 top, 2 importance-weighted, App. D); "similar to the detection score" (App. D).
- **Ayonrinde et al. 2024.** The quote "both accurate and concise" (abstract).
- **Ayonrinde & Jaburi 2025.** Simplicity is among the explanatory virtues.
- **Bloom 2024.** 2 Feb 2024; expansion factor 32; 300M OpenWebText tokens.
- **References.** Crossref confirms the DOIs of Youden 1950, Wilcoxon 1945, Efron 1979, Rissanen 1978 and Chernozhukov et al. 2018. Holm 1979 is JSTOR 4615733. arXiv:2407.21783 is "The Llama 3 Herd of Models".
- **Neuronpedia bucket.** `activations/`, `explanations/`, `features/` exist under `v1/gpt2-small/6-res-jb/`.

## 6. Remaining limitations (not errors; all disclosed in the paper)

- One SAE (GPT-2 small, layer 6) and one scorer family; the second scorer was checked on 50 latents only.
- 25 + 25 contexts per explanation. Within-latent split-half Spearman of J is 0.66, so per-latent claims are weak. The hypotheses are about means over 200 latents.
- Test positives are strong activations only (all ≥ 10% of the latent's top exported activation), so absolute J is not comparable with Paulo et al.
- The same model explains (Exp 2) and scores. Comparisons between Exp 2 and Neuronpedia explanations, including H4's candidate pool, are confounded.
- Deviation 2 (the negatives) was made after the first pilot had been seen, which the paper discloses. The H sweep is post hoc: it picks the best of five budgets and is uncorrected.
- Not independently verifiable today: that DeepInfra served fp8 on 2026-10-31 (a dated snapshot of a page that has since changed).

## 7. Reproduce this audit

```bash
uv run pytest -q                                   # 65 tests, including the three new ones
uv run python run.py --phase analyze --sample main # numbers.json, figures, README, reports, paper
uv run python scripts/independent_check.py         # 43 of 43 checks match
```
