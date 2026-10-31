# Methodology

This page explains how the project works and why each design choice was made. The pre-registered constants are in [PREREG.md](../PREREG.md), and results are in the [README](../README.md) and [reports/](../reports/).

## 1. Base method: detection scoring

Auto-interpretability scores an SAE latent's natural-language explanation by checking whether an LLM, given only that explanation, can predict where the latent fires. Two families are common:

| Method | Source | What the scorer LLM does | Cost / requirements |
|---|---|---|---|
| Simulation scoring | [Bills et al. 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) | Predicts a 0–10 activation for every token. The score is the correlation with the true activations. | The "all at once" variant needs prompt-token logprobs. |
| **Detection scoring** | [Paulo et al. 2024](https://arxiv.org/abs/2410.13928) | Says, for each of 5 shuffled contexts, whether it contains the latent (1/0) (§3.3.1, App. D.1). | Plain chat completions; logprobs optional. delphi's `DetectionScorer` defaults to 1 context per call; we set 5, as Paulo et al. do. |

We use **detection scoring**, through EleutherAI's [delphi](https://github.com/EleutherAI/delphi) library (Apache-2.0), with its prompt, few-shot examples, parser and metric code unchanged.

- SAEBench, a standardized SAE benchmark suite ([Karvonen et al. 2025](https://arxiv.org/abs/2503.09532)), uses a detection-style autointerp metric similar to it, with gpt-4o-mini as the judge (§3.2.2).
- A 2026 study of autointerp pipelines ([van der Ben et al. 2026](https://arxiv.org/abs/2607.19386)) found it the most stable of the four autointerp metrics it compared, on two subject models.
- It is at least 5× cheaper than all-at-once simulation, which needs prompt-token logprobs (Paulo et al., §4.1, Table 1).

Because the scorer sees non-activating contexts too, detection measures both precision and recall of an explanation (Paulo et al., §3.3.1).

## 2. The new score

delphi reports balanced accuracy `BA = (TPR + TNR) / 2`. We convert it and add a length factor:

```
J = 2·BA − 1                          Youden's J (Youden 1950): 0 = chance, 1 = perfect
L = len(tiktoken.get_encoding("o200k_base").encode(explanation.strip()))
S = max(J, 0) · 2^(−L / H)            H = half-life in tokens (pre-registered: 20)
```

Why each part:

- **J rather than BA.** A random explanation gets BA ≈ 0.5. If BA were multiplied by a length factor, a one-word random explanation would beat a long accurate one. With J, chance is 0, so brevity cannot be rewarded without accuracy above chance. The clip `max(J, 0)` does the same for below-chance explanations.
- **Multiplicative exponential factor.** `log S = log J − (ln 2 / H)·L`: each token costs the same amount of log-score. This is the two-part code of minimum description length: fit plus description length ([Rissanen 1978](https://doi.org/10.1016/0005-1098(78)90005-5)). [Ayonrinde et al. 2024](https://arxiv.org/abs/2410.11179) apply MDL to the SAE itself (latent code plus decoder); here it is applied to the text explanation. The score halves every H tokens, which makes H easy to state and tune.
- **Tokens of a fixed public tokenizer.** Using o200k_base ([tiktoken](https://github.com/openai/tiktoken)) means L does not depend on which LLM scores. Words and characters are reported as a robustness check.
- **H is a value judgement.** It is the exchange rate between accuracy and brevity, not something measured. At H = 20, a 10-token explanation keeps 71% of its J, and a 40-token one keeps 25%. The README reports a sweep over H.

## 3. Data: public exports plus verified negatives

Nothing is trained. [Neuronpedia](https://www.neuronpedia.org) publishes bulk exports of its SAE dashboards ([bucket index](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html)). For `gpt2-small/6-res-jb` ([Joseph Bloom's residual SAEs](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream); 24,576 latents at `blocks.6.hook_resid_pre`), each latent has:

- explanations from gpt-3.5-turbo and gpt-4o-mini, made with the OpenAI token–activation-pair method, plus a few human ones;
- activating 17-token contexts with per-token activations, covering the top activations and lower quantiles (at least 45 for every eligible latent);
- 25 contexts where the latent is exactly 0.

The activating contexts are the positives. The zero rows were the pre-registered negatives, but the pilot showed they are overlapping sliding windows over a few documents, shared across latents (PREREG deviation 2). So the negatives are random OpenWebText windows on which a forward pass of GPT-2 small and the same SAE (sae_lens `gpt2-small-res-jb`) gives exactly 0 at every displayed token; see [exp0](../reports/exp0_pilot_and_negatives.md). This forward pass is the only local model run.

Two of delphi's three detection few-shot examples are latents of this SAE (6048 and 9396), so those two latents are excluded; the third is latent 12654 of the layer-8 SAE (`delphi/scorers/classifier/prompts/detection_prompt.py`).

Per latent, the split is:

- **train:** the 20 strongest contexts. Only the Exp 2 explainer sees them.
- **test positives:** 25 of the rest, 5 from each of 5 activation-quantile bins.
- **test negatives:** 25 verified-zero random OpenWebText windows.

Every condition of a latent is scored on the same 50 contexts, in the same shuffled batches. The scorer's shuffle seed is fixed, so conditions differ only in the explanation.

## 4. Experiments

- **Exp 1 (existing explanations).** We score the gpt-3.5-turbo and gpt-4o-mini explanations, a top-token baseline (`the token "X"`) and a random-explanation control (another latent's explanation).
- **Exp 2 (incentive test).** Llama-3.1-70B, with delphi's default explainer prompt, writes explanations asked to be about 2, 5, 10, 20 or 40 words, plus one unconstrained.
  - If the length factor works as intended, J should rise with length and then flatten (H2), while S should peak at an intermediate length (H3).
  - We also choose, per latent, among 8 candidates by max J, by max S, or by shortest. Selection and evaluation are cross-fitted ([Chernozhukov et al. 2018](https://doi.org/10.1111/ectj.12097)) on disjoint halves of the test contexts, to avoid the winner's curse (H4).
- **Reliability.** On 50 latents we re-score with a different shuffle seed, and again with Llama-3.3-70B as the scorer.

## 5. Implementation map

| File | Role |
|---|---|
| `run.py` | Entry point; runs the phases in order. |
| `src/concise_interp/config.py` | All pre-registered constants. |
| `src/concise_interp/data.py` | Downloads the Neuronpedia export, checks eligibility, draws the samples. |
| `src/concise_interp/negatives.py` | Verified random negatives: GPT-2 small + the SAE on OpenWebText windows. |
| `src/concise_interp/records.py` | Neuronpedia rows → delphi `LatentRecord` (train/test/not_active). |
| `src/concise_interp/baselines.py` | Top-token baseline and random-explanation control. |
| `src/concise_interp/clients.py` | delphi's `OpenRouter` client, plus full prompt/response logging and provider pinning. |
| `src/concise_interp/scoring.py` | delphi `DetectionScorer` over all jobs; resumable JSONL output. |
| `src/concise_interp/explain.py` | delphi `DefaultExplainer` with a word-budget sentence (Exp 2). |
| `src/concise_interp/length_score.py` | **The method:** J, L, the length factor, S. |
| `src/concise_interp/analysis.py` | Metrics (delphi's own `compute_confusion`), bootstrap CIs, tests, cross-fit selection. |
| `src/concise_interp/figures.py` | Figures from results. |
| `src/concise_interp/render.py` | Fills templates/ with numbers from `results/main/numbers.json`. |
| `scripts/build_paper.sh` | Builds `paper/paper.pdf` from the rendered Markdown (pandoc + xelatex). |
