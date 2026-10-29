---
license: cc-by-4.0
language:
- en
pretty_name: Length-penalized detection scores for GPT-2 small SAE feature explanations
tags:
- interpretability
- sparse-autoencoders
- autointerp
- mechanistic-interpretability
size_categories:
- 1K<n<10K
configs:
- config_name: scores
  data_files: results/main/scores.jsonl
- config_name: generated_explanations
  data_files: results/main/generated_explanations.jsonl
---

# Length-penalized detection scores for GPT-2 small SAE feature explanations

<!-- Generated from templates/hf/README.md; numbers come from results/main/numbers.json. -->

Data from the project [accurate-and-simple-mechanistic-interpretability-explanations](https://github.com/BurnyCoder/accurate-and-simple-mechanistic-interpretability-explanations) (author: Libor "Burny" Burian). It contains 2002 natural-language explanations of 200 SAE latents of GPT-2 small (`gpt2-small-res-jb`, layer 6), each scored with EleutherAI delphi's detection scorer (Llama-3.1-70B-Instruct), with every per-context prediction.

The project's new score multiplies the above-chance detection score by a length factor: `S = max(J, 0) · 2^(−L/H)`, where `J = 2·BA − 1` and L is in o200k_base tokens. S is not stored; it is recomputed from the predictions and the explanation text.

## Files

| Path | Content |
|---|---|
| `results/main/scores.jsonl` | One row per (latent, condition): `explanation`, `scorer_model`, `shuffle_seed`, and `samples`, the 50 test contexts with `id`, `activating`, `prediction` and `quantile`. |
| `results/main/scores_retest.jsonl`, `scores_altscorer.jsonl` | Re-scoring of 50 latents: shuffle seed 43, and Llama-3.3-70B as the scorer. |
| `results/main/generated_explanations.jsonl` | Llama-3.1-70B explanations at word budgets ~2, 5, 10, 20, 40, and unconstrained. |
| `results/main/numbers.json` | All analysis numbers (pre-registered hypothesis tests). |
| `data/processed/` | The Neuronpedia rows of the sampled latents, the sample lists, and the verified negative contexts. |
| `logs/` | Every LLM prompt and response, untruncated. |

## Conditions

- `gpt35` and `gpt4omini`: Neuronpedia's explanations.
- `top_token`: a baseline of the form `the token "X"`.
- `random`: another latent's explanation (control).
- `gen_b{2,5,10,20,40}` and `gen_unconstrained`: Llama-3.1-70B explanations from delphi's explainer prompt, with a word budget.
- `human*`: Neuronpedia human explanations.

## Main results

- Plain detection (J) prefers longer explanations: J goes from 0.464 at ~2 words to 0.577 at ~40 words.
- At H = 20, mean S is highest for the shortest explanations, so the pre-registered interior optimum was falsified.
- Choosing explanations by S instead of J makes them 39% shorter, for 0.031 held-out J.

See the repository README and paper for the full write-up.

## Sources and licenses

- Explanations and activations come from [Neuronpedia's public exports](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html).
- Negatives are windows of [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) (CC0).
- The SAE is by [Joseph Bloom](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted).
- Scorer prompts are from [delphi](https://github.com/EleutherAI/delphi) (Apache-2.0).
