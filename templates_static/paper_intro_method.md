# 1. Introduction

Sparse autoencoders (SAEs) decompose a language model's activations into many sparse latents ("features"), and each latent is usually given a natural-language explanation written by an LLM. Explanations are judged by *auto-interpretability* scores. An LLM is given only the explanation and asked to predict where the latent fires. Two ways of doing this are simulating per-token activations ([Bills et al. 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)) and classifying whole contexts as activating or not, called detection ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)). Detection scoring is the basis of SAEBench's autointerp metric ([Karvonen et al. 2025](https://arxiv.org/abs/2503.09532)) and is implemented in EleutherAI's open-source [delphi](https://github.com/EleutherAI/delphi) library. A recent study of autointerp pipelines found it to be the most stable autointerp metric ([van der Ben et al. 2026](https://arxiv.org/abs/2607.19386)).

These scores reward accuracy only. Paulo et al. state that "current scoring methods do not account for interpretation length". An explanation that lists every pattern it noticed is not penalized for its length, even though a human reader, or a downstream circuit analysis, pays for every word. LLM judges in other settings are known to prefer longer outputs ([Saito et al. 2023](https://arxiv.org/abs/2310.10076)), and leaderboards have had to correct for length ([Dubois et al. 2024](https://arxiv.org/abs/2404.04475)).

We make the smallest change that puts brevity into the objective. We multiply the above-chance detection score by a factor that halves every H tokens:

$$S = \max(J, 0)\cdot 2^{-L/H},\qquad J = 2\,\mathrm{BA} - 1.$$

Here BA is delphi's balanced accuracy and L is the explanation's length in tokens. In log space, $\log S = \log J - (\ln 2/H)\,L$: a constant cost per token. This is the two-part code of minimum description length ([Rissanen 1978](https://doi.org/10.1016/0005-1098(78)90005-5)), applied to the text explanation rather than to the SAE code as in [Ayonrinde et al. 2024](https://arxiv.org/abs/2410.11179).

To keep custom work minimal, we use existing explanations and activation data from [Neuronpedia's](https://www.neuronpedia.org) public export for GPT-2 small, and delphi's scorer and explainer unchanged.

# 2. Background: detection scoring

For a latent and an explanation, the scorer LLM sees contexts in shuffled batches of 5. Some contexts contain the latent's activation and some do not. For each context it answers 1 or 0 ("does this context match the explanation?"). Paulo et al. use 100 activating contexts (10 per activation decile) and 100 random non-activating contexts, with Llama-3.1-70B-Instruct (4-bit) as both explainer and scorer. Because non-activating contexts are included, the score reflects both recall (do activating contexts match?) and precision (do non-activating ones not match?) of the explanation.

We use delphi's `DetectionScorer` with its default prompt and three few-shot examples. Two of those examples are latents of the very SAE we study (6048 and 9396), so we exclude them.

# 3. Method

**Score.** delphi reports balanced accuracy $\mathrm{BA} = (\mathrm{TPR} + \mathrm{TNR})/2$, computed with its own `compute_confusion` and `compute_classification_metrics`. We use Youden's $J = 2\,\mathrm{BA} - 1 = \mathrm{TPR} - \mathrm{FPR}$, because chance is then 0. If the length factor multiplied BA instead, a short random explanation (BA ≈ 0.5) would beat a long accurate one. The clip $\max(J, 0)$ keeps below-chance explanations from gaining by being short.

L is the token count of the stripped explanation under tiktoken's `o200k_base` encoding. That tokenizer is fixed, so L does not depend on the scorer. Words and characters are checked as alternatives.

The half-life H is the exchange rate between accuracy and brevity. We pre-registered H = 20 tokens: a typical 10-token explanation keeps 71% of its J and a 40-token one keeps 25%. We report a sweep over H.
