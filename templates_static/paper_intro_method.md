# 1. Introduction

Sparse autoencoders (SAEs) decompose a language model's activations into many sparse latents ("features"), and each latent is usually given a natural-language explanation written by an LLM. Explanations are judged by *auto-interpretability* scores. An LLM is given only the explanation and asked to predict where the latent fires. Two ways of doing this are simulating per-token activations ([Bills et al. 2023](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html)) and classifying whole contexts as activating or not, called detection ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)). SAEBench's autointerp metric is a detection-style score similar to it, with gpt-4o-mini as the judge ([Karvonen et al. 2025](https://arxiv.org/abs/2503.09532), §3.2.2), and detection is implemented in EleutherAI's open-source [delphi](https://github.com/EleutherAI/delphi) library. A recent study of autointerp pipelines found it to be the most stable of the four autointerp metrics it compared, on two subject models ([van der Ben et al. 2026](https://arxiv.org/abs/2607.19386)).

These scores reward accuracy only. Paulo et al. state that "current scoring methods do not account for interpretation length". An explanation that lists every pattern it noticed is not penalized for its length, even though a human reader, or a downstream circuit analysis, pays for every word. In other settings, GPT-4 as a judge prefers longer answers more than human raters do ([Saito et al. 2023](https://arxiv.org/abs/2310.10076)), and leaderboards have had to correct for length ([Dubois et al. 2024](https://arxiv.org/abs/2404.04475)).

We make the smallest change that puts brevity into the objective. We multiply the above-chance detection score by a factor that halves every H tokens:

$$S = \max(J, 0)\cdot 2^{-L/H},\qquad J = 2\,\mathrm{BA} - 1.$$

Here BA is delphi's balanced accuracy and L is the explanation's length in tokens. In log space, $\log S = \log J - (\ln 2/H)\,L$: a constant cost per token. This is analogous to, not an instance of, the two-part code of minimum description length ([Rissanen 1978](https://doi.org/10.1016/0005-1098(78)90005-5)): the length term prices the explanation, but $-\log J$ is a fit penalty, not the code length of the data given the explanation. Ayonrinde et al. ([2024](https://arxiv.org/abs/2410.11179)) apply MDL to the SAE's latent code and decoder; we apply the same fit-plus-length idea to the text explanation.

To keep custom work minimal, we use existing explanations and activation data from [Neuronpedia's](https://www.neuronpedia.org) public export for GPT-2 small, and delphi's scorer and explainer unchanged.

# 2. Background: detection scoring

For a latent and an explanation, the scorer LLM sees contexts in shuffled batches of 5 (Paulo et al., App. D.1). Some contexts contain the latent's activation and some do not. For each context it answers 1 or 0 ("does this context match the explanation?"). Paulo et al. use 100 activating contexts (10 per activation decile) and 100 random non-activating contexts (§4.2, App. D.4, D.6), with Llama-3.1-70B-Instruct as both explainer and scorer (§3.2, App. D.6; their App. F states 4-bit quantization for its experiments). Because non-activating contexts are included, the score reflects both recall (do activating contexts match?) and precision (do non-activating ones not match?) of the explanation (§3.3.1).

We use delphi's `DetectionScorer` with its default prompt and three few-shot examples. Two of those examples are latents of the very SAE we study (6048 and 9396), so we exclude them; the third comes from the layer-8 SAE.

# 3. Method

**Score.** delphi reports balanced accuracy $\mathrm{BA} = (\mathrm{TPR} + \mathrm{TNR})/2$, computed with its own `compute_confusion` and `compute_classification_metrics`. We use Youden's $J = 2\,\mathrm{BA} - 1 = \mathrm{TPR} - \mathrm{FPR}$ ([Youden 1950](https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3)), because chance is then 0. If the length factor multiplied BA instead, a short random explanation (BA ≈ 0.5) would beat a long accurate one. The clip $\max(J, 0)$ keeps below-chance explanations from gaining by being short.

L is the token count of the stripped explanation under tiktoken's `o200k_base` encoding. That tokenizer is fixed, so L does not depend on the scorer. Words and characters are checked as alternatives.

The half-life H is the exchange rate between accuracy and brevity. We pre-registered H = 20 tokens: a 10-token explanation keeps 71% of its J and a 40-token one keeps 25% (Neuronpedia's explanations have a median of {{by_condition.gpt4omini.tokens_median:.0f}} tokens). We report a sweep over H.
