# Accurate and simple mechanistic-interpretability explanations

**Length-penalized detection scoring for sparse-autoencoder (SAE) feature explanations.**

Detection scoring ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)), as implemented in EleutherAI's [delphi](https://github.com/EleutherAI/delphi), is the standard way to score SAE feature explanations, and SAEBench's autointerp metric is built on it ([Karvonen et al. 2025](https://arxiv.org/abs/2503.09532)). It asks an LLM to pick which text snippets match an explanation, but the score rewards accuracy only. This repo multiplies the above-chance detection score by a length factor, so a short explanation beats a long one of equal accuracy:

```
S = max(J, 0) · 2^(−L / H)
J = 2·balanced_accuracy − 1      (0 = chance, 1 = perfect)
L = explanation length in tokens (tiktoken o200k_base)
H = half-life in tokens (pre-registered default 20)
```

The explanations and activation examples come from [Neuronpedia's public exports](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html) for GPT-2 small (`6-res-jb`), so no model or SAE has to be run.

> Status: work in progress. The pre-registration is in [PREREG.md](PREREG.md). Results, figures and the full write-up will be added once the experiments have run.
