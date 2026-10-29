# Related work

**Auto-interpretability scoring.**

- [Bills et al. (2023)](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) introduced explain-then-simulate scoring for neurons.
- [Paulo et al. (2024)](https://arxiv.org/abs/2410.13928) introduced cheaper scorers (detection, fuzzing, surprisal, embedding, intervention) and the delphi library. They note that "current scoring methods do not account for interpretation length".
- SAEBench ([Karvonen et al. 2025](https://arxiv.org/abs/2503.09532)) adopted their LLM-as-judge detection metric.
- [van der Ben et al. (2026)](https://arxiv.org/abs/2607.19386) show that pipeline choices dominate autointerp score variance, with detection the most stable metric.

None of these scores penalizes length. Our score is a one-line extension of detection that does.

**Simplicity and description length.**

- The minimum description length principle ([Rissanen 1978](https://doi.org/10.1016/0005-1098(78)90005-5)) scores a hypothesis by the bits needed to state it plus the bits needed to encode the data given it.
- [Ayonrinde, Pearce and Sharkey (2024)](https://arxiv.org/abs/2410.11179) apply MDL to the SAE's latent code, seeking explanations that are "both accurate and concise".
- [Ayonrinde and Jaburi (2025)](https://arxiv.org/abs/2505.01372) list explanatory simplicity among the virtues of mechanistic explanations.

We apply the same fit-plus-length trade-off to the natural-language explanation itself, with a single interpretable exchange rate H.

**Length bias in LLM evaluation.**

- LLM judges prefer longer answers ([Saito et al. 2023](https://arxiv.org/abs/2310.10076)).
- Length-controlled AlpacaEval removes this bias by regression ([Dubois et al. 2024](https://arxiv.org/abs/2404.04475)).

Our goal is the opposite of length *control*. We deliberately *reward* brevity, because shorter explanations are cheaper for humans and downstream tools to use.

**Data.**

- The SAEs are [Bloom (2024)](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream)'s open residual-stream SAEs for GPT-2 small.
- Explanations and dashboards are from [Neuronpedia](https://www.neuronpedia.org).
- Random contexts are from [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext).
- The SAE is loaded with [sae_lens](https://github.com/jbloomAus/SAELens).
