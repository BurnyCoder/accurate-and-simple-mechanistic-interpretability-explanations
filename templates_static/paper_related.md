# Related work

**Auto-interpretability scoring.**

- [Bills et al. (2023)](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) introduced explain-then-simulate scoring for neurons.
- [Paulo et al. (2024)](https://arxiv.org/abs/2410.13928) introduced cheaper scorers (detection, fuzzing, surprisal, embedding, intervention) and the delphi library. They note that "current scoring methods do not account for interpretation length".
- SAEBench ([Karvonen et al. 2025](https://arxiv.org/abs/2503.09532)) uses an LLM-judge metric similar to their detection score, with gpt-4o-mini judging 14 sequences per latent (§3.2.2, App. D).
- [van der Ben et al. (2026)](https://arxiv.org/abs/2607.19386) show that pipeline choices (corpus, example draw, explainer, paraphrase; the scorer was held fixed) account for more autointerp score variance than SAE architecture, and report detection as the most stable of the four metrics they compared, on two subject models. In an appendix they cap explanations at ten words; their text reports lower detection scores for one subject model (Apertus-8B) and little effect for the other (App. E.8; the figure caption there disagrees on detection).

None of these scores penalizes length. Our score is a one-line extension of detection that does.

**Simplicity and description length.**

- The minimum description length principle ([Rissanen 1978](https://doi.org/10.1016/0005-1098(78)90005-5)) scores a hypothesis by the bits needed to state it plus the bits needed to encode the data given it.
- [Ayonrinde, Pearce and Sharkey (2024)](https://arxiv.org/abs/2410.11179) use MDL to compare SAEs as explanations of activations: among SAEs within a reconstruction-error tolerance, prefer the one whose latent code needs the fewest bits (the decoder is a constant cost they neglect). They motivate explanations that are "both accurate and concise".
- [Ayonrinde and Jaburi (2025)](https://arxiv.org/abs/2505.01372) list explanatory simplicity among the virtues of mechanistic explanations.

We apply a fit-plus-length trade-off to the natural-language explanation itself, with a single interpretable exchange rate H; unlike MDL-SAEs, accuracy is a multiplicative factor, not a constraint.

**Length bias in LLM evaluation.**

- In the setting of [Saito et al. (2023)](https://arxiv.org/abs/2310.10076), GPT-4 as a judge preferred longer answers more than human raters did.
- Length-controlled AlpacaEval reduces this bias with a regression (a generalized linear model) that predicts preferences at zero length difference ([Dubois et al. 2024](https://arxiv.org/abs/2404.04475)).

Our goal is the opposite of length *control*. We deliberately *reward* brevity, because shorter explanations are cheaper for humans and downstream tools to use.

**Data.**

- The SAEs are [Bloom (2024)](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream)'s open residual-stream SAEs for GPT-2 small ([Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).
- Explanations and dashboards are from [Neuronpedia](https://www.neuronpedia.org).
- Random contexts are from [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext).
- The SAE is loaded with [sae_lens](https://github.com/jbloomAus/SAELens).
