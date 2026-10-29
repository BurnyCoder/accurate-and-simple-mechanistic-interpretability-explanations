## Key terms

| Term | Meaning | In plain words |
|---|---|---|
| SAE latent / feature | One unit of a sparse autoencoder trained on a model's internal activations. | A "detector" inside the model that switches on for some pattern in text. |
| Explanation | A short text describing when a latent fires, e.g. `the token "notice"`. | A label that says what the detector detects. |
| Detection scoring | An LLM sees the explanation and 5 text snippets at a time, and says which snippets the latent fires on ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928)). | A quiz: given only the label, can someone pick out the right snippets? |
| BA (balanced accuracy) | (TPR + TNR) / 2: the mean of the hit rate on activating snippets and the correct-rejection rate on non-activating ones. | The quiz score, fair even if the classes are unbalanced. |
| J (Youden's J) | 2·BA − 1 = TPR − FPR. 0 = chance, 1 = perfect. | How far above guessing the label gets you. |
| L | Length of the explanation in tokens (o200k_base tokenizer). | How long the label is; one token is roughly ¾ of a word. |
| H (half-life) | The length at which the length factor halves the score. Pre-registered as 20 tokens. | How strongly length is penalized. |
| S | max(J, 0) · 2^(−L/H): **the proposed score**. | Accurate *and* short labels score high. |
| Top-token baseline | `the token "X"`, where X is the most common peak token in the strongest examples. | The laziest possible label. |
| Random control | Another latent's explanation. | A wrong label; it should score ~0. |
| Verified negatives | Random OpenWebText snippets on which the latent's activation is exactly 0 (checked by running GPT-2 small + the SAE). | Snippets where the detector is known to be off. |
| Cross-fitting | Choose an explanation using half of the test snippets, and measure it on the other half. | Don't grade yourself on the questions you used to choose your answer. |
| Bootstrap CI | 95% interval from resampling latents 10,000 times. | The range the average would plausibly fall in with a different set of latents. |
| Wilcoxon / Holm | Paired signed-rank test; Holm corrects p-values for running several tests. | Is the difference real, after accounting for running many tests? |
