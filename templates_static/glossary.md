## Key terms

| Term | Meaning | In plain words |
|---|---|---|
| SAE latent / feature | One unit of a sparse autoencoder trained on a model's internal activations. | A "detector" inside the model that switches on for some pattern in text. |
| Explanation | A short text describing when a latent fires, e.g. `the token "notice"`. | A label that says what the detector detects. |
| Detection scoring | An LLM sees the explanation and 5 text snippets at a time, and says which snippets the latent fires on ([Paulo et al. 2024](https://arxiv.org/abs/2410.13928), §3.3.1, App. D.1). | A quiz: given only the label, can someone pick out the right snippets? |
| BA (balanced accuracy) | (TPR + TNR) / 2: the mean of the hit rate on activating snippets and the correct-rejection rate on non-activating ones. | The quiz score, fair even if the classes are unbalanced. |
| J (Youden's J) | 2·BA − 1 = TPR − FPR. 0 = chance, 1 = perfect ([Youden 1950](https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3)). | How far above guessing the label gets you. |
| L | Length of the explanation in tokens (o200k_base tokenizer). | How long the label is; in these explanations one token is {{exploratory.unit_robustness_spearman.words_per_token:.2f}} words on average. |
| H (half-life) | The length at which the length factor halves the score. Pre-registered as 20 tokens. | How strongly length is penalized. |
| S | max(J, 0) · 2^(−L/H): **the proposed score**. | Accurate *and* short labels score high. |
| Top-token baseline | `the token "X"`, where X is the most common peak token in the strongest examples. | The laziest possible label. |
| Random control | Another latent's explanation. | A wrong label; it should score ~0. |
| Verified negatives | Random OpenWebText snippets on which the latent's activation is exactly 0 (checked by running GPT-2 small + the SAE). | Snippets where the detector is known to be off. |
| Cross-fitting | Choose an explanation using half of the test snippets, and measure it on the other half, then swap ([Chernozhukov et al. 2018](https://doi.org/10.1111/ectj.12097)). | Don't grade yourself on the questions you used to choose your answer. |
| Bootstrap CI | 95% percentile interval from resampling latents 10,000 times ([Efron 1979](https://doi.org/10.1214/aos/1176344552)). | The range the average would plausibly fall in with a different set of latents. |
| Wilcoxon / Holm | Paired signed-rank test ([Wilcoxon 1945](https://doi.org/10.2307/3001968)); Holm corrects p-values for running several tests ([Holm 1979](https://www.jstor.org/stable/4615733)). | Is the difference real, after accounting for running many tests? |
