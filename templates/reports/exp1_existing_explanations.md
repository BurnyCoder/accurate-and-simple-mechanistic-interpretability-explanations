# Experiment 1: existing Neuronpedia explanations, a baseline and a control

<!-- Generated from templates/reports/exp1_existing_explanations.md; numbers come from results/main/numbers.json. -->

**Question.** How do existing explanations score under plain detection scoring (J) and under the length-penalized score (S)? Does the scorer behave sensibly: chance for unrelated explanations, above chance for real ones?

## Setup

- **Latents:** {{n_latents}} latents of `gpt2-small/6-res-jb`, sampled at random (seed 0) from the {{data.n_eligible}} eligible ones.
- **Conditions per latent:**
  - `gpt35` and `gpt4omini`: Neuronpedia's explanations, made with OpenAI's token–activation-pair explainer method.
  - `top_token` (baseline): `the token "X"`, with X the most frequent peak token among the 20 strongest contexts.
  - `random` (control): another sampled latent's gpt-4o-mini explanation, via a derangement.
  - Human explanations where Neuronpedia has them. Only {{exploratory.n_humans}} exist in the sample, so they are descriptive only.
- **Scoring:** delphi `DetectionScorer`, Llama-3.1-70B-Instruct, 25 activating + 25 verified-negative contexts in shuffled batches of 5. All conditions of a latent see the same batches.

## Results

{{table:conditions}}

(The table also lists the Exp 2 conditions for comparison.)

![J and S per explanation source](../figures/fig1_conditions.png)

**H1 (validity of the base scorer): {{H1.supported}}.**

- The random control scores J = {{H1.random.mean:.3f}} [{{H1.random.lo:.3f}}, {{H1.random.hi:.3f}}]. That is within the pre-registered |mean| < 0.05, but its CI lies just below 0: unrelated explanations make the scorer flag negatives slightly more often than positives.
- Neuronpedia explanations score well above chance: gpt-3.5-turbo J = {{H1.gpt35.mean:.3f}} [{{H1.gpt35.lo:.3f}}, {{H1.gpt35.hi:.3f}}], gpt-4o-mini J = {{H1.gpt4omini.mean:.3f}} [{{H1.gpt4omini.lo:.3f}}, {{H1.gpt4omini.hi:.3f}}].

**Exploratory: gpt-4o-mini vs gpt-3.5-turbo.**

- gpt-4o-mini's explanations are more accurate by J = {{exploratory.gpt4omini_minus_gpt35_J.mean:+.3f}} [{{exploratory.gpt4omini_minus_gpt35_J.lo:+.3f}}, {{exploratory.gpt4omini_minus_gpt35_J.hi:+.3f}}] (two-sided Wilcoxon p = {{exploratory.gpt4omini_minus_gpt35_J.two_sided_p:.3f}}).
- Both have the same median length ({{by_condition.gpt35.tokens_median:.0f}} vs {{by_condition.gpt4omini.tokens_median:.0f}} tokens), so the length factor does not reverse the order: ΔS = {{exploratory.gpt4omini_minus_gpt35_S.mean:+.3f}} [{{exploratory.gpt4omini_minus_gpt35_S.lo:+.3f}}, {{exploratory.gpt4omini_minus_gpt35_S.hi:+.3f}}].
- The pre-registered exploratory question was whether they swap rank. They don't.

**The top-token baseline.** `the token "X"` reaches J = {{by_condition.top_token.j_mean:.3f}} in a median of {{by_condition.top_token.tokens_median:.0f}} tokens. That is comparable to gpt-3.5-turbo's J ({{by_condition.gpt35.j_mean:.3f}}). Many layer-6 GPT-2 latents fire mainly on one token.

Under S, the baseline ({{by_condition.top_token.s_mean:.3f}}) beats gpt-3.5-turbo ({{by_condition.gpt35.s_mean:.3f}}) and matches gpt-4o-mini ({{by_condition.gpt4omini.s_mean:.3f}}). An honest length-aware score *should* credit a one-token label when that is all the latent does. The flip side is that the score gives LLM explainers no extra credit for nuance that the detection test cannot measure.

**Human explanations (descriptive only, n = {{exploratory.n_humans}}).**

- Latent {{exploratory.humans.0.latent}}, "{{exploratory.humans.0.explanation}}": J = {{exploratory.humans.0.j:.2f}}, {{exploratory.humans.0.n_tokens}} tokens.
- Latent {{exploratory.humans.1.latent}}, "{{exploratory.humans.1.explanation}}": J = {{exploratory.humans.1.j:.2f}}, {{exploratory.humans.1.n_tokens}} tokens.

## Interpretation and limitations

- The base scorer passes its sanity check, so differences in J are meaningful.
- Neuronpedia's explainers saw the top activations. Holding out the 20 strongest contexts reduces but cannot rule out overlap with our test positives. The Exp 2 explanations are fully held out.
- 50 test contexts per explanation make single-explanation J noisy, so the conclusions rest on means over {{n_latents}} latents.
