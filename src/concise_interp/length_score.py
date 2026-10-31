"""The method: a length factor on top of detection scoring.

Global context: delphi's DetectionScorer gives a balanced accuracy (BA) per explanation
(Paulo et al. 2024, https://arxiv.org/abs/2410.13928). Paulo et al. note that "current
scoring methods do not account for interpretation length". This module turns BA into an
above-chance score J and multiplies it by a factor that halves every `half_life` tokens:

    S = max(J, 0) * 2^(-L / H),   J = 2 * BA - 1

In log space, log S = log J - (ln 2 / H) * L: every token costs the same amount of log-score,
the minimum-description-length trade-off between fit and description length
(Rissanen 1978; for SAEs, Ayonrinde et al. 2024, https://arxiv.org/abs/2410.11179).
"""

import math
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # o200k_base is the public GPT-4o tokenizer (https://github.com/openai/tiktoken);
    # it is fixed and model-agnostic here, so lengths do not depend on the scorer LLM.
    return tiktoken.get_encoding("o200k_base")


def n_tokens(text: str) -> int:
    """Explanation length L in o200k_base tokens, ignoring surrounding whitespace."""
    # disallowed_special=(): an explanation may literally mention "<|endoftext|>"; count it as
    # plain text instead of raising (tiktoken's default refuses special-token strings).
    return len(_encoding().encode(text.strip(), disallowed_special=()))


def n_words(text: str) -> int:
    """Whitespace-separated word count (robustness check for the length unit)."""
    return len(text.split())


def n_chars(text: str) -> int:
    """Character count without surrounding whitespace (robustness check)."""
    return len(text.strip())


def youden_j(balanced_accuracy: float) -> float:
    """Youden's J = TPR + TNR - 1 = 2*BA - 1, so chance is 0 and perfect is 1
    (Youden 1950, https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3)."""
    return 2.0 * balanced_accuracy - 1.0


def length_factor(length: float, half_life: float) -> float:
    """2^(-L/H); H = inf turns the factor off, leaving S = max(J, 0) (detection scoring clipped at 0)."""
    if math.isinf(half_life):
        return 1.0
    return 2.0 ** (-length / half_life)


def length_penalized_score(j: float, length: float, half_life: float) -> float:
    """S = max(J, 0) * 2^(-L/H). Clipping at 0 stops below-chance explanations from
    earning credit for brevity, and keeps S non-negative so log S is defined when J > 0."""
    return max(j, 0.0) * length_factor(length, half_life)
