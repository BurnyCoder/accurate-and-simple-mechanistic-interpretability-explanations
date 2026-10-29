"""Baseline and control explanations (PREREG section 5).

Global context: the top-token baseline is the shortest explanation that could plausibly be
accurate (many SAE latents are single-token detectors). The random control (another latent's
explanation) checks that the scorer gives ~0 to explanations unrelated to the latent; delphi
has the same control as `random_explanation_loader`.
"""

import random
from collections import Counter


def top_token_explanation(train_rows: list[dict]) -> str:
    """`the token "X"`: X is the most frequent max-activating token in the train rows.
    Ties go to the token with the larger summed max activation."""
    counts: Counter[str] = Counter()
    weight: Counter[str] = Counter()
    for row in train_rows:
        raw = row["tokens"][row["values"].index(max(row["values"]))]
        # Strip so " notice" and "notice" count as the same token (keep whitespace-only tokens).
        token = raw.strip() or raw
        counts[token] += 1
        weight[token] += row["maxValue"]
    best = max(counts, key=lambda t: (counts[t], weight[t]))
    return f'the token "{best}"'


def random_derangement(latents: list[int], seed: int) -> dict[int, int]:
    """Map each latent to a different latent, uniformly over derangements (by rejection)."""
    rng = random.Random(seed)
    targets = list(latents)
    while True:
        rng.shuffle(targets)
        if all(a != b for a, b in zip(latents, targets)):
            return dict(zip(latents, targets))
