"""Test-set audits: what the scorer actually saw, beyond the pre-registered split.

Global context: called from `analysis.run_analysis` on the same per-latent splits the scorer
used (`experiment.prepare_records`). The numbers qualify three statements in the paper:
- Exp 2 explanations are held out: how many test positives overlap a train context?
- positives and negatives differ only in the latent: Neuronpedia's windows can cross document
  boundaries (`<|endoftext|>`), our random negatives are one window inside one document;
- which activations are tested: test positives come from the export's rows below the top 20,
  so their activation range is set by Neuronpedia's export, not by the corpus-wide deciles
  Paulo et al. (2024, https://arxiv.org/abs/2410.13928, §4.2) sample from.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from .records import Split, sample_key

EOT = "<|endoftext|>"  # GPT-2's document separator, as Neuronpedia exports it
OVERLAP_TOKENS = 8  # a shared run this long means two windows come from the same passage


def longest_shared_run(a: list[str], b: list[str]) -> int:
    """Length of the longest run of consecutive tokens that `a` and `b` share
    (longest common substring by dynamic programming over token positions)."""
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        # cur[j] = length of the shared run ending at x and b[j-1]
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def split_checks(splits: dict[int, Split], sample_ids: dict[int, dict]) -> tuple[dict, dict[int, set[str]]]:
    """Train/test overlap, <|endoftext|> shares and activation ratios over all latents' splits.
    Also returns, per latent, the sample ids of test positives that contain <|endoftext|>."""
    n_pos = n_neg = n_neg_eot = 0
    overlap_latents: list[int] = []
    eot_ids: dict[int, set[str]] = {}
    ratios: defaultdict[int, list[float]] = defaultdict(list)
    for latent, split in splits.items():
        top = split.train[0]["maxValue"]  # train is sorted, so this is the latent's export maximum
        ids = sample_ids[latent]
        eot_ids[latent] = {ids[sample_key(r["tokens"], True)] for _, r in split.test_pos if EOT in r["tokens"]}
        for q, row in split.test_pos:
            n_pos += 1
            ratios[q].append(row["maxValue"] / top)
            if max(longest_shared_run(row["tokens"], t["tokens"]) for t in split.train) >= OVERLAP_TOKENS:
                overlap_latents.append(latent)
        n_neg += len(split.test_neg)
        n_neg_eot += sum(EOT in r["tokens"] for r in split.test_neg)
    n_eot = sum(len(v) for v in eot_ids.values())
    summary = {
        "n_test_positives": n_pos,
        "train_overlap": {"min_shared_tokens": OVERLAP_TOKENS, "n": len(overlap_latents),
                          "pct": 100 * len(overlap_latents) / n_pos, "n_latents": len(set(overlap_latents))},
        "eot": {"pct_positives": 100 * n_eot / n_pos, "pct_negatives": 100 * n_neg_eot / n_neg,
                "n_latents": sum(bool(v) for v in eot_ids.values())},
        # Activation of each test positive relative to its latent's top exported activation, per
        # quantile bin (bin 0 = strongest).
        "activation_ratio": {"median_by_bin": [float(np.median(ratios[q])) for q in sorted(ratios)],
                             "min": float(min(min(v) for v in ratios.values()))},
    }
    return summary, eot_ids


def positive_rates(df: pd.DataFrame, condition: str, ids: dict[int, set[str]]) -> dict:
    """Share of activating samples that `condition`'s explanations got a "yes" on, for the listed
    ids vs all other positives. For the random control, a gap would mean the scorer reacts to
    something in the listed contexts other than the explanation."""
    listed, other = [], []
    for r in df[df.condition == condition].itertuples():
        for s in r.samples:
            if s["activating"] and s["prediction"] is not None:
                (listed if s["id"] in ids.get(r.latent, ()) else other).append(s["prediction"])
    return {"listed": float(np.mean(listed)), "other": float(np.mean(other)), "n_listed": len(listed)}
