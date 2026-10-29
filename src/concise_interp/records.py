"""Neuronpedia activation rows -> delphi `LatentRecord` (PREREG section 3).

Global context: Neuronpedia exports, per latent, ~54 activating 17-token contexts (top and
quantile examples) and ~25 contexts where the latent is exactly 0. delphi's scorers consume
`LatentRecord.test` (activating) and `LatentRecord.not_active` (non-activating), and its
explainers consume `LatentRecord.train`. This module does that conversion with the
pre-registered split, so no model or SAE has to be run.
"""

import random
from typing import NamedTuple

import numpy as np
import torch
from delphi.latents import ActivatingExample, LatentRecord, NonActivatingExample
from delphi.latents.latents import Latent
from delphi.latents.samplers import normalize_activations

from . import config


class Split(NamedTuple):
    train: list[dict]  # top-N_TRAIN activating rows, highest first
    test_pos: list[tuple[int, dict]]  # (quantile bin, row); bin 0 = highest activations
    test_neg: list[dict]  # rows with maxValue == 0


def split_examples(rows: list[dict], rng: random.Random) -> Split:
    """Top-20 to train, 5 positives from each of 5 equal-count quantile bins of the rest,
    and 25 zero-activation rows as negatives."""
    # Stable sort keeps export order among ties, so the split is deterministic.
    activating = sorted((r for r in rows if r["maxValue"] > 0), key=lambda r: -r["maxValue"])
    zeros = [r for r in rows if r["maxValue"] == 0]
    train, rest = activating[: config.N_TRAIN], activating[config.N_TRAIN :]
    per_bin = config.N_TEST_POS // config.N_QUANTILE_BINS
    # np.array_split on indices gives equal-count bins in descending-activation order.
    bins = np.array_split(np.arange(len(rest)), config.N_QUANTILE_BINS)
    test_pos = [(q, rest[i]) for q, idx in enumerate(bins) for i in rng.sample(list(idx), per_bin)]
    test_neg = rng.sample(zeros, config.N_TEST_NEG)
    return Split(train, test_pos, test_neg)


def _placeholder_ids(n: int) -> torch.Tensor:
    # Neuronpedia exports decoded token strings only. delphi's DetectionScorer and
    # DefaultExplainer read `str_tokens` and `activations`, never `tokens`, so ids are -1.
    return torch.full((n,), -1, dtype=torch.long)


def _activating(row: dict, quantile: int = 0) -> ActivatingExample:
    return ActivatingExample(
        tokens=_placeholder_ids(len(row["tokens"])),
        activations=torch.tensor(row["values"], dtype=torch.float32),
        str_tokens=list(row["tokens"]),
        quantile=quantile,
    )


def build_record(latent: int, split: Split) -> LatentRecord:
    """A delphi record whose train/test/not_active lists are exactly the pre-registered split."""
    record = LatentRecord(latent=Latent(f"{config.MODEL_ID}/{config.SOURCE_ID}", latent))
    train = [_activating(r) for r in split.train]
    # delphi quantises explainer activations to 0..10 relative to the latent's max
    # (delphi/latents/samplers.py::normalize_activations); the top train row is that max.
    record.train = normalize_activations(train, max_activation=split.train[0]["maxValue"])
    record.examples = record.train
    record.test = [_activating(r, quantile=q) for q, r in split.test_pos]
    record.not_active = [
        NonActivatingExample(
            tokens=_placeholder_ids(len(r["tokens"])),
            activations=torch.zeros(len(r["tokens"])),
            str_tokens=list(r["tokens"]),
            distance=-1.0,  # delphi's value for random (non-neighbour) negatives
        )
        for r in split.test_neg
    ]
    return record


def sample_key(str_tokens: list[str], activating: bool) -> tuple[str, bool]:
    """How a scored sample is matched back to its test row: its text plus its label."""
    return "".join(str_tokens), bool(activating)


def sample_ids_for(split: Split) -> dict[tuple[str, bool], str]:
    """Stable ids p0..p24 / n0..n24, used to split test samples into cross-fit halves."""
    ids = {sample_key(r["tokens"], True): f"p{i}" for i, (_, r) in enumerate(split.test_pos)}
    ids |= {sample_key(r["tokens"], False): f"n{i}" for i, r in enumerate(split.test_neg)}
    return ids
