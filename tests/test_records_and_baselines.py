"""Tests for turning Neuronpedia activation rows into delphi records, and for the baselines.

Global context: the detection scorer only sees what `records.build_record` puts into
`LatentRecord.test` and `LatentRecord.not_active`, so the split must be exactly the
pre-registered one (top-20 train, 25 stratified positives, 25 zero-activation negatives).
"""

import random

import pandas as pd
import pytest
from delphi.log.result_analysis import compute_classification_metrics, compute_confusion

from concise_interp import config
from concise_interp.baselines import random_derangement, top_token_explanation
from concise_interp.records import build_record, sample_ids_for, split_examples


def _row(max_value: float, i: int) -> dict:
    # A 4-token context whose 3rd token carries the activation, like Neuronpedia's rows.
    values = [0.0, 0.0, max_value, 0.0]
    return {"tokens": ["a", f" w{i}", " hit", "."], "values": values, "maxValue": max_value}


@pytest.fixture
def rows():
    return [_row(float(100 - i), i) for i in range(60)]


NEGATIVES = [["x", f" n{i}", "."] for i in range(40)]


def test_split_uses_top_train_and_stratified_test(rows):
    split = split_examples(rows, NEGATIVES, random.Random(0))
    assert len(split.train) == config.N_TRAIN
    assert len(split.test_pos) == config.N_TEST_POS
    assert len(split.test_neg) == config.N_TEST_NEG
    # Train is exactly the 20 highest activations.
    assert [r["maxValue"] for r in split.train] == [float(100 - i) for i in range(20)]
    # No train row leaks into the test positives; negatives are all exactly zero.
    train_ids = {id(r) for r in split.train}
    assert not any(id(r) in train_ids for _, r in split.test_pos)
    assert all(r["maxValue"] == 0.0 for r in split.test_neg)
    # Stratification: each of the 5 quantile bins contributes 25 / 5 = 5 positives.
    bins = pd.Series([q for q, _ in split.test_pos]).value_counts()
    assert sorted(bins.index) == list(range(config.N_QUANTILE_BINS)) and set(bins) == {5}


def test_split_is_deterministic_given_seed(rows):
    a = split_examples(rows, NEGATIVES, random.Random(0))
    b = split_examples(rows, NEGATIVES, random.Random(0))
    assert [r["maxValue"] for _, r in a.test_pos] == [r["maxValue"] for _, r in b.test_pos]


def test_build_record_matches_delphi_types(rows):
    split = split_examples(rows, NEGATIVES, random.Random(0))
    record = build_record(7, split)
    assert record.latent.latent_index == 7
    assert len(record.train) == 20 and len(record.test) == 25 and len(record.not_active) == 25
    # The explainer needs activations quantised to 0..10, relative to the top train example.
    assert record.train[0].normalized_activations.max().item() == 10
    assert all(ex.str_tokens is not None for ex in record.test + record.not_active)
    # Sample ids map each scored context back to its test row for the cross-fit split.
    ids = sample_ids_for(split)
    assert len(ids) == 50 and len(set(ids.values())) == 50


def test_delphi_metrics_give_balanced_accuracy():
    # 20 positives, 30 negatives: TPR = 15/20, TNR = 24/30 -> BA = (0.75 + 0.8) / 2.
    df = pd.DataFrame(
        {"activating": [True] * 20 + [False] * 30,
         "prediction": [True] * 15 + [False] * 5 + [False] * 24 + [True] * 6}
    )
    metrics = compute_classification_metrics(compute_confusion(df))
    assert metrics["accuracy"] == pytest.approx(0.775)


def test_top_token_baseline_takes_most_frequent_max_token():
    train = [
        {"tokens": ["x", " notice", "."], "values": [0, 5.0, 0], "maxValue": 5.0},
        {"tokens": ["x", " notice", "."], "values": [0, 4.0, 0], "maxValue": 4.0},
        {"tokens": [" saw", "y", "."], "values": [9.0, 0, 0], "maxValue": 9.0},
    ]
    assert top_token_explanation(train) == 'the token "notice"'


def test_random_derangement_has_no_fixed_points():
    latents = list(range(50))
    mapping = random_derangement(latents, seed=0)
    assert sorted(mapping) == latents and sorted(mapping.values()) == latents
    assert all(k != v for k, v in mapping.items())
    assert mapping == random_derangement(latents, seed=0)
