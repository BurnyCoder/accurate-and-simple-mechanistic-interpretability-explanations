"""Tests for the test-set audits in checks.py (train/test overlap, <|endoftext|>, activation range).

Global context: these audits describe what the scorer actually saw, beyond the pre-registered
split checks in test_records_and_baselines.py, so their counting must be exact on small inputs.
"""

import random

import pandas as pd
import pytest

from concise_interp.checks import EOT, longest_shared_run, positive_rates, split_checks
from concise_interp.records import sample_ids_for, split_examples


def test_longest_shared_run_counts_consecutive_tokens_only():
    assert longest_shared_run(["a", "b", "c", "d"], ["x", "b", "c", "d"]) == 3
    assert longest_shared_run(["a", "b", "c"], ["c", "b", "a"]) == 1
    assert longest_shared_run(["a"], ["b"]) == 0


def _row(value: float, tokens: list[str]) -> dict:
    return {"tokens": tokens, "values": [value] + [0.0] * (len(tokens) - 1), "maxValue": value}


def test_split_checks_counts_overlap_eot_and_activation_ratios():
    # 45 activating rows with distinct tokens: the top-20 go to train and the other 25 are all test
    # positives. One test row copies 8 tokens of the top row, another contains <|endoftext|>.
    rows = [_row(float(100 - i), [f"t{i}_{k}" for k in range(10)]) for i in range(45)]
    rows[20] = _row(80.0, rows[0]["tokens"][:8] + ["u", "v"])
    rows[21] = _row(79.0, [EOT] + [f"e{k}" for k in range(9)])
    negatives = [[f"n{i}", "."] for i in range(30)]
    split = split_examples(rows, negatives, random.Random(0))
    summary, eot_ids = split_checks({7: split}, {7: sample_ids_for(split)})
    assert summary["n_test_positives"] == 25
    assert summary["train_overlap"]["n"] == 1 and summary["train_overlap"]["n_latents"] == 1
    assert len(eot_ids[7]) == 1 and summary["eot"]["pct_positives"] == pytest.approx(4.0)
    assert summary["eot"]["pct_negatives"] == 0
    assert 0 < summary["activation_ratio"]["min"] <= summary["activation_ratio"]["median_by_bin"][0] <= 1


def test_positive_rates_split_positives_by_listed_ids():
    samples = [{"id": "p0", "activating": True, "prediction": True},
               {"id": "p1", "activating": True, "prediction": False},
               {"id": "p2", "activating": True, "prediction": False},
               {"id": "n0", "activating": False, "prediction": True}]
    df = pd.DataFrame([{"latent": 1, "condition": "random", "samples": samples}])
    out = positive_rates(df, "random", {1: {"p0"}})
    assert out["listed"] == pytest.approx(1.0) and out["other"] == pytest.approx(0.0) and out["n_listed"] == 1
