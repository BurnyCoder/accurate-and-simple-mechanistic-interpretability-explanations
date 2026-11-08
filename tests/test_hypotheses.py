"""Tests for the pre-registered hypothesis verdicts (H1-H3) and the paired statistics behind them.

Global context: `hypotheses.py` turns per-explanation J and S into the PREREG verdicts that the
README and paper report; these tests build small synthetic latent x condition tables whose
verdict is known by construction, so a wrong threshold, test direction or Holm family shows up.
"""

import math

import numpy as np
import pandas as pd
import pytest

from concise_interp.conditions import BUDGET_CONDITIONS
from concise_interp.hypotheses import h1, h2, h3
from concise_interp.metrics import explanation_metrics
from concise_interp.stats import paired, two_sided, wide

TOKENS = {"gen_b2": 2, "gen_b5": 6, "gen_b10": 11, "gen_b20": 20, "gen_b40": 45, "gen_unconstrained": 30}


def _frame(mean_j: dict[str, float], tokens: dict[str, int] | None = None, n: int = 30, seed: int = 0) -> pd.DataFrame:
    """n latents x the given conditions; J = mean + small per-latent noise, so paired tests have variance."""
    rng = np.random.default_rng(seed)
    rows = [{"latent": lat, "condition": c, "j": m + rng.normal(0, 0.02),
             "n_tokens": (tokens or {}).get(c, 8)} for lat in range(n) for c, m in mean_j.items()]
    df = pd.DataFrame(rows)
    df["s"] = df.j.clip(lower=0) * 2 ** (-df.n_tokens / 20)
    return df


def test_h1_supported_when_random_is_chance_and_real_explanations_are_above():
    df = _frame({"random": 0.0, "gpt35": 0.45, "gpt4omini": 0.5, "top_token": 0.4})
    assert h1(df)["supported"]


def test_h1_fails_when_the_random_control_scores_clearly_above_chance():
    df = _frame({"random": 0.2, "gpt35": 0.45, "gpt4omini": 0.5, "top_token": 0.4})
    assert not h1(df)["supported"]


def test_h2_supported_for_rising_j_with_diminishing_gain_per_word():
    j = {"gen_b2": 0.3, "gen_b5": 0.42, "gen_b10": 0.5, "gen_b20": 0.53, "gen_b40": 0.55, "gen_unconstrained": 0.54}
    out = h2(_frame(j, TOKENS))
    assert out["supported"] and out["rise_b40_minus_b2"]["mean"] == pytest.approx(0.25, abs=0.02)


def test_h2_fails_when_j_is_flat():
    out = h2(_frame({c: 0.5 for c in [*BUDGET_CONDITIONS, "gen_unconstrained"]}, TOKENS))
    assert not out["supported"]


def test_h3_finds_an_interior_optimum_when_a_middle_budget_beats_both_ends():
    # S(~5 words) = 0.6 * 2^(-6/20) = 0.49 beats S(~2) = 0.19 and S(~40) = 0.14.
    j = {"gen_b2": 0.2, "gen_b5": 0.6, "gen_b10": 0.62, "gen_b20": 0.64, "gen_b40": 0.66, "gen_unconstrained": 0.6}
    out = h3(_frame(j, TOKENS))
    assert out["supported"] and 5 in out["intermediate_budgets_beating_both_endpoints"]
    assert len(out["tests"]) == 6  # Holm family: 3 intermediate budgets x 2 endpoints


def test_h3_is_falsified_when_s_falls_monotonically_with_length():
    out = h3(_frame({c: 0.5 for c in [*BUDGET_CONDITIONS, "gen_unconstrained"]}, TOKENS))
    assert not out["supported"] and out["intermediate_budgets_beating_both_endpoints"] == []


def test_paired_tests_that_a_is_greater_than_b():
    w = pd.DataFrame({"a": np.arange(20) + 1.0, "b": np.zeros(20)})
    assert paired(w, "a", "b")["p_greater"] < 0.001  # a > b everywhere
    assert paired(w, "b", "a")["p_greater"] > 0.99  # the reverse direction is not significant


def test_two_sided_returns_p_one_when_every_difference_is_zero():
    w = pd.DataFrame({"a": [0.5] * 5, "b": [0.5] * 5})
    assert two_sided(w, "a", "b")["two_sided_p"] == 1.0


def test_explanation_metrics_j_is_nan_when_a_class_has_no_valid_prediction():
    # All negatives failed to parse: delphi would report TNR = 0 and so J = TPR - 1 < 0.
    samples = [{"id": f"p{i}", "activating": True, "prediction": True} for i in range(5)]
    samples += [{"id": f"n{i}", "activating": False, "prediction": None} for i in range(5)]
    m = explanation_metrics({"explanation": "x", "samples": samples}, half_life=20)
    assert math.isnan(m["j"]) and math.isnan(m["s"]) and m["n_failed"] == 5


def test_wide_is_latent_by_condition():
    df = _frame({"a": 0.1, "b": 0.2}, n=3)
    assert wide(df, "j").shape == (3, 2)
