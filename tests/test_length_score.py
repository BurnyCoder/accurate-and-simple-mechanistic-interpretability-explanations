"""Tests for the length-penalized score S = max(J, 0) * 2^(-L/H).

Global context: `length_score` is the only new method in this repo; everything else
reuses delphi's detection scorer. These tests pin down the math the paper reports.
"""

import math

import pytest

from concise_interp.length_score import (
    length_factor,
    length_penalized_score,
    n_chars,
    n_tokens,
    n_words,
    youden_j,
)


def test_youden_j_maps_chance_to_zero_and_perfect_to_one():
    # Balanced accuracy 0.5 is chance, so J must be 0; 1.0 is perfect, so J must be 1.
    assert youden_j(0.5) == 0.0
    assert youden_j(1.0) == 1.0
    assert youden_j(0.0) == -1.0
    assert youden_j(0.75) == pytest.approx(0.5)


def test_length_factor_halves_every_half_life():
    # The half-life H is the length at which the factor drops to 1/2, and it keeps halving.
    assert length_factor(0, 20) == 1.0
    assert length_factor(20, 20) == pytest.approx(0.5)
    assert length_factor(40, 20) == pytest.approx(0.25)
    assert length_factor(10, 20) == pytest.approx(2 ** -0.5)


def test_infinite_half_life_disables_the_penalty():
    # H = inf recovers plain detection scoring, used as the "no length factor" ablation.
    assert length_factor(1000, math.inf) == 1.0
    assert length_penalized_score(0.6, 1000, math.inf) == pytest.approx(0.6)


def test_below_chance_explanations_get_zero_regardless_of_length():
    # A short explanation must not earn credit for being short when it is at or below chance.
    assert length_penalized_score(-0.3, 1, 20) == 0.0
    assert length_penalized_score(0.0, 1, 20) == 0.0


def test_shorter_wins_at_equal_accuracy_and_accuracy_wins_at_equal_length():
    assert length_penalized_score(0.8, 5, 20) > length_penalized_score(0.8, 25, 20)
    assert length_penalized_score(0.9, 10, 20) > length_penalized_score(0.7, 10, 20)


def test_log_score_is_log_j_minus_constant_cost_per_token():
    # MDL reading of the factor: log S = log J - (ln 2 / H) * L.
    j, length, h = 0.7, 13, 20
    expected = math.log(j) - math.log(2) / h * length
    assert math.log(length_penalized_score(j, length, h)) == pytest.approx(expected)


def test_length_units():
    text = "  the token \"notice\"  "
    # Leading/trailing whitespace must not count toward length.
    assert n_words(text) == 3
    assert n_chars(text) == len('the token "notice"')
    assert n_tokens(text) == n_tokens(text.strip())
    assert 3 <= n_tokens(text) <= 6
    assert n_tokens("") == 0
