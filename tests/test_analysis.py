"""Tests for the analysis helpers: per-explanation metrics, bootstrap CIs, cross-fit selection.

Global context: every number in README/paper comes from `analysis.py` reading results/*.jsonl;
these tests check the statistics on small inputs whose answers are known by hand.
"""

import math

import numpy as np
import pandas as pd
import pytest

from concise_interp import config
from concise_interp.analysis import (
    api_usage,
    bootstrap_ci,
    condition_table,
    crossfit_selection,
    explanation_metrics,
    holm,
    mean_ci,
    wide,
)
from concise_interp.data import filter_counts


def _samples(tp, fn, tn, fp, failed=0):
    s = [{"id": f"p{i}", "activating": True, "prediction": i < tp} for i in range(tp + fn)]
    s += [{"id": f"n{i}", "activating": False, "prediction": i >= tn} for i in range(tn + fp)]
    s += [{"id": f"x{i}", "activating": True, "prediction": None} for i in range(failed)]
    return s


def test_explanation_metrics_ba_j_and_s():
    row = {"explanation": "the token \"notice\"", "samples": _samples(tp=20, fn=5, tn=25, fp=0, failed=2)}
    m = explanation_metrics(row, half_life=20)
    assert m["ba"] == pytest.approx((20 / 25 + 1.0) / 2)
    assert m["j"] == pytest.approx(0.8)
    assert m["n_failed"] == 2
    assert m["s"] == pytest.approx(0.8 * 2 ** (-m["n_tokens"] / 20))


def test_explanation_metrics_on_a_subset_of_sample_ids():
    row = {"explanation": "x", "samples": _samples(tp=2, fn=2, tn=2, fp=2)}
    # Only p0, p1 (both predicted positive) and n0, n1 (both predicted negative): perfect.
    m = explanation_metrics(row, half_life=math.inf, ids={"p0", "p1", "n0", "n1"})
    assert m["j"] == pytest.approx(1.0)


def test_bootstrap_ci_contains_mean_and_shrinks_with_n():
    rng = np.random.default_rng(0)
    small, large = rng.normal(0.3, 1, 20), rng.normal(0.3, 1, 2000)
    lo_s, hi_s = bootstrap_ci(small, seed=0, n_boot=2000)
    lo_l, hi_l = bootstrap_ci(large, seed=0, n_boot=2000)
    assert lo_s < small.mean() < hi_s and lo_l < large.mean() < hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_holm_adjusts_in_step_down_order():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_crossfit_selection_trades_accuracy_for_length():
    # "long" (60 tokens) is perfect, J = 1 on both halves, so S = 2^-3 = 0.125.
    # "short" (1 token) has TPR 1, TNR 0.5 on each half, so J = 0.5 and S ~ 0.48.
    perfect = _samples(tp=4, fn=0, tn=4, fp=0)
    half_right = [{"id": f"p{i}", "activating": True, "prediction": True} for i in range(4)]
    half_right += [{"id": f"n{i}", "activating": False, "prediction": i % 2 == 1} for i in range(4)]
    rows = pd.DataFrame([
        {"latent": 1, "condition": "long", "explanation": " ".join(["word"] * 60), "samples": perfect},
        {"latent": 1, "condition": "short", "explanation": "word", "samples": half_right},
    ])
    halves = {1: ({"p0", "p1", "n0", "n1"}, {"p2", "p3", "n2", "n3"})}
    out = crossfit_selection(rows, halves, half_life=20).set_index("rule")
    assert out.loc["argmax_j", "j"] == pytest.approx(1.0) and out.loc["argmax_j", "n_tokens"] == 60
    assert out.loc["argmax_s", "j"] == pytest.approx(0.5) and out.loc["argmax_s", "n_tokens"] == 1
    assert out.loc["argmin_l", "n_tokens"] == 1


def test_crossfit_ties_go_to_the_shorter_explanation():
    perfect = _samples(tp=4, fn=0, tn=4, fp=0)
    rows = pd.DataFrame([
        {"latent": 1, "condition": "long", "explanation": "a b c d e f", "samples": perfect},
        {"latent": 1, "condition": "short", "explanation": "a", "samples": perfect},
    ])
    halves = {1: ({"p0", "p1", "n0", "n1"}, {"p2", "p3", "n2", "n3"})}
    out = crossfit_selection(rows, halves, half_life=20).set_index("rule")
    assert out.loc["argmax_j", "n_tokens"] == 1


def test_condition_table_ci_is_the_h1_ci_whatever_the_row_order():
    # scores.jsonl rows arrive in API-completion order; the table must not depend on it.
    rng = np.random.default_rng(0)
    df = pd.DataFrame([{"latent": lat, "condition": c, "j": rng.normal(), "s": rng.random(),
                        "ba": rng.random(), "n_tokens": 5} for lat in range(30) for c in ("a", "b")])
    shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    t1, t2 = (condition_table(d).set_index("condition") for d in (df, shuffled))
    pd.testing.assert_frame_equal(t1, t2)
    assert t1.loc["a", "j_lo"] == mean_ci(wide(df, "j")["a"].dropna())["lo"]


def test_api_usage_counts_providers_per_output(tmp_path, monkeypatch):
    meta = 'x INFO concise_interp.llm: META [{tag}] provider={p} usage={{"cost": 0.5, "total_tokens": 10}}\n'
    (tmp_path / "1_score_main.log").write_text(
        meta.format(tag="explain/1/gen_b2", p="DeepInfra")
        + "x INFO concise_interp.scoring: score -> scores.jsonl: 1 jobs to run\n"
        + meta.format(tag="score/1/gpt35", p="DeepInfra") + meta.format(tag="score/1/gpt35", p="Groq"),
        encoding="utf-8")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    usage = api_usage("*score_main.log")
    assert usage["calls"] == 3 and usage["cost_usd"] == pytest.approx(1.5) and usage["total_tokens"] == 30
    assert usage["providers"] == {"explain": {"DeepInfra": 1}, "scores": {"DeepInfra": 1, "Groq": 1}}


def test_api_usage_keeps_committed_totals_without_logs(tmp_path, monkeypatch):
    # A fresh clone has no logs/; analyze must not overwrite the committed totals with zeros.
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    committed = {"calls": 7, "cost_usd": 1.0, "total_tokens": 9, "providers": {"scores": {"DeepInfra": 7}}}
    assert api_usage("*score_main.log", committed) == committed
    assert api_usage("*score_main.log")["calls"] == 0


def test_filter_counts_counts_each_eligibility_filter_alone():
    explanations = {1: [{"source": "gpt35"}, {"source": "gpt4omini"}], 2: [{"source": "gpt35"}]}
    counts = {1: (config.MIN_ACTIVATING, 0), 2: (0, config.MIN_ZERO), 3: (config.MIN_ACTIVATING, config.MIN_ZERO)}
    assert filter_counts(explanations, counts) == {"both_llm_explanations": 1, "min_activating": 2, "min_zero_rows": 2}
