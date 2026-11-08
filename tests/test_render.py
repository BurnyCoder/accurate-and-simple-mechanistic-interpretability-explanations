"""Tests for template rendering: numbers must come verbatim from numbers.json."""

import pytest

from concise_interp.render import (
    lookup,
    render,
    table_conditions,
    table_h3,
    table_h4,
    table_h_sweep,
    table_providers,
    table_reliability,
)


def test_lookup_walks_dicts_and_lists():
    numbers = {"a": {"b": [{"c": 0.12345}]}}
    assert lookup(numbers, "a.b.0.c") == 0.12345


def test_render_formats_and_leaves_text_alone():
    numbers = {"H3": {"supported": True, "s": {"mean": 0.41234}}}
    out = render("S = {{H3.s.mean:.2f}}, supported: {{H3.supported}}.", numbers)
    assert out == "S = 0.41, supported: True."


def test_include_renders_the_included_file(tmp_path):
    part = tmp_path / "part.md"
    part.write_text("J = {{j:.1f}}", encoding="utf-8")
    # An absolute path stays absolute under ROOT / path.
    assert render("A {{include:" + str(part) + "}} B", {"j": 0.42}) == "A J = 0.4 B"


def test_render_fails_loudly_on_unknown_key():
    with pytest.raises(KeyError):
        render("{{missing.key}}", {})


CI = {"mean": 0.5, "lo": 0.4, "hi": 0.6}


def _rows(table: str) -> list[str]:
    return [line for line in table.splitlines() if line.startswith("|")]


def test_table_conditions_orders_rows_and_formats_cis():
    row = {"n": 200, "ba": 0.75, "j_mean": 0.5, "j_lo": 0.45, "j_hi": 0.55, "tokens_median": 8.0,
           "s_mean": 0.37, "s_lo": 0.34, "s_hi": 0.41}
    numbers = {"half_life": 20.0, "conditions": [row | {"condition": "gpt35"}, row | {"condition": "random"}]}
    lines = _rows(table_conditions(numbers))
    assert len(lines) == 4 and "(H=20)" in lines[0]
    assert lines[2].startswith("| random-explanation control | control |")  # ORDER puts the control first
    assert "0.500 [0.450, 0.550]" in lines[3]


def test_table_h3_labels_each_comparison():
    numbers = {"half_life": 20.0, "H3": {"tests": {"b5_vs_b2": CI | {"p_greater": 0.99, "p_holm": 1.0}}}}
    assert "| ~5 words − ~2 words | +0.500 [+0.400, +0.600] | 0.99 | 1 |" in table_h3(numbers)


def test_table_h4_has_one_row_per_rule():
    numbers = {"H4": {"j": {r: CI for r in ("argmax_j", "argmax_s", "argmin_l")},
                      "median_tokens": {"argmax_j": 9.0, "argmax_s": 5.5, "argmin_l": 2.0}}}
    lines = _rows(table_h4(numbers))
    assert len(lines) == 5 and "| pick highest S (this method) | 0.500 [0.400, 0.600] | 5.5 |" in lines


def test_table_h_sweep_has_one_row_per_h_and_marks_infinity():
    sweep = {h: {"mean_s": {"random": 0.01, "gen_b2": 0.4}, "best_budget": "gen_b2"} for h in ("20", "inf")}
    lines = _rows(table_h_sweep({"exploratory": {"h_sweep": sweep}}))
    assert len(lines) == 4 and lines[3].startswith("| ∞") and lines[2].endswith("| ~2 words |")


def test_table_reliability_skips_missing_runs():
    r = {"n_latents": 50, "n_explanations": 502, "spearman_j": 0.9, "spearman_s": 0.9,
         "spearman_j_within": 0.8, "spearman_s_within": 0.8}
    lines = _rows(table_reliability({"reliability_retest": r}))
    assert len(lines) == 3 and "| 50 | 502 | 0.900 |" in lines[2]


def test_table_providers_lists_each_output():
    numbers = {"api_usage": {p: {"providers": {"scores": {"DeepInfra": 3}}} for p in ("score", "reliability")}}
    assert "| score | `scores` | DeepInfra: 3 |" in table_providers(numbers)
