"""Tests for template rendering: numbers must come verbatim from numbers.json."""

import pytest

from concise_interp.render import lookup, render


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
