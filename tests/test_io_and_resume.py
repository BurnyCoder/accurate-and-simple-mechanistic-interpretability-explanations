"""Tests for resumable JSONL results, the scoring resume rule, and the API-key check.

Global context: the score and explain phases append one JSONL row per finished job and skip
finished jobs on a rerun (PREREG section 8: API failures are retried). A job whose every
prediction failed must therefore not be saved, a crash mid-write must not break the rerun, and
a missing OpenRouter key must stop the run before thousands of doomed calls.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from concise_interp import clients, config, scoring
from concise_interp.jsonl import append_jsonl, done_keys, read_jsonl
from concise_interp.scoring import Job, run_scoring


def test_read_jsonl_skips_a_truncated_last_line(tmp_path, caplog):
    path = tmp_path / "x.jsonl"
    path.write_text('{"latent": 1, "condition": "a"}\n{"latent": 2, "condition": "b"}\n{"latent": 3, "con',
                    encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        rows = read_jsonl(path)
    assert [r["latent"] for r in rows] == [1, 2] and "truncated" in caplog.text
    assert done_keys(path) == {(1, "a"), (2, "b")}


def test_read_jsonl_still_fails_on_a_corrupt_line_in_the_middle(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"latent": 1}\nnot json\n{"latent": 2}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_jsonl(path)


def test_append_then_read_round_trips_and_missing_file_is_empty(tmp_path):
    path = tmp_path / "x.jsonl"
    assert read_jsonl(path) == [] and done_keys(path) == set()
    append_jsonl(path, {"latent": 1, "condition": "a", "text": "é"})
    assert read_jsonl(path) == [{"latent": 1, "condition": "a", "text": "é"}]


def _fake_scoring(monkeypatch, predictions_by_condition):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

    async def fake_score(client, record, explanation, shuffle_seed):
        return [SimpleNamespace(str_tokens=["a"], activating=True, prediction=p, distance=0)
                for p in predictions_by_condition[explanation]]

    monkeypatch.setattr(scoring, "LoggedOpenRouter", FakeClient)
    monkeypatch.setattr(scoring, "score_explanation", fake_score)


def test_run_scoring_does_not_save_a_job_whose_predictions_all_failed(tmp_path, monkeypatch):
    _fake_scoring(monkeypatch, {"ok": [True], "outage": [None]})
    out = tmp_path / "scores.jsonl"
    jobs = [Job(1, "good", "ok"), Job(1, "bad", "outage")]
    asyncio.run(run_scoring(jobs, {1: None}, {1: {("a", True): "p0"}}, out, "m", 42, None))
    assert done_keys(out) == {(1, "good")}  # the failed job stays to do, so a rerun retries it


def test_logged_openrouter_refuses_to_start_without_an_api_key(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", None)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        clients.LoggedOpenRouter("m")
