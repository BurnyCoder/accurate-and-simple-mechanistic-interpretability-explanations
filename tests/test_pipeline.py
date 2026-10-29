"""Tests for sampling, the Exp 2 budget prompt, and the request-body hook of the client.

Global context: these are the places where this repo touches delphi or the data. Sampling
must exclude delphi's few-shot latents; the budget explainer must change only the last user
message; the client hook must add provider/seed to every request without dropping fields.
"""

import asyncio
import random

from concise_interp import config
from concise_interp.clients import _MergeIntoBody
from concise_interp.data import draw_samples, eligible_latents
from concise_interp.explain import BUDGET_INSTRUCTION, BudgetExplainer, condition_name
from concise_interp.records import build_record, split_examples


def test_eligibility_requires_both_llm_explanations_counts_and_no_fewshot():
    both = [{"source": "gpt35"}, {"source": "gpt4omini"}]
    explanations = {1: both, 2: both[:1], 3: both, 6048: both, 4: both}
    counts = {1: (50, 25), 2: (50, 25), 3: (44, 25), 6048: (60, 30), 4: (45, 24)}
    assert eligible_latents(explanations, counts) == [1]


def test_samples_are_disjoint_and_reliability_is_inside_main():
    samples = draw_samples(list(range(1000)))
    assert len(samples["main"]) == config.N_MAIN and len(samples["pilot"]) == config.N_PILOT
    assert not set(samples["main"]) & set(samples["pilot"])
    assert set(samples["reliability"]) <= set(samples["main"])
    assert samples == draw_samples(list(range(1000)))


def _record():
    rows = [{"tokens": ["a", " b", " hit"], "values": [0, 0, float(100 - i)], "maxValue": float(100 - i)} for i in range(50)]
    rows += [{"tokens": ["a", " b", " c"], "values": [0, 0, 0], "maxValue": 0.0} for _ in range(30)]
    return build_record(1, split_examples(rows, random.Random(0)))


def test_budget_explainer_appends_instruction_only_to_last_message():
    record = _record()
    default = BudgetExplainer(client=None)._build_prompt(record.train)
    budget = BudgetExplainer(client=None, word_budget=5)._build_prompt(record.train)
    assert default[:-1] == budget[:-1]
    assert budget[-1]["content"] == default[-1]["content"] + BUDGET_INSTRUCTION.format(budget=5)
    assert "<<" in default[-1]["content"]  # delphi highlights activating tokens with << >>
    assert condition_name(5) == "gen_b5" and condition_name(None) == "gen_unconstrained"


def test_merge_into_body_adds_fields_without_dropping_any():
    seen = {}

    class FakeHTTP:
        async def post(self, url, json, **kwargs):
            seen.update(json=json, kwargs=kwargs)
            return "ok"

    hook = _MergeIntoBody(FakeHTTP(), {"provider": {"order": ["DeepInfra"]}, "seed": 0})
    asyncio.run(hook.post("u", json={"model": "m", "messages": []}, timeout=100))
    assert seen["json"] == {"model": "m", "messages": [], "provider": {"order": ["DeepInfra"]}, "seed": 0}
    assert seen["kwargs"] == {"timeout": 100}
