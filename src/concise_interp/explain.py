"""Experiment 2: delphi's default explainer, asked for explanations of different lengths.

Global context: to test whether the length factor creates an incentive for short-but-accurate
explanations, we need explanations of the same latent that differ mainly in length. We reuse
delphi's `DefaultExplainer` prompt (system prompt + 3 few-shot examples + highlighted
train contexts) and append one sentence with a word budget.
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from delphi.explainers import DefaultExplainer

from . import config
from .clients import LoggedOpenRouter, job_tag

log = logging.getLogger(__name__)
BUDGET_INSTRUCTION = "\nYour final explanation (the [EXPLANATION]: line) must be approximately {budget} words long."
PARSE_FAILURE = "Explanation could not be parsed."  # delphi's sentinel in Explainer.__call__


def condition_name(budget: int | None) -> str:
    return "gen_unconstrained" if budget is None else f"gen_b{budget}"


@dataclass
class BudgetExplainer(DefaultExplainer):
    word_budget: int | None = None
    """Target length in words; None keeps delphi's default prompt unchanged."""

    def _build_prompt(self, examples):
        messages = super()._build_prompt(examples)
        if self.word_budget is not None:
            # The last message is the user turn holding the highlighted train contexts.
            last = messages[-1]
            messages[-1] = {**last, "content": last["content"] + BUDGET_INSTRUCTION.format(budget=self.word_budget)}
        return messages


async def generate_explanations(records: dict, out_path, budgets=(*config.BUDGETS, None)) -> None:
    """One explanation per (latent, budget), appended to `out_path` (JSONL, resumable)."""
    done = set()
    if out_path.exists():
        done = {(r["latent"], r["condition"]) for r in map(json.loads, out_path.open())}
    client = LoggedOpenRouter(config.EXPLAINER_MODEL, provider=config.MAIN_PROVIDER, max_tokens=500)
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)

    async def one(latent: int, budget: int | None):
        condition = condition_name(budget)
        async with sem:
            job_tag.set(f"explain/{latent}/{condition}")
            explainer = BudgetExplainer(client, temperature=0.0, word_budget=budget,
                                        generation_kwargs={"max_retries": config.MAX_RETRIES})
            try:
                result = await explainer(records[latent])
            except RuntimeError as e:  # delphi raises after all API retries fail
                log.error("explain %s %s failed, will retry on next run: %r", latent, condition, e)
                return
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"latent": latent, "condition": condition, "budget": budget,
                                "explanation": result.explanation,
                                "parsed": result.explanation != PARSE_FAILURE}) + "\n")
        log.info("explained %s %s -> %r", latent, condition, result.explanation)

    jobs = [(lat, b) for lat in records for b in budgets if (lat, condition_name(b)) not in done]
    log.info("explain: %d jobs to run (%d already done)", len(jobs), len(done))
    await asyncio.gather(*(one(lat, b) for lat, b in jobs))
