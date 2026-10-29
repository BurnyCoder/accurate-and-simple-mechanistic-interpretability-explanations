"""Run delphi's DetectionScorer on (latent, condition, explanation) jobs.

Global context: this is the unmodified base method (Paulo et al. 2024). For each job, the
scorer LLM sees the explanation and 50 test contexts (25 activating, 25 not) in shuffled
batches of 5 and answers which ones match. We store every per-sample prediction, so the
analysis can recompute balanced accuracy, J and S, and split samples for cross-fitting.
"""

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from delphi.latents import LatentRecord
from delphi.scorers import DetectionScorer

from . import config
from .clients import LoggedOpenRouter, job_tag
from .records import sample_key

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    latent: int
    condition: str  # e.g. gpt35, gpt4omini, human0, top_token, random, gen_b10
    explanation: str


async def score_explanation(client, record: LatentRecord, explanation: str, shuffle_seed: int) -> list:
    """delphi DetectionScorer on one explanation; returns its ClassifierOutput list.
    A new scorer per job with a fixed seed gives every condition of a latent the same
    shuffled batches, so conditions differ only in the explanation."""
    scorer = DetectionScorer(
        client,
        n_examples_shown=config.N_EXAMPLES_SHOWN,
        temperature=0.0,
        # Both go through **generation_kwargs: `seed` binds to Classifier(seed=...),
        # `max_retries` is forwarded to OpenRouter.generate(max_retries=...).
        seed=shuffle_seed,
        max_retries=config.MAX_RETRIES,
    )
    job_record = copy.copy(record)  # shallow: shares examples, own explanation
    job_record.explanation = explanation
    return (await scorer(job_record)).score


def _done_keys(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    return {(r["latent"], r["condition"]) for r in map(json.loads, path.open(encoding="utf-8"))}


async def run_scoring(jobs: list[Job], records: dict, sample_ids: dict, out_path: Path,
                      scorer_model: str, shuffle_seed: int, provider: dict | None) -> None:
    """Score all jobs not yet in `out_path` (resumable JSONL, one row per job)."""
    done = _done_keys(out_path)
    todo = [j for j in jobs if (j.latent, j.condition) not in done]
    log.info("score -> %s: %d jobs to run (%d already done), scorer=%s seed=%d",
             out_path.name, len(todo), len(done), scorer_model, shuffle_seed)
    client = LoggedOpenRouter(scorer_model, provider=provider, max_tokens=50)
    sem = asyncio.Semaphore(config.MAX_CONCURRENCY)

    async def one(job: Job):
        async with sem:
            job_tag.set(f"score/{job.latent}/{job.condition}")
            outputs = await score_explanation(client, records[job.latent], job.explanation, shuffle_seed)
        ids = sample_ids[job.latent]
        samples = [{"id": ids[sample_key(o.str_tokens, o.activating)], "activating": o.activating,
                    "prediction": o.prediction, "quantile": o.distance if o.activating else None}
                   for o in outputs]
        row = {"latent": job.latent, "condition": job.condition, "explanation": job.explanation,
               "scorer_model": scorer_model, "shuffle_seed": shuffle_seed,
               "time": datetime.now(timezone.utc).isoformat(), "samples": samples}
        with out_path.open("a", encoding="utf-8") as f:  # one write per job: crash-safe resume
            f.write(json.dumps(row) + "\n")
        n_failed = sum(s["prediction"] is None for s in samples)
        log.info("scored %s/%s failed=%d/%d", job.latent, job.condition, n_failed, len(samples))

    await asyncio.gather(*(one(j) for j in todo))
