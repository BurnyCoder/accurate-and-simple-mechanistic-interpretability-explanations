"""Build the per-latent records and the list of scoring jobs (PREREG section 5).

Global context: sits between `data.py` (raw rows) and `scoring.py` (delphi scorer). For every
latent in a sample it fixes the example split once (seed 0), so every condition of that
latent is scored on the same 50 test contexts.
"""

import json
import logging
import random
from pathlib import Path

from . import config
from .baselines import random_derangement, top_token_explanation
from .negatives import load_negatives
from .records import build_record, sample_ids_for, split_examples
from .scoring import Job

log = logging.getLogger(__name__)


def prepare_records(latents_data: dict[int, dict], latent_ids: list[int]):
    """latent -> (delphi record, sample-id map, split), with the pre-registered split."""
    records, sample_ids, splits = {}, {}, {}
    windows, zero_windows = load_negatives()
    for latent in latent_ids:
        negatives = [windows[i] for i in zero_windows[latent]]
        # A per-latent RNG keeps each latent's split independent of which others are in the sample.
        split = split_examples(latents_data[latent]["rows"], negatives, random.Random(f"{config.SEED_MAIN}-{latent}"))
        splits[latent], records[latent], sample_ids[latent] = split, build_record(latent, split), sample_ids_for(split)
        if len(sample_ids[latent]) < config.N_TEST_POS + config.N_TEST_NEG:
            log.warning("latent %d has duplicate test contexts (%d unique ids)", latent, len(sample_ids[latent]))
    return records, sample_ids, splits


def _first(explanations: list[dict], source: str) -> str:
    return next(e["description"] for e in explanations if e["source"] == source)


def exp1_jobs(latents_data: dict[int, dict], latent_ids: list[int], splits: dict) -> list[Job]:
    """Existing explanations, the top-token baseline, and the random-explanation control."""
    derangement = random_derangement(latent_ids, seed=config.SEED_MAIN)
    jobs = []
    for latent in latent_ids:
        expl = latents_data[latent]["explanations"]
        jobs += [Job(latent, "gpt35", _first(expl, "gpt35")),
                 Job(latent, "gpt4omini", _first(expl, "gpt4omini")),
                 Job(latent, "top_token", top_token_explanation(splits[latent].train)),
                 Job(latent, "random", _first(latents_data[derangement[latent]]["explanations"], "gpt4omini"))]
        humans = [e["description"] for e in expl if e["source"] == "human"]
        jobs += [Job(latent, f"human{k}", text) for k, text in enumerate(humans)]
    return jobs


def exp2_jobs(generated_path: Path, latent_ids: list[int]) -> list[Job]:
    """Generated explanations (one per word budget) that parsed successfully."""
    if not generated_path.exists():
        return []
    rows = [r for r in map(json.loads, generated_path.open(encoding="utf-8")) if r["latent"] in set(latent_ids)]
    failed = [r for r in rows if not r["parsed"]]
    if failed:
        log.warning("%d generated explanations failed to parse and are not scored", len(failed))
    return [Job(r["latent"], r["condition"], r["explanation"]) for r in rows if r["parsed"]]
