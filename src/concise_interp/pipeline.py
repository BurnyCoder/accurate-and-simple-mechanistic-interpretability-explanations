"""The pipeline phases that `run.py` dispatches to, and the run log they write.

Global context: `run.py` is only the command-line wrapper. Each function here is one phase of
README "What happens when you run it": download -> negatives -> score (Exp 2 explanations, then
Exp 1 + Exp 2 scoring) -> reliability -> analyze. Every phase is resumable: finished jobs are
read back from results/ and skipped.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import config


def setup_logging(name: str) -> Path:
    """Timestamped log file logs/<YYYYmmdd-HHMMSS>_<name>.log plus the terminal, both untruncated."""
    config.LOG_DIR.mkdir(exist_ok=True)
    path = config.LOG_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = [logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    logging.info("logging to %s", path)
    return path


def results_dir(sample: str) -> Path:
    """results/<sample>/, created if needed."""
    path = config.RESULTS_DIR / sample
    path.mkdir(parents=True, exist_ok=True)
    return path


def download() -> dict:
    """Phase `download`: Neuronpedia export -> eligible latents -> pre-registered samples."""
    from .data import prepare
    return prepare()


def negatives() -> dict:
    """Phase `negatives`: verified zero-activation OpenWebText windows (GPT-2 small + SAE)."""
    from .data import load_processed
    from .negatives import build_negatives
    return build_negatives(load_processed()[0])


def explain_and_score(sample: str) -> None:
    """Phase `score`: Exp 2 explanations (Llama-3.1-70B, word budgets), then delphi detection
    scoring of every Exp 1 and Exp 2 explanation of the sample."""
    from .data import load_processed
    from .experiment import exp1_jobs, exp2_jobs, prepare_records
    from .explain import generate_explanations
    from .scoring import run_scoring

    latents_data, samples = load_processed()
    ids = samples[sample]
    records, sample_ids, splits = prepare_records(latents_data, ids)
    out = results_dir(sample)
    asyncio.run(generate_explanations(records, out / "generated_explanations.jsonl"))
    jobs = exp1_jobs(latents_data, ids, splits) + exp2_jobs(out / "generated_explanations.jsonl", ids)
    asyncio.run(run_scoring(jobs, records, sample_ids, out / "scores.jsonl", config.SCORER_MODEL,
                            config.SHUFFLE_SEED, config.MAIN_PROVIDER))


def reliability() -> None:
    """Phase `reliability`: re-score the 50-latent subset with a new shuffle seed, then with the
    alternative scorer (PREREG section 7)."""
    from .data import load_processed
    from .experiment import exp1_jobs, exp2_jobs, prepare_records
    from .scoring import run_scoring

    latents_data, samples = load_processed()
    main_ids, subset = samples["main"], set(samples["reliability"])
    records, sample_ids, splits = prepare_records(latents_data, main_ids)
    out = results_dir("main")
    # Exactly the main run's jobs (random-control pairing included), restricted to the subset.
    jobs = [j for j in exp1_jobs(latents_data, main_ids, splits)
            + exp2_jobs(out / "generated_explanations.jsonl", main_ids) if j.latent in subset]
    asyncio.run(run_scoring(jobs, records, sample_ids, out / "scores_retest.jsonl", config.SCORER_MODEL,
                            config.RETEST_SHUFFLE_SEED, config.MAIN_PROVIDER))
    asyncio.run(run_scoring(jobs, records, sample_ids, out / "scores_altscorer.jsonl", config.ALT_SCORER_MODEL,
                            config.SHUFFLE_SEED, None))


def analyze(sample: str) -> list[Path]:
    """Phase `analyze`: numbers.json and figures; for the main sample also README, reports, paper."""
    from .analysis import run_analysis  # heavy plotting imports only here
    from .render import render_all

    run_analysis(sample)
    return render_all() if sample == "main" else []  # docs are rendered from main results only
