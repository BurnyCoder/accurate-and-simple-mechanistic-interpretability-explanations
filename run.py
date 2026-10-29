"""Entry point: runs the pipeline phases in order (see README "What happens when you run it").

    uv run python run.py --phase all --sample pilot     # 10-latent debugging run
    uv run python run.py --phase all --sample main      # pre-registered 200-latent run
    uv run python run.py --phase reliability            # retest + alternative scorer, 50 latents
    uv run python run.py --phase analyze                # every table and figure from results/

Phases: download -> negatives -> explain (Exp 2) -> score (Exp 1 + Exp 2) -> reliability -> analyze.
Every phase is resumable: finished jobs are read back from results/ and skipped.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from concise_interp import config
from concise_interp.data import load_processed, prepare
from concise_interp.experiment import exp1_jobs, exp2_jobs, prepare_records
from concise_interp.negatives import build_negatives
from concise_interp.explain import generate_explanations
from concise_interp.scoring import run_scoring


def setup_logging(name: str) -> None:
    """Timestamped log file in logs/ plus the terminal, both untruncated."""
    config.LOG_DIR.mkdir(exist_ok=True)
    path = config.LOG_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{name}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = [logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    logging.info("logging to %s", path)


def results_dir(sample: str):
    path = config.RESULTS_DIR / sample
    path.mkdir(parents=True, exist_ok=True)
    return path


def phase_explain_and_score(sample: str) -> None:
    latents_data, samples = load_processed()
    ids = samples[sample]
    records, sample_ids, splits = prepare_records(latents_data, ids)
    out = results_dir(sample)
    asyncio.run(generate_explanations(records, out / "generated_explanations.jsonl"))
    jobs = exp1_jobs(latents_data, ids, splits) + exp2_jobs(out / "generated_explanations.jsonl", ids)
    asyncio.run(run_scoring(jobs, records, sample_ids, out / "scores.jsonl", config.SCORER_MODEL,
                            config.SHUFFLE_SEED, config.MAIN_PROVIDER))


def phase_reliability() -> None:
    """Re-score the 50-latent reliability subset: new shuffle seed, then alternative scorer."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["download", "negatives", "score", "all", "reliability", "analyze"], required=True)
    parser.add_argument("--sample", choices=["pilot", "main"], default="pilot")
    args = parser.parse_args()
    setup_logging(f"{args.phase}_{args.sample}")
    if args.phase in ("download", "all"):
        logging.info("download: %s", prepare())
    if args.phase in ("negatives", "all"):
        logging.info("negatives: %s", build_negatives(load_processed()[0]))
    if args.phase in ("score", "all"):
        phase_explain_and_score(args.sample)
    if args.phase == "reliability":
        phase_reliability()
    if args.phase == "analyze":
        from concise_interp.analysis import run_analysis  # heavy plotting imports only here
        from concise_interp.render import render_all
        run_analysis(args.sample)
        if args.sample == "main":  # README, reports and paper are rendered from main results only
            logging.info("rendered: %s", [str(p) for p in render_all()])


if __name__ == "__main__":
    main()
