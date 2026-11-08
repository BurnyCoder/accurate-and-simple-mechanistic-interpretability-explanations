"""Entry point: runs the pipeline phases in order (see README "What happens when you run it").

    uv run python run.py --phase all --sample pilot     # download + negatives + score, 10 latents
    uv run python run.py --phase score --sample main    # pre-registered 200-latent run
    uv run python run.py --phase reliability            # retest + alternative scorer, 50 latents
    uv run python run.py --phase analyze --sample main  # every table, figure and doc from results/

Phases: download -> negatives -> score (Exp 2 explanations, then Exp 1 + Exp 2 scoring)
-> reliability -> analyze. `all` runs the first three. Every phase is resumable: finished
jobs are read back from results/ and skipped. The phases live in `concise_interp.pipeline`.
"""

import argparse
import logging

from concise_interp import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["download", "negatives", "score", "all", "reliability", "analyze"], required=True)
    parser.add_argument("--sample", choices=["pilot", "main"], default="pilot")
    args = parser.parse_args()
    # The reliability phase always uses the main sample, so its log is not tagged with --sample.
    pipeline.setup_logging("reliability" if args.phase == "reliability" else f"{args.phase}_{args.sample}")
    if args.phase in ("download", "all"):
        logging.info("download: %s", pipeline.download())
    if args.phase in ("negatives", "all"):
        logging.info("negatives: %s", pipeline.negatives())
    if args.phase in ("score", "all"):
        pipeline.explain_and_score(args.sample)
    if args.phase == "reliability":
        pipeline.reliability()
    if args.phase == "analyze":
        logging.info("rendered: %s", [str(p) for p in pipeline.analyze(args.sample)])


if __name__ == "__main__":
    main()
