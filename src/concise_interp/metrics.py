"""Per-explanation metrics: balanced accuracy (delphi), J, lengths and S, from stored predictions.

Global context: scoring.py stores every per-context prediction, so all metrics are recomputed
here, at any half-life H and on any subset of the test contexts (cross-fit halves, robustness
checks), without new LLM calls. `hypotheses.py` and `analysis.py` build on these frames.
"""

import math
from pathlib import Path

import pandas as pd
from delphi.log.result_analysis import compute_classification_metrics, compute_confusion

from . import config
from .jsonl import read_jsonl
from .length_score import length_penalized_score, n_chars, n_tokens, n_words, youden_j
from .stats import wide


def explanation_metrics(row: dict, half_life: float, ids: set[str] | None = None) -> dict:
    """BA (delphi), J, lengths and S for one scored explanation, optionally on a subset of samples.
    J is NaN unless both classes have at least one valid prediction: delphi scores an empty class's
    rate as 0, which would turn a class lost to parse failures into a spuriously negative J."""
    samples = [s for s in row["samples"] if ids is None or s["id"] in ids]
    conf = compute_confusion(pd.DataFrame(samples, columns=["activating", "prediction"]))
    ba = float(compute_classification_metrics(conf)["accuracy"])  # delphi's "accuracy" is balanced
    defined = conf["total_positives"] > 0 and conf["total_negatives"] > 0
    j = youden_j(ba) if defined else math.nan
    text = row["explanation"]
    length = n_tokens(text)
    return {"ba": ba if defined else math.nan, "j": j,
            "n_tokens": length, "n_words": n_words(text), "n_chars": n_chars(text),
            "s": length_penalized_score(j, length, half_life) if defined else math.nan,
            "n_valid": int(conf["total_examples"]), "n_failed": int(conf["failed_count"])}


def scores_frame(rows: list[dict], half_life: float = config.HALF_LIFE_TOKENS) -> pd.DataFrame:
    """One row per (latent, condition) with raw samples and metrics at the default H."""
    metrics = pd.DataFrame([explanation_metrics(r, half_life) for r in rows])
    return pd.concat([pd.DataFrame(rows), metrics], axis=1)


def load_scores(path: Path, half_life: float = config.HALF_LIFE_TOKENS) -> pd.DataFrame:
    """A scores*.jsonl file as a `scores_frame`."""
    return scores_frame(read_jsonl(path), half_life)


def without_samples(df: pd.DataFrame, drop: dict[int, set[str]]) -> pd.DataFrame:
    """The same explanations with some test samples removed (per latent, by id), metrics recomputed."""
    rows = [r | {"samples": [s for s in r["samples"] if s["id"] not in drop.get(r["latent"], set())]}
            for r in df[["latent", "condition", "explanation", "samples"]].to_dict("records")]
    return scores_frame(rows)


def with_half_life(df: pd.DataFrame, half_life: float) -> pd.Series:
    """S of every row recomputed at another half-life H (J and L are unchanged)."""
    return pd.Series([length_penalized_score(j, n, half_life) for j, n in zip(df.j, df.n_tokens)], index=df.index)


def s_wide(df: pd.DataFrame, half_life: float) -> pd.DataFrame:
    """latent x condition table of S at half-life H."""
    return wide(df.assign(s=with_half_life(df, half_life)), "s")
