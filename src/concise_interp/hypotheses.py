"""The pre-registered hypotheses H1-H4 (PREREG section 6) and the cross-fitting behind H4.

Global context: `analysis.run_analysis` calls h1-h4 on the main sample's scores frame (see
`metrics.py`) and stores their outputs, including the boolean verdicts, in numbers.json, which
the README, reports and paper are rendered from. Thresholds come from `config`.
"""

import math
import random

import numpy as np
import pandas as pd

from . import config
from .conditions import BUDGET_CONDITIONS, CANDIDATES, EXP1_CONDITIONS
from .metrics import explanation_metrics, s_wide
from .stats import holm, mean_ci, paired, wide

SHORTEST, LONGEST = config.BUDGETS[0], config.BUDGETS[-1]  # H3 endpoints: ~2 and ~40 words
INTERMEDIATE = config.BUDGETS[1:-1]  # H3 candidates for an interior optimum: ~5, ~10, ~20 words


def h1(df: pd.DataFrame) -> dict:
    """Validity: the random control is at chance, Neuronpedia's explanations are above it."""
    j = wide(df, "j")
    out = {c: mean_ci(j[c].dropna()) for c in EXP1_CONDITIONS}
    rnd = out["random"]
    out["supported"] = bool((rnd["lo"] <= 0 <= rnd["hi"] or abs(rnd["mean"]) < config.H1_RANDOM_TOL)
                            and out["gpt35"]["lo"] > 0 and out["gpt4omini"]["lo"] > 0)
    return out


def h2(df: pd.DataFrame) -> dict:
    """J rises from ~2 to ~40 words, and the gain per word shrinks from 2->10 to 10->40 words."""
    j = wide(df, "j")
    rise = paired(j, f"gen_b{LONGEST}", f"gen_b{SHORTEST}")
    mid = 10  # PREREG H2 compares the gain per nominal word over 2->10 with that over 10->40
    gain = ((j[f"gen_b{mid}"] - j[f"gen_b{SHORTEST}"]) / (mid - SHORTEST)
            - (j[f"gen_b{LONGEST}"] - j[f"gen_b{mid}"]) / (LONGEST - mid)).dropna()
    out = {"j_by_budget": {c: mean_ci(j[c].dropna()) for c in BUDGET_CONDITIONS + ["gen_unconstrained"]},
           "rise_b40_minus_b2": rise, "diminishing_gain_per_word": mean_ci(gain)}
    out["supported"] = bool(rise["p_greater"] < config.ALPHA and out["diminishing_gain_per_word"]["lo"] > 0)
    return out


def h3(df: pd.DataFrame, half_life: float = config.HALF_LIFE_TOKENS) -> dict:
    """Primary: some intermediate budget has higher mean S than both endpoints (Holm over 6 tests)."""
    s = s_wide(df, half_life)
    tests = {(b, e): paired(s, f"gen_b{b}", f"gen_b{e}") for b in INTERMEDIATE for e in (SHORTEST, LONGEST)}
    adjusted = dict(zip(tests, holm([t["p_greater"] for t in tests.values()])))
    winners = [b for b in INTERMEDIATE if adjusted[(b, SHORTEST)] < config.ALPHA and adjusted[(b, LONGEST)] < config.ALPHA]
    # Why S falls from ~2 to ~10 words: ratio of mean J vs ratio of the length factor at median L.
    med = df.groupby("condition").n_tokens.median()
    j_mean = wide(df, "j").mean()
    illustration = {"j_ratio_b10_over_b2": float(j_mean["gen_b10"] / j_mean["gen_b2"]),
                    "factor_ratio_b10_over_b2": 2 ** (-(med["gen_b10"] - med["gen_b2"]) / half_life)}
    return {"illustration": illustration, "s_by_budget": {c: mean_ci(s[c].dropna()) for c in BUDGET_CONDITIONS + ["gen_unconstrained"]},
            "tests": {f"b{b}_vs_b{e}": {**t, "p_holm": adjusted[(b, e)]} for (b, e), t in tests.items()},
            "intermediate_budgets_beating_both_endpoints": winners, "supported": bool(winners)}


def crossfit_halves(df: pd.DataFrame, seed: int = config.SEED_ANALYSIS) -> dict[int, tuple[set, set]]:
    """Per latent, split its test-sample ids into two halves, stratified by class."""
    halves = {}
    for latent, group in df.groupby("latent"):
        ids = {s["id"]: s["activating"] for s in group.iloc[0]["samples"]}
        a, b = set(), set()
        for label in (True, False):
            members = sorted(i for i, act in ids.items() if act == label)
            random.Random(f"{seed}-{latent}-{label}").shuffle(members)
            a |= set(members[: len(members) // 2])
            b |= set(members[len(members) // 2:])
        halves[latent] = (a, b)
    return halves


def _rankable(x: float) -> float:
    """A half where every prediction failed has J = NaN; rank it last instead of letting NaN
    comparisons make max()/min() depend on candidate order."""
    return -math.inf if math.isnan(x) else x


def crossfit_selection(df: pd.DataFrame, halves: dict, half_life: float) -> pd.DataFrame:
    """Select a candidate per latent on one half, evaluate its J on the other, both ways.
    Rules: argmax J, argmax S, argmin L. Ties go to the shorter explanation."""
    out = []
    for latent, group in df.groupby("latent"):
        rows = group.to_dict("records")
        evaluations = {rule: [] for rule in ("argmax_j", "argmax_s", "argmin_l")}
        for select_ids, eval_ids in (halves[latent], halves[latent][::-1]):
            sel = [explanation_metrics(r, half_life, select_ids) | {"r": r} for r in rows]
            picks = {"argmax_j": max(sel, key=lambda m: (_rankable(m["j"]), -m["n_tokens"])),
                     "argmax_s": max(sel, key=lambda m: (_rankable(m["s"]), -m["n_tokens"])),
                     "argmin_l": min(sel, key=lambda m: (m["n_tokens"], -_rankable(m["j"])))}
            for rule, pick in picks.items():
                evaluated = explanation_metrics(pick["r"], half_life, eval_ids)
                evaluations[rule].append((evaluated["j"], pick["n_tokens"], pick["r"]["condition"]))
        for rule, ev in evaluations.items():
            out.append({"rule": rule, "latent": latent, "j": float(np.mean([e[0] for e in ev])),
                        "n_tokens": float(np.mean([e[1] for e in ev])), "picked": [e[2] for e in ev]})
    return pd.DataFrame(out)


def h4(df: pd.DataFrame, half_life: float = config.HALF_LIFE_TOKENS, seed: int = config.SEED_ANALYSIS) -> dict:
    """Selection: picking by S gives much shorter picks than by J, at little held-out J, and beats
    picking the shortest. `median_tokens` is the median over latents of each latent's mean pick
    length over the two cross-fit directions."""
    cands = df[df.condition.isin(CANDIDATES)]
    sel = crossfit_selection(cands, crossfit_halves(df, seed), half_life).rename(columns={"rule": "condition"})
    j, length = wide(sel, "j"), wide(sel, "n_tokens")
    loss = (j["argmax_j"] - j["argmax_s"]).dropna()
    out = {"j": {r: mean_ci(j[r].dropna()) for r in j.columns},
           "median_tokens": {r: float(length[r].median()) for r in length.columns},
           "j_loss_argmaxJ_minus_argmaxS": mean_ci(loss),
           "argmaxS_minus_argminL": paired(j, "argmax_s", "argmin_l"),
           "picked_counts": {r: pd.Series(sum(sel[sel.condition == r].picked, [])).value_counts().to_dict()
                             for r in ("argmax_j", "argmax_s", "argmin_l")}}
    ratio = out["median_tokens"]["argmax_s"] / out["median_tokens"]["argmax_j"]
    out["length_ratio_S_over_J"], out["length_reduction"] = ratio, 1 - ratio
    out["n_selections"] = 2 * int(sel.latent.nunique())  # two cross-fit directions per latent
    out["supported"] = bool(ratio <= config.H4_MAX_LENGTH_RATIO
                            and out["j_loss_argmaxJ_minus_argmaxS"]["mean"] <= config.H4_MAX_J_LOSS
                            and out["argmaxS_minus_argminL"]["lo"] > 0)
    return out


def h4_split_sensitivity(df: pd.DataFrame, n_splits: int = config.N_H4_SPLITS) -> dict:
    """Exploratory: H4 re-run on `n_splits` random half-splits (seeds 0..n-1; seed 0 is the
    pre-registered one): how much the result depends on which contexts land in the selection half."""
    runs = [h4(df, seed=seed) for seed in range(n_splits)]

    def spread(values: list[float]) -> dict:
        return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}

    return {"n_splits": n_splits, "n_supported": sum(r["supported"] for r in runs),
            "j_loss": spread([r["j_loss_argmaxJ_minus_argmaxS"]["mean"] for r in runs]),
            "length_reduction": spread([r["length_reduction"] for r in runs]),
            "s_minus_shortest": spread([r["argmaxS_minus_argminL"]["mean"] for r in runs])}


def all_hypotheses(df: pd.DataFrame) -> dict:
    """H1-H4 on one scores frame (the main analysis and the robustness re-runs)."""
    return {"H1": h1(df), "H2": h2(df), "H3": h3(df), "H4": h4(df)}
