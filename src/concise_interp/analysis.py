"""Pre-registered analysis (PREREG section 6-7): every reported number is computed here.

Global context: last phase of `run.py`. Reads results/<sample>/scores*.jsonl (one row per
scored explanation, with every per-sample prediction), recomputes balanced accuracy with
delphi's own metric code, derives J, L and S, runs the hypothesis tests, and writes
results/<sample>/numbers.json + tables.md, which README, reports and paper are rendered from.
"""

import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from delphi.log.result_analysis import compute_classification_metrics, compute_confusion
from scipy import stats

from . import config
from .length_score import length_penalized_score, n_chars, n_tokens, n_words, youden_j

log = logging.getLogger(__name__)
BUDGET_CONDITIONS = [f"gen_b{b}" for b in config.BUDGETS]
EXP1_CONDITIONS = ["gpt35", "gpt4omini", "top_token", "random"]
CANDIDATES = [*BUDGET_CONDITIONS, "gen_unconstrained", "gpt35", "gpt4omini"]  # H4 candidate set


# ---------- per-explanation metrics ----------

def explanation_metrics(row: dict, half_life: float, ids: set[str] | None = None) -> dict:
    """BA (delphi), J, lengths and S for one scored explanation, optionally on a subset of samples."""
    samples = [s for s in row["samples"] if ids is None or s["id"] in ids]
    df = pd.DataFrame(samples, columns=["activating", "prediction"])
    conf = compute_confusion(df)
    m = compute_classification_metrics(conf)
    valid = int(conf["total_examples"])
    j = youden_j(float(m["accuracy"])) if valid else math.nan
    text = row["explanation"]
    length = n_tokens(text)
    return {"ba": float(m["accuracy"]) if valid else math.nan, "j": j,
            "precision": float(m["precision"]), "recall": float(m["recall"]),
            "n_tokens": length, "n_words": n_words(text), "n_chars": n_chars(text),
            "s": length_penalized_score(j, length, half_life) if valid else math.nan,
            "n_valid": valid, "n_failed": int(conf["failed_count"])}


def load_scores(path: Path, half_life: float = config.HALF_LIFE_TOKENS) -> pd.DataFrame:
    """One row per (latent, condition) with raw samples and metrics at the default H."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    df = pd.DataFrame(rows)
    metrics = pd.DataFrame([explanation_metrics(r, half_life) for r in rows])
    return pd.concat([df, metrics], axis=1)


def with_half_life(df: pd.DataFrame, half_life: float) -> pd.Series:
    return pd.Series([length_penalized_score(j, n, half_life) for j, n in zip(df.j, df.n_tokens)], index=df.index)


# ---------- statistics ----------

def bootstrap_ci(values, seed: int = config.SEED_ANALYSIS, n_boot: int = config.N_BOOTSTRAP, level: float = 0.95):
    """Percentile bootstrap CI of the mean (resampling latents)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return tuple(np.quantile(means, [(1 - level) / 2, (1 + level) / 2]).tolist())


def mean_ci(values) -> dict:
    values = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(values)
    return {"mean": float(values.mean()), "lo": lo, "hi": hi, "n": int(len(values))}


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (Holm 1979), in the input order."""
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """latent x condition table of `value` (paired analyses drop latents missing a condition)."""
    return df.pivot_table(index="latent", columns="condition", values=value, aggfunc="first")


def paired(df_wide: pd.DataFrame, a: str, b: str) -> dict:
    """Mean of a - b over latents having both, with bootstrap CI and one-sided Wilcoxon (a > b)."""
    d = (df_wide[a] - df_wide[b]).dropna()
    p = float(stats.wilcoxon(d, alternative="greater").pvalue) if (d != 0).any() else 1.0
    return {**mean_ci(d), "p_greater": p}


# ---------- hypotheses ----------

def h1(df: pd.DataFrame) -> dict:
    j = wide(df, "j")
    out = {c: mean_ci(j[c].dropna()) for c in EXP1_CONDITIONS}
    rnd = out["random"]
    out["supported"] = bool((rnd["lo"] <= 0 <= rnd["hi"] or abs(rnd["mean"]) < 0.05)
                            and out["gpt35"]["lo"] > 0 and out["gpt4omini"]["lo"] > 0)
    return out


def h2(df: pd.DataFrame) -> dict:
    j = wide(df, "j")
    rise = paired(j, "gen_b40", "gen_b2")
    gain = ((j["gen_b10"] - j["gen_b2"]) / 8 - (j["gen_b40"] - j["gen_b10"]) / 30).dropna()
    out = {"j_by_budget": {c: mean_ci(j[c].dropna()) for c in BUDGET_CONDITIONS + ["gen_unconstrained"]},
           "rise_b40_minus_b2": rise, "diminishing_gain_per_word": mean_ci(gain)}
    out["supported"] = bool(rise["p_greater"] < 0.05 and out["diminishing_gain_per_word"]["lo"] > 0)
    return out


def h3(df: pd.DataFrame, half_life: float = config.HALF_LIFE_TOKENS) -> dict:
    s = wide(df.assign(s=with_half_life(df, half_life)), "s")
    tests = {(b, e): paired(s, f"gen_b{b}", f"gen_b{e}") for b in (5, 10, 20) for e in (2, 40)}
    adjusted = dict(zip(tests, holm([t["p_greater"] for t in tests.values()])))
    winners = [b for b in (5, 10, 20) if adjusted[(b, 2)] < 0.05 and adjusted[(b, 40)] < 0.05]
    # Why S falls from ~2 to ~10 words: ratio of mean J vs ratio of the length factor at median L.
    med = df.groupby("condition").n_tokens.median()
    j_mean = wide(df, "j").mean()
    illustration = {"j_ratio_b10_over_b2": float(j_mean["gen_b10"] / j_mean["gen_b2"]),
                    "factor_ratio_b10_over_b2": 2 ** (-(med["gen_b10"] - med["gen_b2"]) / half_life)}
    return {"illustration": illustration,"s_by_budget": {c: mean_ci(s[c].dropna()) for c in BUDGET_CONDITIONS + ["gen_unconstrained"]},
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


def crossfit_selection(df: pd.DataFrame, halves: dict, half_life: float) -> pd.DataFrame:
    """Select a candidate per latent on one half, evaluate its J on the other, both ways.
    Rules: argmax J, argmax S, argmin L. Ties go to the shorter explanation."""
    out = []
    for latent, group in df.groupby("latent"):
        rows = group.to_dict("records")
        evaluations = {rule: [] for rule in ("argmax_j", "argmax_s", "argmin_l")}
        for select_ids, eval_ids in (halves[latent], halves[latent][::-1]):
            sel = [explanation_metrics(r, half_life, select_ids) | {"r": r} for r in rows]
            picks = {"argmax_j": max(sel, key=lambda m: (m["j"], -m["n_tokens"])),
                     "argmax_s": max(sel, key=lambda m: (m["s"], -m["n_tokens"])),
                     "argmin_l": min(sel, key=lambda m: (m["n_tokens"], -m["j"]))}
            for rule, pick in picks.items():
                evaluated = explanation_metrics(pick["r"], half_life, eval_ids)
                evaluations[rule].append((evaluated["j"], pick["n_tokens"], pick["r"]["condition"]))
        for rule, ev in evaluations.items():
            out.append({"rule": rule, "latent": latent, "j": float(np.mean([e[0] for e in ev])),
                        "n_tokens": float(np.mean([e[1] for e in ev])), "picked": [e[2] for e in ev]})
    return pd.DataFrame(out)


def h4(df: pd.DataFrame, half_life: float = config.HALF_LIFE_TOKENS) -> dict:
    cands = df[df.condition.isin(CANDIDATES)]
    sel = crossfit_selection(cands, crossfit_halves(df), half_life)
    j, length = wide(sel.rename(columns={"rule": "condition"}), "j"), wide(sel.rename(columns={"rule": "condition"}), "n_tokens")
    loss = (j["argmax_j"] - j["argmax_s"]).dropna()
    out = {"j": {r: mean_ci(j[r]) for r in j.columns},
           "median_tokens": {r: float(length[r].median()) for r in length.columns},
           "j_loss_argmaxJ_minus_argmaxS": mean_ci(loss),
           "argmaxS_minus_argminL": paired(j, "argmax_s", "argmin_l"),
           "picked_counts": {r: pd.Series(sum(sel[sel.rule == r].picked, [])).value_counts().to_dict()
                             for r in ("argmax_j", "argmax_s", "argmin_l")}}
    ratio = out["median_tokens"]["argmax_s"] / out["median_tokens"]["argmax_j"]
    out["length_ratio_S_over_J"], out["length_reduction"] = ratio, 1 - ratio
    out["n_selections"] = 2 * int(sel.latent.nunique())  # two cross-fit directions per latent
    out["supported"] = bool(ratio <= 0.8 and out["j_loss_argmaxJ_minus_argmaxS"]["mean"] <= 0.05
                            and out["argmaxS_minus_argminL"]["lo"] > 0)
    return out


# ---------- exploratory ----------

def exploratory(df: pd.DataFrame) -> dict:
    j, s = wide(df, "j"), wide(df, "s")
    sweep = {}
    for h in config.H_SWEEP:
        sh = wide(df.assign(s=with_half_life(df, h)), "s")
        means = {c: float(sh[c].mean()) for c in sh.columns}
        best_budget = max(BUDGET_CONDITIONS, key=lambda c: means.get(c, -1))
        # Post-hoc (not pre-registered): is the best budget better than both endpoints at this H?
        tests = {f"best_minus_{e}": paired(sh, best_budget, e) for e in ("gen_b2", "gen_b40") if e != best_budget}
        sweep["inf" if math.isinf(h) else str(int(h))] = {"mean_s": means, "best_budget": best_budget, "tests": tests}
    # Length unit robustness: re-express H in words/chars with the mean words-per-token ratio.
    w_per_t = df.n_words.sum() / df.n_tokens.sum()
    c_per_t = df.n_chars.sum() / df.n_tokens.sum()
    h = config.HALF_LIFE_TOKENS
    s_words = [length_penalized_score(a, n, h * w_per_t) for a, n in zip(df.j, df.n_words)]
    s_chars = [length_penalized_score(a, n, h * c_per_t) for a, n in zip(df.j, df.n_chars)]
    valid = df.s.notna()
    return {
        "gpt4omini_minus_gpt35_J": paired(j, "gpt4omini", "gpt35") | {"two_sided_p": float(stats.wilcoxon((j.gpt4omini - j.gpt35).dropna()).pvalue)},
        "gpt4omini_minus_gpt35_S": paired(s, "gpt4omini", "gpt35") | {"two_sided_p": float(stats.wilcoxon((s.gpt4omini - s.gpt35).dropna()).pvalue)},
        "h_sweep": sweep,
        "unit_robustness_spearman": {
            "tokens_vs_words": float(stats.spearmanr(df.s[valid], np.array(s_words)[valid]).statistic),
            "tokens_vs_chars": float(stats.spearmanr(df.s[valid], np.array(s_chars)[valid]).statistic),
            "words_per_token": float(w_per_t), "chars_per_token": float(c_per_t)},
        "spearman_j_vs_tokens_all": float(stats.spearmanr(df.j[valid], df.n_tokens[valid]).statistic),
        "n_humans": int(df.condition.str.startswith("human").sum()),
        "humans": df[df.condition.str.startswith("human")][["latent", "condition", "explanation", "j", "n_tokens", "s"]].to_dict("records"),
    }


def reliability(main: pd.DataFrame, other: pd.DataFrame) -> dict:
    m = main.merge(other, on=["latent", "condition"], suffixes=("", "_2"))
    out = {"n_explanations": int(len(m)), "n_latents": int(m.latent.nunique())}
    for col in ("j", "s"):
        ok = m[col].notna() & m[f"{col}_2"].notna()
        out[f"spearman_{col}"] = float(stats.spearmanr(m[col][ok], m[f"{col}_2"][ok]).statistic)
    out["mean_s_by_budget_other"] = {c: float(other[other.condition == c].s.mean()) for c in BUDGET_CONDITIONS}
    return out


def condition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition: mean BA, J, tokens and S with CIs (README/paper main table)."""
    rows = []
    for cond, g in df.groupby("condition"):
        if cond.startswith("human"):
            continue
        rows.append({"condition": cond, "n": len(g), "ba": g.ba.mean(), **{f"j_{k}": v for k, v in mean_ci(g.j.dropna()).items() if k != "n"},
                     "tokens_median": g.n_tokens.median(), **{f"s_{k}": v for k, v in mean_ci(g.s.dropna()).items() if k != "n"}})
    return pd.DataFrame(rows)


def export_explorer_data(df: pd.DataFrame, path: Path) -> None:
    """docs/data.json for the GitHub Pages explorer (it recomputes S in the browser for any H)."""
    latents = [{"latent": int(lat), "explanations": [
        {"condition": r.condition, "explanation": r.explanation, "j": None if math.isnan(r.j) else round(r.j, 4),
         "ba": None if math.isnan(r.ba) else round(r.ba, 4), "n_tokens": int(r.n_tokens)}
        for r in g.itertuples()]} for lat, g in df.groupby("latent")]
    path.write_text(json.dumps({"half_life_default": config.HALF_LIFE_TOKENS, "latents": latents}))


def api_usage(log_glob: str) -> dict:
    """OpenRouter calls and cost summed from the META lines of the run logs (logs/ is local;
    the totals are written into numbers.json so a fresh clone still has them)."""
    calls, cost, tokens = 0, 0.0, 0
    for path in sorted(config.LOG_DIR.glob(log_glob)):
        for line in path.open(encoding="utf-8"):
            if " META [" in line and "usage=" in line:
                usage = json.loads(line.split("usage=", 1)[1])
                calls, cost, tokens = calls + 1, cost + usage.get("cost", 0.0), tokens + usage.get("total_tokens", 0)
    return {"calls": calls, "cost_usd": cost, "total_tokens": tokens}


def neuronpedia_zero_rows() -> dict:
    """Diversity of Neuronpedia's zero-activation rows for the sampled latents (why PREREG
    deviation 2 replaced them): total rows vs distinct texts."""
    from .data import load_processed

    latents, _ = load_processed()
    texts = ["".join(r["tokens"]) for d in latents.values() for r in d["rows"] if r["maxValue"] == 0]
    return {"n_latents": len(latents), "rows": len(texts), "distinct_texts": len(set(texts))}


def data_summary() -> dict:
    samples = json.loads((config.PROCESSED_DIR / "samples.json").read_text())
    from .negatives import load_report  # avoid importing torch/sae_lens just for analysis
    return {"n_latents_total": samples["n_latents"], "n_eligible": samples["n_eligible"],
            "eligible_pct": 100 * samples["n_eligible"] / samples["n_latents"], "negatives": load_report(),
            "neuronpedia_zero_rows": neuronpedia_zero_rows()}


def run_analysis(sample: str = "main") -> dict:
    out_dir = config.RESULTS_DIR / sample
    df = load_scores(out_dir / "scores.jsonl")
    numbers = {
        "data": data_summary(),
        "api_usage": {"score": api_usage(f"*score_{sample}.log"), "reliability": api_usage("*_reliability.log")},
        "n_latents": int(df.latent.nunique()), "n_explanations": int(len(df)),
        "sample_failure_rate": float(df.n_failed.sum() / (df.n_failed.sum() + df.n_valid.sum())),
        "flagged_explanations": int((df.n_failed > 0.2 * (df.n_failed + df.n_valid)).sum()),
        "half_life": config.HALF_LIFE_TOKENS,
        "conditions": condition_table(df).to_dict("records"),
        "by_condition": condition_table(df).set_index("condition").to_dict("index"),  # for templates
        "H1": h1(df), "H2": h2(df), "H3": h3(df), "H4": h4(df), "exploratory": exploratory(df),
    }
    for name in ("retest", "altscorer"):
        path = out_dir / f"scores_{name}.jsonl"
        if path.exists():
            numbers[f"reliability_{name}"] = reliability(df, load_scores(path))
    rel = [numbers[k][f"spearman_{c}"] for k in ("reliability_retest", "reliability_altscorer") if k in numbers for c in "js"]
    numbers["reliability_min_spearman"] = min(rel) if rel else None
    (out_dir / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    export_explorer_data(df, (config.ROOT / "docs" if sample == "main" else out_dir) / "data.json")
    log.info("analysis written to %s", out_dir / "numbers.json")
    from .figures import make_figures  # matplotlib only needed here
    # The pilot is for debugging only: its figures never overwrite the main ones.
    make_figures(df, numbers, config.FIGURES_DIR if sample == "main" else out_dir / "figures")
    return numbers
