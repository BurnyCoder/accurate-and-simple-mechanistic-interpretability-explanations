"""The analysis phase: every reported number is computed here or in the modules it calls.

Global context: last phase of `run.py`. Reads results/<sample>/scores*.jsonl (one row per
scored explanation, with every per-sample prediction), recomputes balanced accuracy, J, L and S
(`metrics.py`), runs the pre-registered hypotheses (`hypotheses.py`, PREREG section 6), the
reliability checks (section 7) and the exploratory analyses, and writes results/<sample>/numbers.json,
which README, reports and paper are rendered from (`render.py`), plus figures and docs/data.json.
"""

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .conditions import BUDGET_CONDITIONS
from .hypotheses import INTERMEDIATE, all_hypotheses, crossfit_halves, h4_split_sensitivity
from .jsonl import read_jsonl
from .length_score import length_penalized_score
from .metrics import explanation_metrics, load_scores, s_wide, without_samples
from .stats import agreement, mean_ci, paired, two_sided, wide

log = logging.getLogger(__name__)


# ---------- exploratory (not pre-registered as confirmatory) ----------

def best_budget_counts(s: pd.DataFrame) -> dict[str, int]:
    """Per latent, the budget with the highest S; ties go to the shorter budget (idxmax takes the
    first column). Counts per budget, over latents that have all five budgets."""
    best = s[BUDGET_CONDITIONS].dropna().idxmax(axis=1)
    return {c: int((best == c).sum()) for c in BUDGET_CONDITIONS}


def robustness_without(df: pd.DataFrame, drop: dict[int, set[str]]) -> dict:
    """H1-H4 re-run after dropping some test samples; returns the verdicts and headline means."""
    hyp = all_hypotheses(without_samples(df, drop))
    return {"supported": {k: v["supported"] for k, v in hyp.items()},
            "random_j": hyp["H1"]["random"]["mean"],
            "s_by_budget": {c: v["mean"] for c, v in hyp["H3"]["s_by_budget"].items()},
            "h4_j_loss": hyp["H4"]["j_loss_argmaxJ_minus_argmaxS"]["mean"],
            "h4_length_reduction": hyp["H4"]["length_reduction"]}


def data_checks(df: pd.DataFrame) -> dict:
    """Test-set audits (checks.py) on the scored latents' splits, and H1-H4 without the test
    positives that contain <|endoftext|> (a token our random negatives never contain)."""
    from .checks import positive_rates, split_checks
    from .data import load_processed
    from .experiment import prepare_records  # loads torch via delphi's records

    latents_data, _ = load_processed()
    _, sample_ids, splits = prepare_records(latents_data, [int(x) for x in sorted(df.latent.unique())])
    summary, eot_ids = split_checks(splits, sample_ids)
    summary["eot"]["random_positive_rate"] = positive_rates(df, "random", eot_ids)
    summary["without_eot_positives"] = robustness_without(df, eot_ids)
    return summary


def h_sweep(df: pd.DataFrame) -> dict:
    """Mean S of every condition at each H of the sweep, the budget with the highest mean, and
    (post hoc, not pre-registered) paired tests of that budget against both endpoints."""
    sweep = {}
    for h in config.H_SWEEP:
        sh = s_wide(df, h)
        means = {c: float(sh[c].mean()) for c in sh.columns}
        best_budget = max(BUDGET_CONDITIONS, key=lambda c: means.get(c, -1))
        tests = {f"best_minus_{e}": paired(sh, best_budget, e) for e in ("gen_b2", "gen_b40") if e != best_budget}
        sweep["inf" if math.isinf(h) else str(int(h))] = {"mean_s": means, "best_budget": best_budget,
                                                          "best_budget_words": int(best_budget.removeprefix("gen_b")),
                                                          "tests": tests}
    return sweep


def unit_robustness(df: pd.DataFrame) -> dict:
    """Spearman of S with L in tokens vs words vs characters; H is re-expressed in the other unit
    with the mean words-per-token (characters-per-token) ratio."""
    w_per_t = df.n_words.sum() / df.n_tokens.sum()
    c_per_t = df.n_chars.sum() / df.n_tokens.sum()
    h = config.HALF_LIFE_TOKENS
    s_words = [length_penalized_score(a, n, h * w_per_t) for a, n in zip(df.j, df.n_words)]
    s_chars = [length_penalized_score(a, n, h * c_per_t) for a, n in zip(df.j, df.n_chars)]
    valid = df.s.notna()
    return {"tokens_vs_words": float(stats.spearmanr(df.s[valid], np.array(s_words)[valid]).statistic),
            "tokens_vs_chars": float(stats.spearmanr(df.s[valid], np.array(s_chars)[valid]).statistic),
            "words_per_token": float(w_per_t), "chars_per_token": float(c_per_t)}


def exploratory(df: pd.DataFrame) -> dict:
    """Everything PREREG lists as exploratory, plus post-hoc checks; no confirmatory claims."""
    j, s = wide(df, "j"), wide(df, "s")
    valid = df.s.notna()
    best = best_budget_counts(s)
    humans = df[df.condition.str.startswith("human")]
    return {
        # Per-latent view of H3 (default H): mean S is monotone, but which budget wins for each latent?
        "per_latent_best_budget": best,
        "per_latent_best_intermediate": sum(best[f"gen_b{b}"] for b in INTERMEDIATE),
        "h4_split_sensitivity": h4_split_sensitivity(df),
        "gpt4omini_minus_gpt35_J": two_sided(j, "gpt4omini", "gpt35"),
        "gpt4omini_minus_gpt35_S": two_sided(s, "gpt4omini", "gpt35"),
        # Post hoc: the top-token baseline against Neuronpedia's explanations (J, and S at the default H).
        "top_token_vs_existing": {f"minus_{c}_{k}": two_sided(w, "top_token", c)
                                  for c in ("gpt35", "gpt4omini") for k, w in (("J", j), ("S", s))},
        "h_sweep": h_sweep(df),
        "unit_robustness_spearman": unit_robustness(df),
        "spearman_j_vs_tokens_all": float(stats.spearmanr(df.j[valid], df.n_tokens[valid]).statistic),
        "n_humans": int(len(humans)),
        "humans": humans[["latent", "condition", "explanation", "j", "n_tokens", "s"]].to_dict("records"),
    }


# ---------- reliability (PREREG section 7) ----------

def split_half_reliability(df: pd.DataFrame, halves: dict, half_life: float = config.HALF_LIFE_TOKENS) -> dict:
    """Agreement between J (and S) of each explanation on the two disjoint cross-fit halves of its
    latent's contexts. The re-scorings reuse the same 50 contexts, so they miss the noise from which
    contexts were drawn; this includes it, on 25 contexts per half (a lower bound for 50).
    The random control is left out: its J is ~0 on both halves and adds only between-condition spread."""
    rows = []
    for r in df[df.condition != "random"].to_dict("records"):
        a, b = (explanation_metrics(r, half_life, ids) for ids in halves[r["latent"]])
        rows.append({"latent": r["latent"], "j": a["j"], "j_2": b["j"], "s": a["s"], "s_2": b["s"]})
    return agreement(pd.DataFrame(rows))


def reliability(main: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Agreement of the main scoring with a re-scoring of the same explanations and contexts."""
    m = main.merge(other, on=["latent", "condition"], suffixes=("", "_2"))
    out = agreement(m)
    out["mean_s_by_budget_other"] = {c: float(other[other.condition == c].s.mean()) for c in BUDGET_CONDITIONS}
    # The main run's means on the same latents, so the re-scored means have a like-for-like reference.
    same = main[main.latent.isin(set(other.latent))]
    out["mean_s_by_budget_main_subset"] = {c: float(same[same.condition == c].s.mean()) for c in BUDGET_CONDITIONS}
    return out


# ---------- tables, provenance and outputs ----------

def condition_ci(df: pd.DataFrame, col: str, condition: str) -> dict:
    """Mean and CI of `col` for one condition, over latents in latent order: the same vector
    H1-H3 bootstrap, so a mean has one CI everywhere (scores.jsonl rows are in API-completion order)."""
    return mean_ci(wide(df, col)[condition].dropna())


def condition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition: mean BA, J, tokens and S with CIs (README/paper main table)."""
    rows = []
    for cond, g in df.groupby("condition"):
        if cond.startswith("human"):
            continue
        rows.append({"condition": cond, "n": len(g), "ba": g.ba.mean(),
                     **{f"j_{k}": v for k, v in condition_ci(df, "j", cond).items() if k != "n"},
                     "tokens_median": g.n_tokens.median(),
                     **{f"s_{k}": v for k, v in condition_ci(df, "s", cond).items() if k != "n"}})
    return pd.DataFrame(rows)


def export_explorer_data(df: pd.DataFrame, path: Path) -> None:
    """docs/data.json for the GitHub Pages explorer (it recomputes S in the browser for any H)."""
    latents = [{"latent": int(lat), "explanations": [
        {"condition": r.condition, "explanation": r.explanation, "j": None if math.isnan(r.j) else round(r.j, 4),
         "ba": None if math.isnan(r.ba) else round(r.ba, 4), "n_tokens": int(r.n_tokens)}
        for r in g.itertuples()]} for lat, g in df.groupby("latent")]
    path.write_text(json.dumps({"half_life_default": config.HALF_LIFE_TOKENS, "latents": latents}))


def api_usage(log_glob: str, previous: dict | None = None) -> dict:
    """OpenRouter calls, cost and serving providers summed from the META lines of the run logs,
    plus delphi's failure lines: "Parsing selections failed" (a response without a parsable answer)
    and "Error generating text" (an API call that failed after all retries).
    logs/ is not in git, so when no logged call matches, `previous` (the committed numbers.json
    value) is returned instead of zeros; a fresh clone's `analyze` keeps the real totals.
    Providers are counted per output: "explain" until scoring.py logs "score -> <file>.jsonl:",
    then that file's stem (no dot, so templates can address it as a dotted path)."""
    calls, cost, tokens, parse_errors, api_errors = 0, 0.0, 0, 0, 0
    providers: defaultdict[str, Counter] = defaultdict(Counter)
    for path in sorted(config.LOG_DIR.glob(log_glob)):
        section = "explain"
        for line in path.open(encoding="utf-8"):
            if "score -> " in line:
                section = line.split("score -> ", 1)[1].split(":", 1)[0].removesuffix(".jsonl")
            if " META [" in line and "usage=" in line:
                usage = json.loads(line.split("usage=", 1)[1])
                calls, cost, tokens = calls + 1, cost + usage.get("cost", 0.0), tokens + usage.get("total_tokens", 0)
                providers[section][line.split("provider=", 1)[1].split(" usage=", 1)[0]] += 1
            parse_errors += " ERROR delphi: Parsing selections failed" in line
            api_errors += " ERROR delphi: Error generating text" in line
    if calls == 0 and previous:
        return previous
    return {"calls": calls, "cost_usd": cost, "total_tokens": tokens,
            "providers": {k: dict(v.most_common()) for k, v in providers.items()},
            "parse_errors": parse_errors, "api_errors": api_errors}


def neuronpedia_zero_rows() -> dict:
    """Diversity of Neuronpedia's zero-activation rows for the sampled latents (why PREREG
    deviation 2 replaced them): total rows vs distinct texts."""
    from .data import load_processed

    latents, _ = load_processed()
    texts = ["".join(r["tokens"]) for d in latents.values() for r in d["rows"] if r["maxValue"] == 0]
    return {"n_latents": len(latents), "rows": len(texts), "distinct_texts": len(set(texts))}


def generated_summary(path: Path) -> dict:
    """How many Exp 2 explanations were generated and how many delphi could parse."""
    rows = read_jsonl(path)
    return {"n": len(rows), "parsed": sum(r["parsed"] for r in rows)}


def data_summary() -> dict:
    """Eligibility counts, negative-verification report and Neuronpedia zero-row diversity."""
    samples = json.loads((config.PROCESSED_DIR / "samples.json").read_text())
    from .negatives import load_report  # avoid importing torch/sae_lens just for analysis
    n = samples["n_latents"]
    # How many latents pass each eligibility filter on its own (data.prepare); n_eligible passes all.
    filters = {k: {"n": v, "pct": 100 * v / n} for k, v in samples.get("filters", {}).items()}
    return {"n_latents_total": n, "n_eligible": samples["n_eligible"],
            "eligible_pct": 100 * samples["n_eligible"] / n, "filters": filters, "negatives": load_report(),
            "neuronpedia_zero_rows": neuronpedia_zero_rows()}


def run_analysis(sample: str = "main") -> dict:
    """Phase `analyze`: numbers.json, docs/data.json (main only) and the figures for one sample."""
    out_dir = config.RESULTS_DIR / sample
    df = load_scores(out_dir / "scores.jsonl")
    old_path = out_dir / "numbers.json"
    previous = json.loads(old_path.read_text()).get("api_usage", {}) if old_path.exists() else {}
    table = condition_table(df)
    numbers = {
        "data": data_summary(),
        "api_usage": {"score": api_usage(f"*score_{sample}.log", previous.get("score")),
                      "reliability": api_usage("*_reliability.log", previous.get("reliability"))},
        "n_latents": int(df.latent.nunique()), "n_explanations": int(len(df)),
        "generated": generated_summary(out_dir / "generated_explanations.jsonl"),
        "sample_failure_rate": float(df.n_failed.sum() / (df.n_failed.sum() + df.n_valid.sum())),
        "flagged_explanations": int((df.n_failed > 0.2 * (df.n_failed + df.n_valid)).sum()),
        "half_life": config.HALF_LIFE_TOKENS,
        "conditions": table.to_dict("records"),
        "by_condition": table.set_index("condition").to_dict("index"),  # for templates
        **all_hypotheses(df), "exploratory": exploratory(df),
    }
    numbers["checks"] = data_checks(df)
    numbers["reliability_split_half"] = split_half_reliability(df, crossfit_halves(df))
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
