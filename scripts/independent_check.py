"""Independent re-implementation of the headline numbers, checked against results/main/numbers.json.

Global context: written for the 2026-11-22 audit (reports/reviews/2026-11-22-independent-audit.md).
It does not import the project's package. BA, J, L, S, the bootstrap, the Wilcoxon tests, Holm
and the H4 cross-fit are re-implemented from the PREREG text, using only tiktoken, numpy and
scipy, so a bug in src/concise_interp/ would show up as a mismatch here. No model or API calls.

    uv run python scripts/independent_check.py      # exit code 1 if any value differs
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import tiktoken
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ENC = tiktoken.get_encoding("o200k_base")  # PREREG section 2: L in o200k_base tokens of e.strip()
BUDGETS = (2, 5, 10, 20, 40)
CANDIDATES = [f"gen_b{b}" for b in BUDGETS] + ["gen_unconstrained", "gpt35", "gpt4omini"]  # PREREG H4


def j_of(samples, ids=None):
    """Youden's J = TPR - FPR over valid predictions (optionally a subset of ids); NaN if a class is empty."""
    s = [x for x in samples if (ids is None or x["id"] in ids) and x["prediction"] is not None]
    pos = [bool(x["prediction"]) for x in s if x["activating"]]
    neg = [bool(x["prediction"]) for x in s if not x["activating"]]
    return np.mean(pos) - np.mean(neg) if pos and neg else math.nan


def s_of(j, length, h):
    """S = max(J, 0) * 2^(-L/H); H = inf leaves max(J, 0)."""
    return math.nan if math.isnan(j) else max(j, 0.0) * (1.0 if math.isinf(h) else 2 ** (-length / h))


def boot(x, n_boot=10_000, seed=0):
    """Mean and 95% percentile bootstrap CI over latents (same resampling scheme as PREREG section 6)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    means = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return [x.mean(), *np.quantile(means, [0.025, 0.975])]


def holm(p):
    """Holm step-down adjusted p-values in input order."""
    adj, running = [0.0] * len(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main() -> int:
    rows = [json.loads(line) for line in open(ROOT / "results/main/scores.jsonl", encoding="utf-8")]
    nums = json.loads((ROOT / "results/main/numbers.json").read_text())
    tab: dict[int, dict[str, dict]] = {}  # latent -> condition -> {j, L, samples}
    for r in rows:
        tab.setdefault(r["latent"], {})[r["condition"]] = {
            "j": j_of(r["samples"]), "L": len(ENC.encode(r["explanation"].strip(), disallowed_special=())),
            "samples": r["samples"]}
    latents = sorted(tab)  # latent order, as the project's bootstrap uses

    def vec(cond, h=None):
        """Per-latent J (h is None) or S at half-life h, in latent order."""
        return np.array([tab[l][cond]["j"] if h is None else s_of(tab[l][cond]["j"], tab[l][cond]["L"], h)
                         for l in latents], float)

    checks = []  # (name, ours, numbers.json)
    for c in ["random", "top_token", "gpt35", "gpt4omini", *CANDIDATES[:6]]:
        bc = nums["by_condition"][c]
        checks += [(f"{c} J mean/lo/hi", boot(vec(c)), [bc["j_mean"], bc["j_lo"], bc["j_hi"]]),
                   (f"{c} S mean/lo/hi", boot(vec(c, 20.0)), [bc["s_mean"], bc["s_lo"], bc["s_hi"]]),
                   (f"{c} median tokens", [np.median([tab[l][c]["L"] for l in latents])], [bc["tokens_median"]])]
    # H2: rise and diminishing gain per nominal word.
    d = np.round(vec("gen_b40") - vec("gen_b2"), 12)  # rounded: float noise must not split ties or zeros
    rise = nums["H2"]["rise_b40_minus_b2"]
    checks.append(("H2 rise mean/lo/hi/p", boot(d) + [stats.wilcoxon(d, alternative="greater").pvalue],
                   [rise["mean"], rise["lo"], rise["hi"], rise["p_greater"]]))
    gain = (vec("gen_b10") - vec("gen_b2")) / 8 - (vec("gen_b40") - vec("gen_b10")) / 30
    g = nums["H2"]["diminishing_gain_per_word"]
    checks.append(("H2 diminishing gain", boot(gain), [g["mean"], g["lo"], g["hi"]]))
    # H3: 6 one-sided tests, Holm over the family.
    keys = [(b, e) for b in BUDGETS[1:-1] for e in (2, 40)]
    diffs = [np.round(vec(f"gen_b{b}", 20.0) - vec(f"gen_b{e}", 20.0), 12) for b, e in keys]
    ps = [stats.wilcoxon(x, alternative="greater").pvalue for x in diffs]
    for (b, e), x, p, a in zip(keys, diffs, ps, holm(ps)):
        t = nums["H3"]["tests"][f"b{b}_vs_b{e}"]
        checks.append((f"H3 b{b}-b{e} mean/lo/hi/p/holm", boot(x) + [p, a],
                       [t["mean"], t["lo"], t["hi"], t["p_greater"], t["p_holm"]]))
    # H4: class-stratified halves (seed 0), select on one half, evaluate J on the other, swap.
    # Equal J from different confusion counts differs in the last float bits: compare after rounding.
    rank = lambda v: -math.inf if math.isnan(v) else round(v, 12)
    res = {r: [] for r in ("argmax_j", "argmax_s", "argmin_l")}
    for lat in latents:
        ids = {x["id"]: x["activating"] for x in tab[lat]["gpt35"]["samples"]}
        half_a, half_b = set(), set()
        for label in (True, False):
            members = sorted(i for i, act in ids.items() if act == label)
            random.Random(f"0-{lat}-{label}").shuffle(members)
            half_a |= set(members[: len(members) // 2])
            half_b |= set(members[len(members) // 2:])
        order = sorted(CANDIDATES)  # exact ties in score and length: seeded shuffle (PREREG clarification 2026-11-22)
        random.Random(f"0-{lat}-ties").shuffle(order)
        per_rule = {r: [] for r in res}
        for sel, ev in ((half_a, half_b), (half_b, half_a)):
            m = [(c, j_of(tab[lat][c]["samples"], sel), tab[lat][c]["L"]) for c in order]
            picks = {"argmax_j": max(m, key=lambda t: (rank(t[1]), -t[2])),
                     "argmax_s": max(m, key=lambda t: (rank(s_of(t[1], t[2], 20.0)), -t[2])),
                     "argmin_l": min(m, key=lambda t: (t[2], -rank(t[1])))}
            for r, (c, _, length) in picks.items():
                per_rule[r].append((j_of(tab[lat][c]["samples"], ev), length))
        for r, v in per_rule.items():
            res[r].append((np.mean([x[0] for x in v]), np.mean([x[1] for x in v])))
    j4 = {r: np.array([x[0] for x in v]) for r, v in res.items()}
    med = {r: np.median([x[1] for x in v]) for r, v in res.items()}
    h4 = nums["H4"]
    for r in res:
        checks.append((f"H4 {r} J mean/lo/hi, median L", boot(j4[r]) + [med[r]],
                       [h4["j"][r]["mean"], h4["j"][r]["lo"], h4["j"][r]["hi"], h4["median_tokens"][r]]))
    loss = h4["j_loss_argmaxJ_minus_argmaxS"]
    checks.append(("H4 J loss", boot(j4["argmax_j"] - j4["argmax_s"]), [loss["mean"], loss["lo"], loss["hi"]]))
    gap = h4["argmaxS_minus_argminL"]
    checks.append(("H4 S minus shortest", boot(j4["argmax_s"] - j4["argmin_l"]), [gap["mean"], gap["lo"], gap["hi"]]))

    bad = 0
    for name, ours, theirs in checks:
        ok = np.allclose(ours, theirs, rtol=1e-9, atol=1e-12)
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name}: {np.round(ours, 4).tolist()} vs {np.round(theirs, 4).tolist()}")
    print(f"{len(checks) - bad} of {len(checks)} checks match")
    return int(bad > 0)


if __name__ == "__main__":
    sys.exit(main())
