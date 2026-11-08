"""Statistics used by every analysis: bootstrap CIs, paired Wilcoxon tests, Holm, Spearman agreement.

Global context: PREREG section 6 fixes the inference: 95% percentile bootstrap CIs over latents
(10,000 resamples, seed 0), paired Wilcoxon signed-rank tests and Holm correction. `hypotheses.py`
and `analysis.py` call these helpers on latent x condition tables built by `wide`.
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def bootstrap_ci(values, seed: int = config.SEED_ANALYSIS, n_boot: int = config.N_BOOTSTRAP, level: float = 0.95):
    """Percentile bootstrap CI of the mean (resampling latents; Efron 1979,
    https://doi.org/10.1214/aos/1176344552). The resamples depend on the order of `values`,
    so callers pass latent-ordered vectors (`wide`) to get one CI per quantity everywhere."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return tuple(np.quantile(means, [(1 - level) / 2, (1 + level) / 2]).tolist())


def mean_ci(values) -> dict:
    """Mean, bootstrap CI and n of a vector of per-latent values."""
    values = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(values)
    return {"mean": float(values.mean()), "lo": lo, "hi": hi, "n": int(len(values))}


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (Holm 1979, https://www.jstor.org/stable/4615733), in the input order."""
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
    """Mean of a - b over latents having both, with bootstrap CI and one-sided Wilcoxon
    signed-rank test (a > b; Wilcoxon 1945, https://doi.org/10.2307/3001968)."""
    d = (df_wide[a] - df_wide[b]).dropna()
    p = float(stats.wilcoxon(d, alternative="greater").pvalue) if (d != 0).any() else 1.0
    return {**mean_ci(d), "p_greater": p}


def two_sided(df_wide: pd.DataFrame, a: str, b: str) -> dict:
    """`paired` plus the two-sided Wilcoxon p (used for post-hoc comparisons with no direction)."""
    d = (df_wide[a] - df_wide[b]).dropna()
    return paired(df_wide, a, b) | {"two_sided_p": float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0}


def agreement(m: pd.DataFrame) -> dict:
    """Spearman between two scorings (columns j/j_2, s/s_2) of the same explanations, pooled and
    within latent. Pooled correlations include the spread between latents (some are simply easier
    to detect), which inflates them; the within-latent one correlates each latent's values minus
    that latent's mean, i.e. how well the scorings agree on which explanation of a latent is better."""
    out = {"n_explanations": int(len(m)), "n_latents": int(m.latent.nunique())}
    for col in ("j", "s"):
        ok = m[m[col].notna() & m[f"{col}_2"].notna()]
        centred = [ok[c] - ok.groupby("latent")[c].transform("mean") for c in (col, f"{col}_2")]
        out[f"spearman_{col}"] = float(stats.spearmanr(ok[col], ok[f"{col}_2"]).statistic)
        out[f"spearman_{col}_within"] = float(stats.spearmanr(*centred).statistic)
    return out
