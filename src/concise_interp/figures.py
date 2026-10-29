"""All figures, drawn only from the analysed results (no hand-typed numbers).

Global context: called at the end of `analysis.run_analysis`. Writes PNGs to figures/, which
README.md and the paper embed. Colours follow a validated colorblind-safe categorical order
(blue, orange, aqua) and a single-hue blue ramp for the ordinal H sweep; one y-axis per panel.
"""

import math

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to files, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .analysis import BUDGET_CONDITIONS, mean_ci, wide, with_half_life

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]  # H = 5..80, light -> dark
LABELS = {"gpt35": "gpt-3.5-turbo (Neuronpedia)", "gpt4omini": "gpt-4o-mini (Neuronpedia)",
          "top_token": "top-token baseline", "random": "random control",
          "gen_unconstrained": "Llama-70B unconstrained",
          **{f"gen_b{b}": f"Llama-70B ~{b} words" for b in config.BUDGETS}}
ORDER = ["random", "top_token", "gpt35", "gpt4omini", *BUDGET_CONDITIONS, "gen_unconstrained"]


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _dot_ci(ax, labels, stats_list, color, xlabel):
    y = np.arange(len(labels))
    means = [s["mean"] for s in stats_list]
    err = [[m - s["lo"] for m, s in zip(means, stats_list)], [s["hi"] - m for m, s in zip(means, stats_list)]]
    ax.errorbar(means, y, xerr=err, fmt="o", color=color, ms=6, lw=1.5, capsize=0)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(xlabel)
    _style(ax)


def fig_conditions(df, path):
    conds = [c for c in ORDER if c in set(df.condition)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.38 * len(conds) + 1.4), sharey=True)
    for ax, col, color, label in ((axes[0], "j", BLUE, "detection score J = 2·BA − 1 (mean, 95% CI)"),
                                  (axes[1], "s", ORANGE, f"length-penalized S, H = {config.HALF_LIFE_TOKENS:g} tokens")):
        _dot_ci(ax, [LABELS[c] for c in conds], [mean_ci(df[df.condition == c][col].dropna()) for c in conds], color, label)
    fig.suptitle("Accuracy alone vs accuracy × brevity, per explanation source", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_budget_curves(df, path):
    tok = [df[df.condition == c].n_tokens.median() for c in BUDGET_CONDITIONS]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for col, color, name in (("j", BLUE, "J (accuracy only)"), ("s", ORANGE, f"S (H = {config.HALF_LIFE_TOKENS:g})")):
        st = [mean_ci(df[df.condition == c][col].dropna()) for c in BUDGET_CONDITIONS]
        ax.fill_between(tok, [s["lo"] for s in st], [s["hi"] for s in st], color=color, alpha=0.15, lw=0)
        ax.plot(tok, [s["mean"] for s in st], "-o", color=color, lw=2, ms=6)
        ax.annotate(name, (tok[-1], st[-1]["mean"]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    for t, b in zip(tok, config.BUDGETS):
        ax.annotate(f"~{b}w", (t, 0), xytext=(0, 4), textcoords="offset points", ha="center", color=MUTED, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("median explanation length (o200k tokens), log scale")
    ax.set_ylabel("mean score over latents (95% CI)")
    ax.set_ylim(bottom=0)
    ax.set_title("Exp 2: same latent, explanations asked to be ~2 … ~40 words", color=INK, fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_h_sweep(df, path):
    tok = [df[df.condition == c].n_tokens.median() for c in BUDGET_CONDITIONS]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    finite = [h for h in config.H_SWEEP if not math.isinf(h)]
    for h, color in [*zip(finite, RAMP), (math.inf, INK)]:
        s = wide(df.assign(s=with_half_life(df, h)), "s")
        means = [s[c].mean() for c in BUDGET_CONDITIONS]
        ax.plot(tok, means, "-o", color=color, lw=2, ms=5)
        ax.annotate("H = ∞ (plain J)" if math.isinf(h) else f"H = {h:g}", (tok[-1], means[-1]),
                    xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("median explanation length (o200k tokens), log scale")
    ax.set_ylabel("mean S over latents")
    ax.set_title("Where the optimum length sits depends on the half-life H", color=INK, fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_selection(numbers, path):
    h4 = numbers["H4"]
    rules = ["argmax_j", "argmax_s", "argmin_l"]
    names = {"argmax_j": "pick max J", "argmax_s": "pick max S", "argmin_l": "pick shortest"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True)
    _dot_ci(axes[0], [names[r] for r in rules], [h4["j"][r] for r in rules], BLUE, "held-out J of the picked explanation (95% CI)")
    axes[1].barh(np.arange(3), [h4["median_tokens"][r] for r in rules], color=AQUA, height=0.5)
    axes[1].set_xlabel("median length of the picked explanation (tokens)")
    _style(axes[1])
    fig.suptitle("H4: choosing among 8 candidate explanations per latent (cross-fit)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_j_vs_length(df, path):
    groups = {"Neuronpedia LLM explanations": ["gpt35", "gpt4omini"],
              "Llama-70B, length budgets": [*BUDGET_CONDITIONS, "gen_unconstrained"],
              "top-token baseline": ["top_token"], "random control": ["random"]}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharex=True, sharey=True)
    grid_l = np.linspace(1, 120, 200)
    for ax, (title, conds) in zip(axes, groups.items()):
        g = df[df.condition.isin(conds)]
        for level in (0.2, 0.4, 0.6):  # iso-S curves: J = S * 2^(L/H)
            j_iso = level * 2 ** (grid_l / config.HALF_LIFE_TOKENS)
            ax.plot(grid_l, j_iso, color=MUTED, lw=0.7, ls="--")
            ax.annotate(f"S={level}", (grid_l[60], j_iso[60]), fontsize=6, color=MUTED)
        ax.scatter(g.n_tokens, g.j, s=10, color=BLUE, alpha=0.5, lw=0)
        ax.set_xscale("log")
        ax.set_ylim(-0.5, 1.05)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("tokens")
        _style(ax)
    axes[0].set_ylabel("J")
    fig.suptitle("Every scored explanation; dashed lines = equal S (same score under the length factor)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_figures(df: pd.DataFrame, numbers: dict, fig_dir) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_conditions(df, fig_dir / "fig1_conditions.png")
    fig_j_vs_length(df, fig_dir / "fig2_j_vs_length.png")
    if set(BUDGET_CONDITIONS) <= set(df.condition):
        fig_budget_curves(df, fig_dir / "fig3_budget_curves.png")
        fig_h_sweep(df, fig_dir / "fig4_h_sweep.png")
        fig_selection(numbers, fig_dir / "fig5_selection.png")
