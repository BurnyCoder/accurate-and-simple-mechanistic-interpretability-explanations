"""Render README, reports and paper from templates + results/main/numbers.json.

Global context: the user's rule is "every number comes from a script that reads results
files". Templates in templates/ contain placeholders that this module fills:

    {{H3.s_by_budget.gen_b10.mean:.3f}}   dotted path into numbers.json, Python format spec
    {{table:conditions}}                  a generated markdown table (see TABLES)
    {{include:templates_static/x.md}}     another file, rendered the same way (shared text)

`uv run python run.py --phase analyze` calls `render_all()` after the analysis.
"""

import json
import re
from pathlib import Path

from . import config

PLACEHOLDER = re.compile(r"\{\{\s*([^}:]+?)\s*(?::([^}]*))?\}\}")
LABELS = {"gpt35": "gpt-3.5-turbo (Neuronpedia)", "gpt4omini": "gpt-4o-mini (Neuronpedia)",
          "top_token": "top-token baseline", "random": "random-explanation control",
          "gen_unconstrained": "Llama-3.1-70B, delphi default prompt",
          **{f"gen_b{b}": f"Llama-3.1-70B, ~{b} words" for b in config.BUDGETS}}
ROLE = {"gpt35": "existing", "gpt4omini": "existing", "top_token": "baseline", "random": "control",
        **{f"gen_b{b}": "Exp 2" for b in config.BUDGETS}, "gen_unconstrained": "Exp 2"}
ORDER = ["random", "top_token", "gpt35", "gpt4omini", *[f"gen_b{b}" for b in config.BUDGETS], "gen_unconstrained"]


def lookup(numbers: dict, path: str):
    value = numbers
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def table_conditions(numbers: dict) -> str:
    rows = {r["condition"]: r for r in numbers["conditions"]}
    lines = [f"| explanation source | role | n | mean BA | mean J [95% CI] | median tokens | mean S (H={numbers['half_life']:g}) [95% CI] |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for c in [c for c in ORDER if c in rows]:
        r = rows[c]
        lines.append(f"| {LABELS[c]} | {ROLE[c]} | {r['n']} | {r['ba']:.3f} | {r['j_mean']:.3f} [{r['j_lo']:.3f}, {r['j_hi']:.3f}] "
                     f"| {r['tokens_median']:.0f} | {r['s_mean']:.3f} [{r['s_lo']:.3f}, {r['s_hi']:.3f}] |")
    return "\n".join(lines)


def table_h3(numbers: dict) -> str:
    lines = [f"| comparison (S at H={numbers['half_life']:g}) | mean difference [95% CI] | Wilcoxon p (one-sided) | Holm p |",
             "|---|---:|---:|---:|"]
    for name, t in numbers["H3"]["tests"].items():
        a, b = name.split("_vs_")
        lines.append(f"| ~{a[1:]} words − ~{b[1:]} words | {t['mean']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}] | {t['p_greater']:.2g} | {t['p_holm']:.2g} |")
    return "\n".join(lines)


def table_h4(numbers: dict) -> str:
    h4 = numbers["H4"]
    names = {"argmax_j": "pick highest J (plain detection)", "argmax_s": "pick highest S (this method)", "argmin_l": "pick shortest"}
    lines = ["| selection rule | held-out mean J [95% CI] | median tokens |", "|---|---:|---:|"]
    for r, name in names.items():
        j = h4["j"][r]
        lines.append(f"| {name} | {j['mean']:.3f} [{j['lo']:.3f}, {j['hi']:.3f}] | {h4['median_tokens'][r]:.1f} |")
    return "\n".join(lines)


def table_h_sweep(numbers: dict) -> str:
    sweep = numbers["exploratory"]["h_sweep"]
    conds = [c for c in ORDER if c in sweep["20"]["mean_s"]]
    lines = ["| H (tokens) | " + " | ".join(LABELS[c] for c in conds) + " | best budget |",
             "|---|" + "---:|" * len(conds) + "---|"]
    for h, v in sweep.items():
        lines.append(f"| {'∞ (no length factor: max(J, 0))' if h == 'inf' else h} | " + " | ".join(f"{v['mean_s'][c]:.3f}" for c in conds)
                     + f" | ~{v['best_budget'][5:]} words |")
    return "\n".join(lines)


def table_reliability(numbers: dict) -> str:
    lines = ["| re-scoring | latents | explanations | Spearman J | Spearman S |", "|---|---:|---:|---:|---:|"]
    for key, name in (("reliability_retest", "same scorer, new shuffle seed (43)"),
                      ("reliability_altscorer", "Llama-3.3-70B as scorer")):
        if key in numbers:
            r = numbers[key]
            lines.append(f"| {name} | {r['n_latents']} | {r['n_explanations']} | {r['spearman_j']:.3f} | {r['spearman_s']:.3f} |")
    return "\n".join(lines)


TABLES = {"conditions": table_conditions, "h3": table_h3, "h4": table_h4, "h_sweep": table_h_sweep,
          "reliability": table_reliability}


def render(template: str, numbers: dict) -> str:
    def fill(match):
        path, spec = match.group(1), match.group(2)
        if path == "table":
            return TABLES[spec](numbers)
        if path == "include":
            return render((config.ROOT / spec).read_text(encoding="utf-8"), numbers)
        value = lookup(numbers, path)
        return format(value, spec) if spec else str(value)
    return PLACEHOLDER.sub(fill, template)


def render_all(sample: str = "main") -> list[Path]:
    """templates/<path> -> <path> for every template file (README.md, paper/, reports/)."""
    numbers = json.loads((config.RESULTS_DIR / sample / "numbers.json").read_text())
    pilot = config.RESULTS_DIR / "pilot" / "numbers.json"
    if pilot.exists():  # the pilot report cites its own (debugging-only) numbers
        numbers["pilot"] = json.loads(pilot.read_text())
    written = []
    for tpl in sorted((config.ROOT / "templates").rglob("*.md")):
        target = config.ROOT / tpl.relative_to(config.ROOT / "templates")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(tpl.read_text(encoding="utf-8"), numbers), encoding="utf-8")
        written.append(target)
    return written
