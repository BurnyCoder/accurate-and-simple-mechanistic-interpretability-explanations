"""Names, roles, display labels and order of the explanation sources ("conditions").

Global context: every scored explanation has a condition (PREREG section 5). The analysis groups
by it, the figures and tables label it, and H4 picks among some of them. Keeping the names here
means `analysis`, `hypotheses`, `figures` and `render` cannot drift apart.
"""

from . import config

BUDGET_CONDITIONS = [f"gen_b{b}" for b in config.BUDGETS]  # Exp 2, asked for ~B words
EXP1_CONDITIONS = ["gpt35", "gpt4omini", "top_token", "random"]  # existing, baseline, control
CANDIDATES = [*BUDGET_CONDITIONS, "gen_unconstrained", "gpt35", "gpt4omini"]  # H4 candidate set
ORDER = ["random", "top_token", "gpt35", "gpt4omini", *BUDGET_CONDITIONS, "gen_unconstrained"]

# Table labels (README, reports, paper).
LABELS = {"gpt35": "gpt-3.5-turbo (Neuronpedia)", "gpt4omini": "gpt-4o-mini (Neuronpedia)",
          "top_token": "top-token baseline", "random": "random-explanation control",
          "gen_unconstrained": "Llama-3.1-70B, delphi default prompt",
          **{f"gen_b{b}": f"Llama-3.1-70B, ~{b} words" for b in config.BUDGETS}}
# Shorter labels for figure axes, where the table labels would not fit.
SHORT_LABELS = LABELS | {"random": "random control", "gen_unconstrained": "Llama-70B unconstrained",
                         **{f"gen_b{b}": f"Llama-70B ~{b} words" for b in config.BUDGETS}}
ROLE = {"gpt35": "existing", "gpt4omini": "existing", "top_token": "baseline", "random": "control",
        **{c: "Exp 2" for c in [*BUDGET_CONDITIONS, "gen_unconstrained"]}}
