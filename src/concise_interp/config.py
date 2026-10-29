"""Every pre-registered constant in one place (see PREREG.md).

Global context: the phases in `run.py` import these names, so changing a value here
changes the whole experiment. Values that may differ per machine (API key, concurrency,
model overrides) come from `.env` via python-dotenv.
"""

import math
import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root = two levels above src/concise_interp/.
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --- Data: Neuronpedia public export of Joseph Bloom's GPT-2 small layer-6 residual SAE.
# Bucket layout checked by listing https://neuronpedia-datasets.s3.us-east-1.amazonaws.com
NEURONPEDIA_BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
MODEL_ID = "gpt2-small"
SOURCE_ID = "6-res-jb"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
FIGURES_DIR = ROOT / "figures"

# delphi's detection few-shot prompt quotes these 6-res-jb latents, so scoring them would leak.
FEWSHOT_LATENTS = frozenset({6048, 9396})

# --- Example split (PREREG section 3).
N_TRAIN = 20  # top activating records, shown only to the Exp 2 explainer
N_TEST_POS = 25  # activating test records, stratified over activation quantiles
N_TEST_NEG = 25  # records whose max activation is exactly 0
N_QUANTILE_BINS = 5
MIN_ACTIVATING = 45  # eligibility: enough activating records for train + test
MIN_ZERO = N_TEST_NEG

# --- Samples of latents.
N_MAIN, SEED_MAIN = 200, 0
N_PILOT, SEED_PILOT = 10, 1
N_RELIABILITY, SEED_RELIABILITY = 50, 2

# --- Scorer and explainer (PREREG section 4).
SCORER_MODEL = os.getenv("SCORER_MODEL", "meta-llama/llama-3.1-70b-instruct")
EXPLAINER_MODEL = os.getenv("EXPLAINER_MODEL", "meta-llama/llama-3.1-70b-instruct")
ALT_SCORER_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Pin the main model to one provider (DeepInfra serves Llama-3.1-70B in fp8) so all calls hit
# the same weights/quantisation; no fallbacks, failed calls are retried instead.
MAIN_PROVIDER = {"order": ["DeepInfra"], "allow_fallbacks": False}
N_EXAMPLES_SHOWN = 5  # Paulo et al. show the scorer 5 shuffled examples per prompt
SHUFFLE_SEED = 42  # delphi's Classifier default seed
RETEST_SHUFFLE_SEED = 43
MAX_RETRIES = 8
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))

# --- Experiment 2: word budgets for generated explanations; None = delphi's default prompt.
BUDGETS = (2, 5, 10, 20, 40)

# --- The length factor (PREREG section 2).
HALF_LIFE_TOKENS = float(os.getenv("HALF_LIFE_TOKENS", "20"))
H_SWEEP = (5.0, 10.0, 20.0, 40.0, 80.0, math.inf)

# --- Analysis.
N_BOOTSTRAP = 10_000
SEED_ANALYSIS = 0
