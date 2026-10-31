"""Download the Neuronpedia export and pick the pre-registered latent samples.

Global context: phase 1 of `run.py`. Neuronpedia publishes, per SAE, gzipped JSONL batches of
`explanations/`, `activations/` and `features/` in a public S3 bucket
(https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html). We download the
explanation and activation batches once to data/raw/, then write a compact per-latent file
for the sampled latents to data/processed/, which the later phases read.
"""

import gzip
import json
import logging
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from . import config

log = logging.getLogger(__name__)
_S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"  # XML namespace of S3 ListObjectsV2
EXPLAINER_LABELS = {"gpt-3.5-turbo": "gpt35", "gpt-4o-mini": "gpt4omini"}


def list_batches(kind: str) -> list[str]:
    """S3 keys of every batch file of `kind`, via the public ListObjectsV2 API (paginated)."""
    prefix = f"v1/{config.MODEL_ID}/{config.SOURCE_ID}/{kind}/"
    keys, token = [], None
    while True:
        query = {"list-type": "2", "prefix": prefix} | ({"continuation-token": token} if token else {})
        with urllib.request.urlopen(f"{config.NEURONPEDIA_BUCKET}/?{urllib.parse.urlencode(query)}") as r:
            root = ET.fromstring(r.read())
        keys += [c.findtext(f"{_S3}Key") for c in root.iter(f"{_S3}Contents")]
        if root.findtext(f"{_S3}IsTruncated") != "true":
            # Top-level batches only: activations/v2/ re-encodes the same rows in a sparse format.
            return sorted(k for k in keys if k.endswith(".jsonl.gz") and "/" not in k[len(prefix):])
        token = root.findtext(f"{_S3}NextContinuationToken")


def download(kind: str) -> list[Path]:
    """Download (once) all batches of `kind` to data/raw/<same key>."""
    paths = []
    for key in list_batches(kind):
        path = config.RAW_DIR / key
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")  # rename at the end = no half files
            urllib.request.urlretrieve(f"{config.NEURONPEDIA_BUCKET}/{key}", tmp)
            tmp.rename(path)
            log.info("downloaded %s (%.1f MB)", key, path.stat().st_size / 1e6)
        paths.append(path)
    return paths


def iter_jsonl_gz(paths: list[Path]) -> Iterator[dict]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from map(json.loads, f)


def load_explanations(paths: list[Path]) -> dict[int, list[dict]]:
    """Latent -> its explanations in export order, without the large embedding field."""
    by_latent: dict[int, list[dict]] = defaultdict(list)
    for row in iter_jsonl_gz(paths):
        model = row.get("explanationModelName")
        source = EXPLAINER_LABELS.get(model, "human" if row["typeName"] == "human" else model)
        by_latent[int(row["index"])].append(
            {"description": row["description"], "source": source, "type": row["typeName"]}
        )
    return by_latent


def count_examples(paths: list[Path]) -> dict[int, tuple[int, int]]:
    """Latent -> (#activating rows, #rows with maxValue == 0); one streaming pass."""
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for row in iter_jsonl_gz(paths):
        counts[int(row["index"])][0 if row["maxValue"] > 0 else 1] += 1
    return {k: (a, z) for k, (a, z) in counts.items()}


def filter_counts(explanations: dict[int, list[dict]], counts: dict[int, tuple[int, int]]) -> dict[str, int]:
    """How many latents pass each eligibility filter on its own (reported next to the joint pass)."""
    both = sum({"gpt35", "gpt4omini"} <= {e["source"] for e in explanations.get(k, [])} for k in counts)
    return {"both_llm_explanations": both,
            "min_activating": sum(a >= config.MIN_ACTIVATING for a, _ in counts.values()),
            "min_zero_rows": sum(z >= config.MIN_ZERO for _, z in counts.values())}


def eligible_latents(explanations: dict[int, list[dict]], counts: dict[int, tuple[int, int]]) -> list[int]:
    """PREREG eligibility: both LLM explanations, >=45 activating rows, >=25 zero rows,
    and not one of delphi's few-shot latents."""
    out = []
    for latent, (n_act, n_zero) in counts.items():
        sources = {e["source"] for e in explanations.get(latent, [])}
        if (
            {"gpt35", "gpt4omini"} <= sources
            and n_act >= config.MIN_ACTIVATING
            and n_zero >= config.MIN_ZERO
            and latent not in config.FEWSHOT_LATENTS
        ):
            out.append(latent)
    return sorted(out)


def draw_samples(eligible: list[int]) -> dict[str, list[int]]:
    """Main (seed 0), a disjoint pilot (seed 1), and a reliability subset of main (seed 2)."""
    main = sorted(random.Random(config.SEED_MAIN).sample(eligible, config.N_MAIN))
    rest = sorted(set(eligible) - set(main))
    pilot = sorted(random.Random(config.SEED_PILOT).sample(rest, config.N_PILOT))
    reliability = sorted(random.Random(config.SEED_RELIABILITY).sample(main, config.N_RELIABILITY))
    return {"main": main, "pilot": pilot, "reliability": reliability}


def extract_rows(paths: list[Path], latents: set[int]) -> dict[int, list[dict]]:
    """Activation rows (tokens, per-token values, max) of the chosen latents, export order."""
    rows: dict[int, list[dict]] = defaultdict(list)
    for row in iter_jsonl_gz(paths):
        if int(row["index"]) in latents:
            rows[int(row["index"])].append(
                {"tokens": row["tokens"], "values": row["values"], "maxValue": row["maxValue"]}
            )
    return rows


def prepare() -> dict:
    """Phase `download`: fetch the export, check eligibility, draw the samples, save them."""
    expl_paths, act_paths = download("explanations"), download("activations")
    explanations = load_explanations(expl_paths)
    counts = count_examples(act_paths)
    eligible = eligible_latents(explanations, counts)
    samples = draw_samples(eligible)
    log.info("latents with activations: %d, eligible: %d (%.1f%%)",
             len(counts), len(eligible), 100 * len(eligible) / len(counts))
    chosen = set(samples["main"]) | set(samples["pilot"])
    rows = extract_rows(act_paths, chosen)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(config.PROCESSED_DIR / "latents.jsonl.gz", "wt", encoding="utf-8") as f:
        for latent in sorted(chosen):
            f.write(json.dumps({"latent": latent, "explanations": explanations[latent],
                                "rows": rows[latent]}) + "\n")
    summary = {"n_latents": len(counts), "n_eligible": len(eligible),
               "filters": filter_counts(explanations, counts), "samples": samples}
    (config.PROCESSED_DIR / "samples.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_processed() -> tuple[dict[int, dict], dict[str, list[int]]]:
    """What later phases read: latent -> {explanations, rows}, and the sample lists."""
    with gzip.open(config.PROCESSED_DIR / "latents.jsonl.gz", "rt", encoding="utf-8") as f:
        latents = {d["latent"]: d for d in map(json.loads, f)}
    samples = json.loads((config.PROCESSED_DIR / "samples.json").read_text())["samples"]
    return latents, samples
