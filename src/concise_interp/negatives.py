"""Verified random negatives: OpenWebText windows on which a latent is exactly zero.

Global context: detection scoring needs contexts where the latent does NOT fire. Paulo et al.
(2024, https://arxiv.org/abs/2410.13928, App. D.4) sample them at random from their activation
corpus (App. D.5 discusses the limits of random negatives), and delphi's default
`non_activating_source` is "random". Neuronpedia's export also has zero-activation rows, but the pilot showed they
are overlapping sliding windows over a few documents shared by many latents (PREREG
deviation 2). So here we run GPT-2 small and the exact same SAE (sae_lens release
`gpt2-small-res-jb`) on random OpenWebText windows and keep, per latent, windows whose
activation is 0 at every displayed token.
"""

import gzip
import json
import logging
import random

import numpy as np
import torch

from . import config

log = logging.getLogger(__name__)


def to_neuronpedia_tokens(tokenizer, ids) -> list[str]:
    """GPT-2 byte-level tokens in Neuronpedia's display format: raw BPE symbols ('Ċ', 'âĢĻ')
    with the space marker 'Ġ' shown as ' ' (checked by `check_token_format`)."""
    return [t.replace("Ġ", " ") for t in tokenizer.convert_ids_to_tokens(ids)]


def from_neuronpedia_tokens(tokenizer, tokens: list[str]) -> list[int]:
    """Inverse of `to_neuronpedia_tokens`; returns unk ids if the format guess is wrong."""
    return tokenizer.convert_tokens_to_ids([t.replace(" ", "Ġ") for t in tokens])


def load_model_and_sae():
    from sae_lens import SAE  # heavy imports only in the phase that runs the model
    from transformer_lens import HookedTransformer

    # sae_lens pretrained SAE; its metadata names the TransformerLens model, hook and the
    # model kwargs it was trained with (https://jbloomaus.github.io/SAELens/).
    sae = SAE.from_pretrained(config.SAE_RELEASE, config.SAE_ID, device=config.DEVICE)
    meta = sae.cfg.metadata
    model = HookedTransformer.from_pretrained(meta.model_name, device=config.DEVICE,
                                              **(meta.model_from_pretrained_kwargs or {}))
    return model, sae


@torch.no_grad()
def latent_acts(model, sae, tokens: torch.Tensor, latents: list[int]) -> torch.Tensor:
    """SAE activations [batch, pos, len(latents)] at the SAE's hook."""
    hook = sae.cfg.metadata.hook_name  # "blocks.6.hook_resid_pre"
    # Run only blocks 0..6: resid_pre of block 6 is computed on entry to block 6.
    _, cache = model.run_with_cache(tokens, names_filter=hook, stop_at_layer=int(hook.split(".")[1]) + 1)
    return sae.encode(cache[hook])[..., latents]


def check_token_format(model, latents_data: dict) -> float:
    """Fraction of Neuronpedia tokens that map back to a real GPT-2 id under our format guess."""
    # GPT-2's unk token IS <|endoftext|>, so an unk id only counts as a miss for other strings.
    tok, unk, total, bad = model.tokenizer, model.tokenizer.unk_token_id, 0, 0
    for d in latents_data.values():
        for row in d["rows"][:5]:
            ids = from_neuronpedia_tokens(tok, row["tokens"])
            bad += sum(i is None or (i == unk and t != tok.unk_token) for i, t in zip(ids, row["tokens"]))
            total += len(ids)
    return 1 - bad / total


@torch.no_grad()
def check_reproduces_neuronpedia(model, sae, latents_data: dict, n_rows: int = 5) -> dict:
    """Re-run each latent's top Neuronpedia windows (BOS + 17 tokens; the export lacks the
    longer original context) and compare our max activation with the exported maxValue."""
    ours, theirs, argmax_match = [], [], []
    for latent, d in latents_data.items():
        for row in d["rows"][:n_rows]:
            if row["maxValue"] <= 0:
                continue
            ids = [model.tokenizer.bos_token_id] + from_neuronpedia_tokens(model.tokenizer, row["tokens"])
            acts = latent_acts(model, sae, torch.tensor([ids], device=config.DEVICE), [latent])[0, 1:, 0]
            ours.append(acts.max().item())
            theirs.append(row["maxValue"])
            argmax_match.append(int(acts.argmax().item()) == int(np.argmax(row["values"])))
    return {"n": len(ours), "pearson_max": float(np.corrcoef(ours, theirs)[0, 1]),
            "frac_argmax_same_token": float(np.mean(argmax_match)),
            "frac_ours_positive": float(np.mean(np.array(ours) > 0))}


@torch.no_grad()
def build_negatives(latents_data: dict) -> dict:
    """Phase `negatives`: random OpenWebText windows, SAE activations, 25 zero windows per latent."""
    from datasets import load_dataset

    model, sae = load_model_and_sae()
    report = {"token_format_roundtrip": check_token_format(model, latents_data),
              "reproduce_neuronpedia": check_reproduces_neuronpedia(model, sae, latents_data)}
    log.info("SAE/format checks: %s", report)
    latents = sorted(latents_data)
    # Same dataset Neuronpedia's dashboards used for this SAE (source.jsonl: Skylion007/openwebtext).
    stream = load_dataset(config.NEG_DATASET, split="train", streaming=True).shuffle(seed=config.SEED_MAIN, buffer_size=10_000)
    rng = random.Random(config.SEED_MAIN)
    windows, maxes, batch = [], [], []

    def flush():
        acts = latent_acts(model, sae, torch.stack(batch).to(config.DEVICE), latents)
        # Only the displayed (last N_WINDOW) positions decide whether the window is negative;
        # the earlier positions are context, as in Neuronpedia's 17-token windows.
        maxes.append(acts[:, -config.N_WINDOW:, :].amax(dim=1).cpu())
        batch.clear()

    for doc in stream:
        ids = model.to_tokens(doc["text"], prepend_bos=True)[0]
        if len(ids) < config.NEG_CONTEXT + 1:
            continue
        start = rng.randrange(0, len(ids) - config.NEG_CONTEXT)  # random window per document
        chunk = ids[start:start + config.NEG_CONTEXT]
        windows.append(to_neuronpedia_tokens(model.tokenizer, chunk[-config.N_WINDOW:].tolist()))
        batch.append(chunk)
        if len(batch) == 32:
            flush()
        if len(windows) == config.NEG_POOL:
            break
    if batch:
        flush()
    zero = torch.cat(maxes) == 0  # [pool, latents]: exactly zero at every displayed token
    per_latent = {lat: torch.nonzero(zero[:, i]).flatten().tolist() for i, lat in enumerate(latents)}
    counts = [len(v) for v in per_latent.values()]
    report |= {"pool": len(windows), "zero_windows_min": min(counts), "zero_windows_median": float(np.median(counts))}
    log.info("negatives pool: %s", report)
    with gzip.open(config.PROCESSED_DIR / "negatives.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"windows": windows, "zero_windows": per_latent, "report": report}, f)
    return report


def _load() -> dict:
    with gzip.open(config.PROCESSED_DIR / "negatives.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def load_negatives() -> tuple[list[list[str]], dict[int, list[int]]]:
    d = _load()
    return d["windows"], {int(k): v for k, v in d["zero_windows"].items()}


def load_report() -> dict:
    """The SAE/format validation and pool statistics recorded when the pool was built."""
    return _load()["report"]
