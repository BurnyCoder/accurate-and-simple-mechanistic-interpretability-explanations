#!/usr/bin/env bash
# Publish the scored data, processed inputs and LLM logs to a Hugging Face dataset repo and
# add it to the project's collection. Requires `hf auth login` (or HF_TOKEN). Run from the
# repo root after `uv run python run.py --phase analyze --sample main` (which renders hf/README.md).
set -euo pipefail

REPO="BurnyCoder/length-penalized-autointerp-gpt2-small"
COLLECTION_TITLE="Accurate and simple interpretability explanations"  # HF limit: 60 chars

hf repos create "$REPO" --type dataset --exist-ok
hf upload "$REPO" hf/README.md README.md --type dataset --commit-message "Dataset card"
hf upload "$REPO" results results --type dataset --commit-message "Scored explanations and analysis numbers"
hf upload "$REPO" data/processed data/processed --type dataset --commit-message "Processed Neuronpedia rows and verified negatives"
# Full prompt/response logs of the main, reliability and pilot runs (gitignored in the repo).
hf upload "$REPO" logs logs --type dataset --include "*.log" --commit-message "LLM prompt/response logs"

# Collection (created once; --exists-ok makes reruns safe).
SLUG=$(hf collections create "$COLLECTION_TITLE" --description "Length-penalized detection scoring of SAE feature explanations" --exists-ok -q)
hf collections add-item "$SLUG" "$REPO" dataset --exists-ok
echo "published https://huggingface.co/datasets/$REPO (collection: $SLUG)"
