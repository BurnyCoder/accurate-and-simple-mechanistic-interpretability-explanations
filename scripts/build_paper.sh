#!/usr/bin/env bash
# Build paper/paper.pdf from the rendered paper/paper.md (run `run.py --phase analyze --sample main`
# first). Needs pandoc (https://pandoc.org) and a LaTeX engine with xelatex (TeX Live or MiKTeX).
# The figures are referenced as ../figures/*.png, so pandoc resolves them from paper/.
set -euo pipefail
cd "$(dirname "$0")/../paper"
pandoc paper.md --pdf-engine=xelatex -o paper.pdf
echo "wrote paper/paper.pdf"
