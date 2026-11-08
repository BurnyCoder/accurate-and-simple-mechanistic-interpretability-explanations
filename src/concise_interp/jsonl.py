"""Append-only JSONL result files, read back to resume a run.

Global context: the explain and score phases append one row per finished (latent, condition) job
and skip finished jobs on the next run (PREREG section 8), and the analysis reads the same files.
A crash can leave a half-written last line; it is skipped with a warning so the job is redone.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    """All rows of `path` ([] if it does not exist), tolerating only a truncated final line."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 1:  # corruption before the end is not a crash artefact: fail loudly
                raise
            log.warning("%s: skipping a truncated last line (crash during a write?)", path)
    return rows


def done_keys(path: Path) -> set[tuple[int, str]]:
    """(latent, condition) pairs already saved in `path`."""
    return {(r["latent"], r["condition"]) for r in read_jsonl(path)}


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row; one write per finished job keeps the file resumable after a crash."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
