"""Small, safe views over trace-summary JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_summary(path: str | Path) -> dict[str, Any]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"tag", "model_a", "model_b", "layers"}
    missing = required.difference(summary)
    if missing:
        raise ValueError(f"summary is missing required keys: {sorted(missing)}")
    return summary


def summary_view(summary: dict[str, Any]) -> dict[str, Any]:
    """Return metadata suitable for terminals and experiment registries."""

    probes = summary.get("probes")
    if isinstance(probes, list):
        probe_count: int | None = len(probes)
    elif isinstance(probes, int):
        probe_count = probes
    else:
        probe_count = None
    return {
        "tag": summary["tag"],
        "parent": summary["model_a"],
        "descendant": summary["model_b"],
        "layers": summary["layers"],
        "readouts": summary.get("readouts"),
        "probe_count": probe_count,
        "seconds": summary.get("seconds"),
    }
