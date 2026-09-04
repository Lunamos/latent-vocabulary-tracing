"""Early compatibility gates for parent-anchored checkpoint comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .provenance import stable_json_hash

ARCHITECTURE_FIELDS = (
    "architecture",
    "hidden_size",
    "n_layers",
    "vocab_size",
    "norm_type",
    "rope",
)


def validate_architecture_pair(
    parent: Mapping[str, Any], descendant: Mapping[str, Any]
) -> None:
    """Reject structural drift before running any paired probes."""

    mismatches = {
        field: (parent.get(field), descendant.get(field))
        for field in ARCHITECTURE_FIELDS
        if parent.get(field) != descendant.get(field)
    }
    if mismatches:
        raise ValueError(f"incompatible parent/descendant architecture: {mismatches!r}")


def validate_tokenizer_vocabularies(
    parent: Mapping[str, int], descendant: Mapping[str, int]
) -> tuple[str, str]:
    """Require an identical full piece-to-ID mapping and return its fingerprints."""

    parent_hash = stable_json_hash(parent)
    descendant_hash = stable_json_hash(descendant)
    if parent_hash != descendant_hash:
        only_parent = sorted(set(parent).difference(descendant))[:5]
        only_descendant = sorted(set(descendant).difference(parent))[:5]
        changed_ids = [
            piece
            for piece in set(parent).intersection(descendant)
            if parent[piece] != descendant[piece]
        ][:5]
        raise ValueError(
            "incompatible full tokenizer vocabularies: "
            f"only_parent={only_parent!r}, only_descendant={only_descendant!r}, "
            f"changed_ids={changed_ids!r}"
        )
    return parent_hash, descendant_hash
