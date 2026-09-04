"""Frozen, machine-checkable checkpoint-lineage registries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .provenance import stable_json_hash


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """One declared parent-to-descendant comparison and its public evidence."""

    tag: str
    parent: str
    descendant: str
    family: str
    target: str
    recipe: str
    analysis_role: str
    lineage_relation: str
    lineage_evidence: str
    evidence_url: str
    license: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EdgeRegistry:
    """A content-addressed collection of unique experimental edges."""

    schema_version: int
    digest: str
    edges: tuple[EdgeRecord, ...]

    @property
    def by_tag(self) -> dict[str, EdgeRecord]:
        return {edge.tag: edge for edge in self.edges}

    def require(self, tag: str, parent: str, descendant: str) -> EdgeRecord:
        """Return an edge only when the tag and both exact checkpoint IDs agree."""

        edge = self.by_tag.get(tag)
        if edge is None:
            raise ValueError(f"edge registry has no entry for tag {tag!r}")
        observed = (parent, descendant)
        expected = (edge.parent, edge.descendant)
        if observed != expected:
            raise ValueError(
                f"edge registry identity mismatch for {tag!r}: "
                f"observed {observed!r}, expected {expected!r}"
            )
        return edge


def load_edge_registry(path: str | Path) -> EdgeRegistry:
    """Load and validate a registry, retaining a canonical content fingerprint."""

    source = Path(path)
    payload: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("edge registry root must be an object")
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported edge registry schema {schema_version!r}")
    rows = payload.get("edges")
    if not isinstance(rows, list) or not rows:
        raise ValueError("edge registry must contain a non-empty edges list")

    fields = set(EdgeRecord.__dataclass_fields__)
    edges: list[EdgeRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"edge registry row {index} is not an object")
        missing = fields.difference(row)
        extra = set(row).difference(fields)
        if missing or extra:
            raise ValueError(
                f"edge registry row {index} has missing={sorted(missing)!r}, "
                f"extra={sorted(extra)!r}"
            )
        if any(not isinstance(row[field], str) or not row[field] for field in fields):
            raise ValueError(f"edge registry row {index} has an empty or non-string field")
        edges.append(EdgeRecord(**row))

    tags = [edge.tag for edge in edges]
    if len(tags) != len(set(tags)):
        raise ValueError("edge registry contains duplicate tags")
    return EdgeRegistry(
        schema_version=schema_version,
        digest=stable_json_hash(payload),
        edges=tuple(edges),
    )
