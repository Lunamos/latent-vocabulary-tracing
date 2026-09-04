"""Deterministic, testable helpers for building probe corpora."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _article_title(text: str) -> str | None:
    stripped = text.strip()
    if (
        stripped.startswith("= ")
        and stripped.endswith(" =")
        and not stripped.startswith("= =")
    ):
        return stripped[2:-2].strip()
    return None


def select_spaced_text_records(
    records: Iterable[Mapping[str, Any]],
    *,
    count: int,
    start: int,
    stride: int,
    min_chars: int = 600,
) -> list[dict[str, Any]]:
    """Select an arithmetic progression among sufficiently long text records.

    Start and the returned qualifying index count only records whose stripped
    text has at least min_chars characters. Top-level WikiText headings are
    tracked so callers can audit document diversity.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if start < 0:
        raise ValueError("start cannot be negative")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if min_chars <= 0:
        raise ValueError("min_chars must be positive")

    targets = [start + offset * stride for offset in range(count)]
    target_to_output = {target: offset for offset, target in enumerate(targets)}
    selected: list[dict[str, Any] | None] = [None] * count
    article: str | None = None
    qualifying_index = -1
    for record in records:
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("every source record must contain string field 'text'")
        heading = _article_title(text)
        if heading is not None:
            article = heading
            continue
        if len(text.strip()) < min_chars:
            continue
        qualifying_index += 1
        output_index = target_to_output.get(qualifying_index)
        if output_index is not None:
            selected[output_index] = {
                "text": text,
                "qualifying_index": qualifying_index,
                "article": article,
            }
        if qualifying_index >= targets[-1]:
            break

    missing = [target for target, row in zip(targets, selected, strict=True) if row is None]
    if missing:
        raise ValueError(f"source ended before qualifying record(s) {missing}")
    return [row for row in selected if row is not None]


def describe_neutral_protocol(probes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize frozen neutral-control metadata for result provenance."""

    metadata = [
        probe.get("meta") or {}
        for probe in probes
        if probe.get("kind") == "neutral"
    ]

    def unique(field: str, *, absent: Any = None) -> Any:
        values = {row.get(field) for row in metadata}
        return next(iter(values)) if len(values) == 1 else absent

    return {
        "neutral_control": unique("control", absent="mixed_or_absent"),
        "neutral_source_context_tokens": unique("source_context_tokens"),
        "neutral_source_dataset": unique("source_dataset"),
        "neutral_source_split": unique("source_split"),
        "neutral_source_sampling": unique("source_sampling"),
        "neutral_source_start": unique("source_start"),
        "neutral_source_stride": unique("source_stride"),
        "neutral_source_document_count": len(
            {row.get("article") for row in metadata if row.get("article") is not None}
        ),
    }
