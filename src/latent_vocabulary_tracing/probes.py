"""Deterministic, testable helpers for building probe corpora."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .spans import validate_role_spans


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


def validate_probe_record(probe: Mapping[str, Any], *, n_tokens: int) -> None:
    """Validate stored lengths and the role geometry used by GPU analyses."""

    key = probe.get("key", "<unknown>")
    if probe.get("n_tok") != n_tokens:
        raise ValueError(
            f"{key}: stored n_tok={probe.get('n_tok')!r}, tokenizer produced {n_tokens}"
        )
    prompt_len = probe.get("prompt_len")
    if not isinstance(prompt_len, int) or prompt_len < 0:
        raise ValueError(f"{key}: prompt_len must be a nonnegative integer")
    prompt_end = min(prompt_len, n_tokens)
    spans = probe.get("role_spans")
    if not isinstance(spans, dict):
        raise ValueError(f"{key}: role_spans must be an object")
    validate_role_spans(spans, n_tokens=n_tokens)

    expected_context = [[0, prompt_end]] if prompt_end else None
    expected_response = [[prompt_end, n_tokens]] if prompt_end < n_tokens else None
    for role, expected in (
        ("input_context", expected_context),
        ("model_response", expected_response),
    ):
        if spans.get(role) != expected:
            raise ValueError(f"{key}: {role}={spans.get(role)!r}, need {expected!r}")

    aliases = {
        "math": ("math_problem", "math_solution"),
        "code": ("code_specification", "code_solution"),
        "neutral": ("neutral_context", "neutral_text"),
    }
    kind = probe.get("kind")
    if kind in aliases:
        context_role, response_role = aliases[kind]
        if spans.get(context_role) != expected_context:
            raise ValueError(f"{key}: {context_role} does not match input_context")
        if spans.get(response_role) != expected_response:
            raise ValueError(f"{key}: {response_role} does not match model_response")

    if kind != "agent" or expected_response is None:
        return
    typed_roles = ("assistant_deliberation", "tool_call", "completion_signal")
    typed_spans = [
        span
        for role in typed_roles
        for span in spans.get(role, [])
    ]
    if not typed_spans:
        raise ValueError(f"{key}: agent response has no typed role spans")
    typed_spans.sort()
    if typed_spans[0][0] != prompt_end or typed_spans[-1][1] != n_tokens:
        raise ValueError(f"{key}: typed agent roles do not cover the response")
    if any(
        left[1] != right[0]
        for left, right in zip(typed_spans, typed_spans[1:], strict=False)
    ):
        raise ValueError(f"{key}: typed agent response roles overlap or leave a gap")

    for observation in spans.get("tool_observation", []):
        if observation[1] > prompt_end:
            raise ValueError(f"{key}: tool_observation extends into the response")
