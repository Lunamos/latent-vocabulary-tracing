"""Sequence-role annotations for cross-domain latent-vocabulary probes."""

from __future__ import annotations

import re
from collections.abc import Sequence

TokenSpan = tuple[int, int]


def _char_to_token_span(
    offsets: Sequence[tuple[int, int]], start: int, end: int
) -> TokenSpan | None:
    positions = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_end > start and token_start < end
    ]
    return (positions[0], positions[-1] + 1) if positions else None


def _first_content_offset(offsets: Sequence[tuple[int, int]], start: int) -> int | None:
    for token_start, token_end in offsets[start:]:
        if token_end > token_start:
            return token_start
    return None


def infer_role_spans(
    text: str,
    *,
    kind: str,
    prompt_len: int,
    offsets: Sequence[tuple[int, int]],
) -> dict[str, list[TokenSpan]]:
    """Infer auditable half-open token spans from a frozen rendered probe.

    Every probe receives domain-neutral ``input_context`` and
    ``model_response`` roles.  Domain-specific aliases make plots readable.
    Agent traces are additionally split into terminal observation, assistant
    deliberation, tool call, and completion signal.  The split uses the
    serialized protocol fields rather than interpreting the model's prose.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    n_tokens = len(offsets)
    if prompt_len < 0:
        raise ValueError("prompt_len cannot be negative")
    # A frozen probe can be truncated before its response begins.  Preserve the
    # stored, pre-truncation prompt length but annotate all retained tokens as
    # context in that case.
    prompt_end = min(prompt_len, n_tokens)

    spans: dict[str, list[TokenSpan]] = {}

    def add(name: str, span: TokenSpan | None) -> None:
        if span is not None and span[1] > span[0]:
            spans.setdefault(name, []).append(span)

    if prompt_end:
        add("input_context", (0, prompt_end))
    if prompt_end < n_tokens:
        add("model_response", (prompt_end, n_tokens))

    aliases = {
        "math": ("math_problem", "math_solution"),
        "code": ("code_specification", "code_solution"),
    }
    if kind in aliases:
        context_name, response_name = aliases[kind]
        if prompt_end:
            add(context_name, (0, prompt_end))
        if prompt_end < n_tokens:
            add(response_name, (prompt_end, n_tokens))
    elif kind == "neutral":
        if prompt_end:
            add("neutral_context", (0, prompt_end))
        if prompt_end < n_tokens:
            add("neutral_text", (prompt_end, n_tokens))

    if kind != "agent":
        return spans

    response_char_start = _first_content_offset(offsets, prompt_end)
    if response_char_start is None:
        response_char_start = len(text)

    user_marker = "<|im_start|>user\n"
    user_start = text.rfind(user_marker, 0, response_char_start)
    if user_start >= 0:
        user_start += len(user_marker)
        user_end = text.find("<|im_end|>", user_start, response_char_start)
        if user_end < 0:
            user_end = response_char_start
        add("tool_observation", _char_to_token_span(offsets, user_start, user_end))

    response = text[response_char_start:]
    command_match = re.search(r'["\']commands["\']\s*:', response)
    completion_match = re.search(r'["\']task_complete["\']\s*:', response)
    command_start = (
        response_char_start + command_match.start() if command_match is not None else None
    )
    completion_start = (
        response_char_start + completion_match.start()
        if completion_match is not None
        else None
    )

    deliberation_end = command_start or completion_start or len(text)
    add(
        "assistant_deliberation",
        _char_to_token_span(offsets, response_char_start, deliberation_end),
    )
    if command_start is not None:
        add(
            "tool_call",
            _char_to_token_span(offsets, command_start, completion_start or len(text)),
        )
    if completion_start is not None:
        add(
            "completion_signal",
            _char_to_token_span(offsets, completion_start, len(text)),
        )
    return spans


def validate_role_spans(
    spans: dict[str, Sequence[Sequence[int]]], *, n_tokens: int
) -> None:
    """Validate the serialized form of role spans."""

    for role, role_spans in spans.items():
        if not role or not isinstance(role_spans, Sequence):
            raise ValueError("role names and span lists must be non-empty")
        for span in role_spans:
            if len(span) != 2:
                raise ValueError(f"role {role!r} contains a malformed span")
            start, end = span
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"role {role!r} span endpoints must be integers")
            if not 0 <= start < end <= n_tokens:
                raise ValueError(
                    f"role {role!r} span {(start, end)} is outside [0, {n_tokens}]"
                )
