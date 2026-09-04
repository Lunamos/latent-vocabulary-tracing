import pytest

from latent_vocabulary_tracing.probes import (
    describe_neutral_protocol,
    select_spaced_text_records,
    validate_probe_record,
)


def test_spaced_selection_tracks_distinct_article_titles():
    records = [
        {"text": "= Article A ="},
        {"text": "a" * 8},
        {"text": "b" * 8},
        {"text": "= Article B ="},
        {"text": "c" * 8},
        {"text": "d" * 8},
        {"text": "= Article C ="},
        {"text": "e" * 8},
    ]
    selected = select_spaced_text_records(
        records,
        count=3,
        start=0,
        stride=2,
        min_chars=8,
    )
    assert [row["qualifying_index"] for row in selected] == [0, 2, 4]
    assert [row["article"] for row in selected] == [
        "Article A",
        "Article B",
        "Article C",
    ]


def test_spaced_selection_validates_protocol_and_source_length():
    with pytest.raises(ValueError, match="count"):
        select_spaced_text_records([], count=0, start=0, stride=1)
    with pytest.raises(ValueError, match="source ended"):
        select_spaced_text_records(
            [{"text": "long enough"}],
            count=2,
            start=0,
            stride=1,
            min_chars=1,
        )


def test_neutral_protocol_reports_document_diversity_and_mixed_fields():
    probes = [
        {
            "kind": "neutral",
            "meta": {
                "control": "chat_matched_continuation",
                "source_context_tokens": 48,
                "article": article,
            },
        }
        for article in ("A", "B")
    ]
    protocol = describe_neutral_protocol(probes)
    assert protocol["neutral_control"] == "chat_matched_continuation"
    assert protocol["neutral_source_context_tokens"] == 48
    assert protocol["neutral_source_document_count"] == 2

    probes[1]["meta"]["source_context_tokens"] = 64
    assert describe_neutral_protocol(probes)["neutral_source_context_tokens"] is None


def test_probe_record_requires_exact_aliases_and_agent_partition():
    neutral = {
        "key": "n",
        "kind": "neutral",
        "prompt_len": 2,
        "n_tok": 5,
        "role_spans": {
            "input_context": [[0, 2]],
            "model_response": [[2, 5]],
            "neutral_context": [[0, 2]],
            "neutral_text": [[2, 5]],
        },
    }
    validate_probe_record(neutral, n_tokens=5)
    neutral["role_spans"]["neutral_text"] = [[1, 5]]
    with pytest.raises(ValueError, match="neutral_text"):
        validate_probe_record(neutral, n_tokens=5)

    agent = {
        "key": "a",
        "kind": "agent",
        "prompt_len": 2,
        "n_tok": 8,
        "role_spans": {
            "input_context": [[0, 2]],
            "model_response": [[2, 8]],
            "tool_observation": [[0, 2]],
            "assistant_deliberation": [[2, 4]],
            "tool_call": [[4, 7]],
            "completion_signal": [[7, 8]],
        },
    }
    validate_probe_record(agent, n_tokens=8)
    agent["role_spans"]["tool_call"] = [[3, 7]]
    with pytest.raises(ValueError, match="overlap or leave a gap"):
        validate_probe_record(agent, n_tokens=8)
