import pytest

from latent_vocabulary_tracing.spans import infer_role_spans, validate_role_spans


def character_offsets(text: str) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(len(text))]


def test_agent_protocol_is_split_into_observation_deliberation_and_action():
    prefix = "<|im_start|>user\nterminal output<|im_end|>\n<|im_start|>assistant\n"
    response = (
        '{"analysis": "inspect", "plan": "run", "commands": [{"keystrokes": "ls"}], '
        '"task_complete": false}'
    )
    text = prefix + response
    spans = infer_role_spans(
        text,
        kind="agent",
        prompt_len=len(prefix),
        offsets=character_offsets(text),
    )
    assert text[slice(*spans["tool_observation"][0])] == "terminal output"
    assert '"analysis"' in text[slice(*spans["assistant_deliberation"][0])]
    assert '"commands"' in text[slice(*spans["tool_call"][0])]
    assert '"task_complete"' in text[slice(*spans["completion_signal"][0])]
    validate_role_spans(spans, n_tokens=len(text))


def test_domain_aliases_and_truncation_are_bounded():
    spans = infer_role_spans(
        "abcdefghij",
        kind="math",
        prompt_len=4,
        offsets=character_offsets("abcdefghij")[:7],
    )
    assert spans["math_problem"] == [(0, 4)]
    assert spans["math_solution"] == [(4, 7)]
    validate_role_spans(spans, n_tokens=7)


def test_span_validator_rejects_out_of_bounds_annotations():
    with pytest.raises(ValueError, match="outside"):
        validate_role_spans({"bad": [[2, 5]]}, n_tokens=4)


def test_probe_truncated_before_response_is_all_context():
    spans = infer_role_spans(
        "abcdef",
        kind="code",
        prompt_len=20,
        offsets=character_offsets("abcdef"),
    )
    assert spans["input_context"] == [(0, 6)]
    assert spans["code_specification"] == [(0, 6)]
    assert "model_response" not in spans


def test_chat_matched_neutral_separates_context_from_continuation():
    text = "abcdefgh"
    spans = infer_role_spans(
        text,
        kind="neutral",
        prompt_len=3,
        offsets=character_offsets(text),
    )
    assert spans == {
        "input_context": [(0, 3)],
        "model_response": [(3, 8)],
        "neutral_context": [(0, 3)],
        "neutral_text": [(3, 8)],
    }
