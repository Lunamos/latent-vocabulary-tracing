import pytest

from latent_vocabulary_tracing.taxonomy import (
    categorize_functional_token,
    categorize_token,
    categorize_trace_token,
    is_displayable_trace_token,
)


@pytest.mark.parametrize(
    ("token", "category"),
    [
        ("<|assistant|>", "special"),
        (" therefore", "discourse"),
        (" the", "function"),
        (" equation", "english"),
        ("\\boxed", "math"),
        (" 42", "number"),
        ("def", "code"),
        ("。", "punct"),
        ("数学", "cjk"),
    ],
)
def test_readable_categories(token, category):
    assert categorize_token(token) == category


def test_non_string_is_rejected():
    with pytest.raises(TypeError):
        categorize_token(3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("token", "category"),
    [
        (" because", "reasoning_process"),
        (" Therefore", "reasoning_process"),
        (" answer", "answer_commitment"),
        ("\\boxed", "answer_commitment"),
        (" equation", "mathematical_content"),
        (" math", "mathematical_content"),
        (" sum", "mathematical_content"),
        (" 42", "symbolic_notation"),
        ("\\frac", "symbolic_notation"),
        ("cancel", "symbolic_notation"),
        ("\n\n", "presentation"),
        ("latex", "presentation"),
        ("<|endoftext|>", "other"),
        (" the", "general_language"),
        ("数学", "other"),
    ],
)
def test_functional_categories(token, category):
    assert categorize_functional_token(token) == category


def test_functional_taxonomy_rejects_non_string():
    with pytest.raises(TypeError):
        categorize_functional_token(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("token", "category"),
    [
        (" because", "reasoning_connectors"),
        (" answer", "answer_markers"),
        (" equation", "mathematical_lexicon"),
        (" function", "ambiguous_formal_lexicon"),
        ("\\frac", "symbolic_notation"),
        (" commands", "tool_interface_lexicon"),
        ("task_complete", "tool_interface_lexicon"),
        (" stderr", "execution_status_lexicon"),
        (" failed", "execution_status_lexicon"),
        ("def", "programming_lexicon"),
        ("\n\n", "formatting"),
        (" the", "general_language"),
        ("数学", "other"),
    ],
)
def test_trace_categories_cover_math_agent_and_code(token, category):
    assert categorize_trace_token(token) == category


def test_trace_taxonomy_rejects_non_string():
    with pytest.raises(TypeError):
        categorize_trace_token(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("token", "displayable"),
    [
        (" because", True),
        ("\\frac", True),
        (" stderr", True),
        (" 42", True),
        ("graph", False),
        ("oret", False),
        ("_derivative", False),
        (".geometry", False),
        ("\x0c", False),
        ("<|endoftext|>", False),
        ("\n\n", False),
    ],
)
def test_displayable_trace_tokens_use_a_frozen_conservative_rule(token, displayable):
    assert is_displayable_trace_token(token) is displayable
