import pytest

from latent_vocabulary_tracing.taxonomy import categorize_functional_token, categorize_token


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
