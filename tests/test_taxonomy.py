import pytest

from latent_vocabulary_tracing.taxonomy import categorize_token


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
