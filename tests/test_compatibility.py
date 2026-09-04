import pytest

from latent_vocabulary_tracing.compatibility import (
    validate_architecture_pair,
    validate_tokenizer_vocabularies,
)


def architecture(**updates):
    signature = {
        "architecture": "Model",
        "hidden_size": 8,
        "n_layers": 4,
        "vocab_size": 16,
        "norm_type": "RMSNorm",
        "rope": {"rope_type": "default"},
    }
    signature.update(updates)
    return signature


def test_architecture_gate_accepts_identical_signatures():
    validate_architecture_pair(architecture(), architecture())


def test_architecture_gate_reports_every_structural_mismatch():
    with pytest.raises(ValueError, match="n_layers") as error:
        validate_architecture_pair(
            architecture(),
            architecture(n_layers=5, rope={"rope_type": "modified"}),
        )
    assert "rope" in str(error.value)


def test_tokenizer_gate_requires_the_full_piece_to_id_mapping():
    parent = {"a": 0, "b": 1}
    hashes = validate_tokenizer_vocabularies(parent, dict(parent))
    assert hashes[0] == hashes[1]

    with pytest.raises(ValueError, match="changed_ids"):
        validate_tokenizer_vocabularies(parent, {"a": 1, "b": 0})
