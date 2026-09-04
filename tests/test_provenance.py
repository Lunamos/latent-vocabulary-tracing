from latent_vocabulary_tracing.provenance import (
    canonicalize_model_config,
    model_config_hash,
    stable_json_hash,
)


def test_config_hash_ignores_loader_path_commit_and_library_version():
    remote = {
        "hidden_size": 4096,
        "rope_theta": 1_000_000.0,
        "_name_or_path": "org/model",
        "_commit_hash": "abc",
        "transformers_version": "5.0",
    }
    local = {
        "hidden_size": 4096.0,
        "rope_theta": 1_000_000,
        "_name_or_path": "/cache/snapshots/abc",
        "_commit_hash": None,
        "transformers_version": "5.1",
    }
    assert model_config_hash(remote) == model_config_hash(local)


def test_config_canonicalization_is_recursive_and_does_not_mutate_input():
    source = {"nested": {"size": 4.0}, "layers": [1.0, 2.5]}
    canonical = canonicalize_model_config(source)
    assert canonical == {"layers": [1, 2.5], "nested": {"size": 4}}
    assert source["nested"]["size"] == 4.0


def test_stable_json_hash_ignores_mapping_order():
    assert stable_json_hash({"b": 2, "a": 1}) == stable_json_hash({"a": 1, "b": 2})
