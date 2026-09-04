import json

from zoo.scripts import stream_pairs
from zoo.scripts.stream_pairs import parse_model_spec


def test_remote_model_spec_separates_reported_identity_from_download_revision():
    spec = parse_model_spec("org/model@revision-name::global_step_20")
    assert spec.repository == "org/model"
    assert spec.revision == "revision-name"
    assert spec.subdirectory == "global_step_20"
    assert spec.identity == "org/model::global_step_20"


def test_local_model_spec_preserves_local_identity(tmp_path):
    spec = parse_model_spec(str(tmp_path))
    assert spec.repository is None
    assert spec.identity == str(tmp_path)


def test_config_repair_uses_overlay_without_mutating_cache(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    blob = tmp_path / "blob-config"
    snapshot.mkdir()
    blob.write_text(json.dumps({"hidden_size": 8.0, "model_type": "fake"}))
    (snapshot / "config.json").symlink_to(blob)
    (snapshot / "weights.safetensors").write_text("weights")
    monkeypatch.setattr(stream_pairs, "RUNTIME_MODELS", tmp_path / "runtime")
    monkeypatch.setattr(stream_pairs, "log", lambda _: None)

    view, overlay = stream_pairs.sanitized_config_view(snapshot)

    assert overlay == view
    assert view != snapshot
    assert json.loads((view / "config.json").read_text())["hidden_size"] == 8
    assert json.loads(blob.read_text())["hidden_size"] == 8.0
    assert (view / "weights.safetensors").is_symlink()


def test_valid_config_uses_original_snapshot(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 8}))
    view, overlay = stream_pairs.sanitized_config_view(tmp_path)
    assert view == tmp_path
    assert overlay is None


def test_subdirectory_view_does_not_copy_config_into_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    checkpoint = snapshot / "step_20"
    checkpoint.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"hidden_size": 8}))
    (checkpoint / "weights.safetensors").write_text("weights")
    monkeypatch.setattr(stream_pairs, "RUNTIME_MODELS", tmp_path / "runtime")
    monkeypatch.setattr(stream_pairs, "log", lambda _: None)

    view, overlay = stream_pairs.subdirectory_model_view(snapshot, "step_20")

    assert overlay == view
    assert not (checkpoint / "config.json").exists()
    assert json.loads((view / "config.json").read_text())["hidden_size"] == 8
    assert (view / "weights.safetensors").is_symlink()
