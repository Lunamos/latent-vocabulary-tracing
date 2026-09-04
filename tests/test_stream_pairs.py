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
