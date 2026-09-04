from pathlib import Path

import pytest

from latent_vocabulary_tracing.manifest import atomic_claim, load_manifest, parse_manifest_line


def test_parse_manifest_line():
    job = parse_manifest_line("trial-1 | org/base | org/child | --layers 4,8 # note")
    assert job is not None
    assert job.tag == "trial-1"
    assert job.extra_args == ("--layers", "4,8")
    assert parse_manifest_line("# only a comment") is None


def test_manifest_rejects_duplicate_tags(tmp_path: Path):
    path = tmp_path / "pairs.txt"
    path.write_text("same | a | b\nsame | a | c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate tag"):
        load_manifest(path)


def test_atomic_claim_is_exclusive_and_released(tmp_path: Path):
    with atomic_claim(tmp_path, "job") as first:
        assert first is not None
        with atomic_claim(tmp_path, "job") as second:
            assert second is None
    with atomic_claim(tmp_path, "job") as third:
        assert third is not None


@pytest.mark.parametrize(
    "manifest",
    ("jobs_confirmatory_qwen8.txt", "jobs_confirmatory_external.txt"),
)
def test_confirmatory_jobs_cannot_silently_use_legacy_measurements(manifest):
    path = Path(__file__).parents[1] / "zoo" / "data" / manifest
    jobs = load_manifest(path)
    assert jobs
    for job in jobs:
        arguments = set(job.extra_args)
        assert {
            "--decoder",
            "parent",
            "--category_stats",
            "--store_fp32",
            "--full_LL",
        } <= arguments
        assert "--no_store" not in arguments
        if "--no_J" in arguments:
            assert "--layers" in arguments
