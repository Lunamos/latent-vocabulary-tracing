import json
from pathlib import Path

import pytest

from latent_vocabulary_tracing.manifest import load_manifest
from latent_vocabulary_tracing.registry import load_edge_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "zoo" / "data" / "edge_registry.json"
MANIFESTS = (
    ROOT / "zoo" / "data" / "jobs_confirmatory_qwen8.txt",
    ROOT / "zoo" / "data" / "jobs_confirmatory_external.txt",
    ROOT / "zoo" / "data" / "jobs_confirmatory_jlens_qwen8.txt",
    ROOT / "zoo" / "data" / "jobs_confirmatory_jlens_external.txt",
)


def test_registry_covers_every_frozen_confirmatory_job_with_exact_identity():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registry = load_edge_registry(REGISTRY)
    assert payload["schema_version"] == 1
    rows = payload["edges"]
    by_tag = {row["tag"]: row for row in rows}
    assert len(by_tag) == len(rows)

    jobs = [job for manifest in MANIFESTS for job in load_manifest(manifest)]
    assert set(by_tag) == {job.tag for job in jobs}
    for job in jobs:
        row = by_tag[job.tag]
        assert (row["parent"], row["descendant"]) == (
            job.parent,
            job.descendant,
        )
        assert row["evidence_url"].startswith("https://huggingface.co/")
        assert row["lineage_relation"] in {
            "direct_declared",
            "declared_ancestor_comparison",
            "repository_name_assertion",
        }
        assert registry.require(job.tag, job.parent, job.descendant).tag == job.tag


def test_registry_does_not_upgrade_missing_lineage_metadata_to_a_direct_edge():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    weak = [
        row for row in payload["edges"] if row["lineage_evidence"] == "missing_model_card_lineage"
    ]
    assert weak
    assert all(row["lineage_relation"] == "repository_name_assertion" for row in weak)


def test_registry_rejects_checkpoint_identity_drift():
    registry = load_edge_registry(REGISTRY)
    with pytest.raises(ValueError, match="identity mismatch"):
        registry.require("q8_base2it_confirm", "Qwen/wrong", "Qwen/Qwen3-8B")


def test_exclusion_ledger_is_disjoint_from_frozen_registry():
    registry = load_edge_registry(REGISTRY)
    ledger = json.loads((ROOT / "zoo" / "data" / "exclusion_ledger.json").read_text())
    registered_tags = {edge.tag for edge in registry.edges}
    attempted_tags = set()
    for row in ledger["exclusions"]:
        assert set(row) == {
            "attempted_tag",
            "attempted_parent",
            "descendant",
            "reason_code",
            "reason",
            "disposition",
        }
        attempted_tags.add(row["attempted_tag"])
    assert attempted_tags.isdisjoint(registered_tags)
