import json

import pytest

from latent_vocabulary_tracing.cli import main


def _write_registry(path, *, parent="org/parent"):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "edges": [
                    {
                        "tag": "edge",
                        "parent": parent,
                        "descendant": "org/descendant",
                        "family": "family",
                        "target": "math",
                        "recipe": "SFT",
                        "analysis_role": "headline",
                        "lineage_relation": "direct_declared",
                        "lineage_evidence": "model_card_base_model",
                        "evidence_url": "https://huggingface.co/org/descendant",
                        "license": "apache-2.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_manifest_cli_can_require_exact_registry_identity(tmp_path, capsys):
    manifest = tmp_path / "jobs.txt"
    manifest.write_text("edge | org/parent | org/descendant\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    _write_registry(registry)

    main(["manifest", "check", str(manifest), "--edge-registry", str(registry)])
    assert "1 unique jobs" in capsys.readouterr().out


def test_manifest_cli_rejects_registry_identity_drift(tmp_path):
    manifest = tmp_path / "jobs.txt"
    manifest.write_text("edge | org/parent | org/descendant\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    _write_registry(registry, parent="org/wrong")

    with pytest.raises(ValueError, match="identity mismatch"):
        main(["manifest", "check", str(manifest), "--edge-registry", str(registry)])
