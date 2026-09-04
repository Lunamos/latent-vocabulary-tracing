import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "zoo" / "data" / "probe_sources.json"


def test_probe_sources_pin_immutable_revisions() -> None:
    sources = json.loads(PROVENANCE.read_text())["sources"]
    assert set(sources) == {"math", "code", "agent", "neutral"}
    for source in sources.values():
        revision = source["revision"]
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)
        assert source["license"]


def test_main_and_robustness_source_ranges_are_disjoint() -> None:
    sources = json.loads(PROVENANCE.read_text())["sources"]
    code = sources["code"]["selection"]
    assert code["main_task_ids"] == ["HumanEval/0", "HumanEval/29"]
    assert code["robustness_task_ids"] == ["HumanEval/60", "HumanEval/89"]

    neutral = sources["neutral"]["selection"]
    main = {
        neutral["main_qualifying_start"] + neutral["qualifying_stride"] * index
        for index in range(neutral["count"])
    }
    robustness = {
        neutral["robustness_qualifying_start"] + neutral["qualifying_stride"] * index
        for index in range(neutral["count"])
    }
    assert main.isdisjoint(robustness)


def test_public_probe_builder_has_no_workstation_path() -> None:
    source = (ROOT / "zoo" / "scripts" / "build_probes.py").read_text()
    assert "/localscratch/" not in source
    assert "/nethome/" not in source
    assert "probe_sources.json" in source
