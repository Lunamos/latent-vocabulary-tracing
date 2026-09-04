import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def load_merge_function():
    path = Path(__file__).parents[1] / "zoo" / "scripts" / "merge_jlens.py"
    spec = importlib.util.spec_from_file_location("merge_jlens_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.merge_jlens


def write_shard(path, value, prompt_slice):
    torch.save(
        {"J": {4: torch.full((2, 2), value)}, "n_prompts": 2, "source_layers": [4], "d_model": 2},
        path,
    )
    metadata = {
        "model": "org/base",
        "revision": "abc",
        "architecture": "Model",
        "config_hash": "config",
        "tokenizer_hash": "tokenizer",
        "source_layers": [4],
        "target_layer": 5,
        "d_model": 2,
        "dim_batch": 1,
        "max_seq_len": 8,
        "prompt_slice": prompt_slice,
    }
    Path(str(path) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_merge_jlens_weights_disjoint_shards_and_writes_provenance(tmp_path):
    merge_jlens = load_merge_function()
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    write_shard(first, 1.0, [0, 2])
    write_shard(second, 3.0, [2, 4])
    output = tmp_path / "merged.pt"
    metadata = merge_jlens(output, [first, second])
    merged = torch.load(output, map_location="cpu", weights_only=True)
    assert merged["n_prompts"] == 4
    assert merged["J"][4].float().numpy() == pytest.approx(2.0)
    assert metadata["prompt_slices"] == [[0, 2], [2, 4]]
    assert metadata["sha256"]


def test_merge_jlens_rejects_overlapping_prompt_slices(tmp_path):
    merge_jlens = load_merge_function()
    first, second = tmp_path / "a.pt", tmp_path / "b.pt"
    write_shard(first, 1.0, [0, 3])
    write_shard(second, 3.0, [2, 4])
    with pytest.raises(ValueError, match="overlapping"):
        merge_jlens(tmp_path / "merged.pt", [first, second])
