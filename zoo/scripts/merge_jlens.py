"""Merge disjoint Jlens fit shards with metadata and hash checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("shards", nargs="+", type=Path)
    return parser.parse_args()


def merge_jlens(output: Path, shard_paths: list[Path]) -> dict:
    if len(shard_paths) < 2:
        raise ValueError("merge requires at least two disjoint shards")

    shards = [torch.load(path, map_location="cpu", weights_only=True) for path in shard_paths]
    required = {"J", "n_prompts", "source_layers", "d_model"}
    for path, shard in zip(shard_paths, shards, strict=True):
        missing = required.difference(shard)
        if missing:
            raise ValueError(f"{path} is missing keys {sorted(missing)}")
        if shard["n_prompts"] <= 0:
            raise ValueError(f"{path} has no fitted prompts")

    first = shards[0]
    for path, shard in zip(shard_paths[1:], shards[1:], strict=True):
        if shard["source_layers"] != first["source_layers"]:
            raise ValueError(f"{path} has different source layers")
        if shard["d_model"] != first["d_model"]:
            raise ValueError(f"{path} has different hidden width")

    metadata = []
    for path in shard_paths:
        metadata_path = Path(str(path) + ".meta.json")
        if not metadata_path.exists():
            raise ValueError(f"missing shard metadata {metadata_path}")
        metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    invariant_fields = (
        "model",
        "revision",
        "architecture",
        "config_hash",
        "tokenizer_hash",
        "source_layers",
        "target_layer",
        "d_model",
        "dim_batch",
        "max_seq_len",
    )
    for field in invariant_fields:
        values = [item.get(field) for item in metadata]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"shard metadata disagrees on {field}: {values!r}")

    slices = sorted(tuple(item["prompt_slice"]) for item in metadata)
    for left, right in zip(slices, slices[1:], strict=False):
        if left[1] > right[0]:
            raise ValueError(f"overlapping prompt slices: {left!r} and {right!r}")

    total = sum(int(shard["n_prompts"]) for shard in shards)
    merged_jacobians = {}
    for layer in first["source_layers"]:
        weighted = sum(
            shard["J"][layer].float() * int(shard["n_prompts"]) for shard in shards
        )
        merged_jacobians[layer] = (weighted / total).half()
    payload = {
        "J": merged_jacobians,
        "n_prompts": total,
        "source_layers": first["source_layers"],
        "d_model": first["d_model"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)

    merged_metadata = {
        "schema_version": 1,
        **{field: metadata[0].get(field) for field in invariant_fields},
        "n_prompts": total,
        "prompt_slices": [list(item) for item in slices],
        "shards": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "n_prompts": int(shard["n_prompts"]),
            }
            for path, shard in zip(shard_paths, shards, strict=True)
        ],
        "sha256": sha256_file(output),
    }
    Path(str(output) + ".meta.json").write_text(
        json.dumps(merged_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return merged_metadata


def main() -> None:
    args = parse_args()
    metadata = merge_jlens(args.output, args.shards)
    print(
        f"merged {len(args.shards)} shards / {metadata['n_prompts']} prompts -> "
        f"{args.output} sha256={metadata['sha256']}"
    )


if __name__ == "__main__":
    main()
