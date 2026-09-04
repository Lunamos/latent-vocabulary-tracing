"""Path-independent fingerprints for model and analysis provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import PurePath
from typing import Any

_VOLATILE_CONFIG_FIELDS = {
    "_commit_hash",
    "_name_or_path",
    "transformers_version",
}


def canonicalize_model_config(value: Any) -> Any:
    """Remove loader metadata and normalize JSON-equivalent numeric values."""

    if isinstance(value, Mapping):
        return {
            str(key): canonicalize_model_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key not in _VOLATILE_CONFIG_FIELDS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonicalize_model_config(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def stable_json_hash(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for a JSON-like value."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def model_config_hash(config: Mapping[str, Any]) -> str:
    """Hash architecture config independently of Hub/local loading paths."""

    return stable_json_hash(canonicalize_model_config(config))


def snapshot_revision_from_path(path: str | PurePath) -> str | None:
    """Extract a content-addressed Hub revision from a snapshot path.

    Hugging Face loaders do not consistently retain ``_commit_hash`` when a
    checkpoint is opened through a local overlay. The resolved snapshot path
    is authoritative and remains available to the queue that downloaded it.
    """

    parts = PurePath(path).parts
    for index, part in enumerate(parts[:-1]):
        if part != "snapshots":
            continue
        revision = parts[index + 1]
        if len(revision) == 40 and all(character in "0123456789abcdef" for character in revision):
            return revision
    return None
