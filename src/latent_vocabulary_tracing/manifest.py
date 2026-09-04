"""Portable experiment-manifest parsing and atomic job claims."""

from __future__ import annotations

import contextlib
import os
import re
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class PairJob:
    """One parent-to-descendant comparison from a pipe-delimited manifest."""

    tag: str
    parent: str
    descendant: str
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _TAG.fullmatch(self.tag):
            raise ValueError(f"unsafe job tag: {self.tag!r}")
        if not self.parent or not self.descendant:
            raise ValueError("parent and descendant must be non-empty")


def parse_manifest_line(line: str) -> PairJob | None:
    """Parse ``TAG | PARENT | DESCENDANT | EXTRA``; comments are allowed."""

    content = line.split("#", 1)[0].strip()
    if not content:
        return None
    fields = [field.strip() for field in content.split("|")]
    if len(fields) not in (3, 4):
        raise ValueError(f"expected 3 or 4 pipe-delimited fields, got {len(fields)}")
    extra = tuple(shlex.split(fields[3])) if len(fields) == 4 else ()
    return PairJob(fields[0], fields[1], fields[2], extra)


def load_manifest(path: str | Path) -> list[PairJob]:
    """Load a manifest and reject duplicate tags."""

    source = Path(path)
    jobs: list[PairJob] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        try:
            job = parse_manifest_line(line)
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error
        if job is None:
            continue
        if job.tag in seen:
            raise ValueError(f"{source}:{line_number}: duplicate tag {job.tag!r}")
        seen.add(job.tag)
        jobs.append(job)
    return jobs


@contextlib.contextmanager
def atomic_claim(directory: str | Path, tag: str) -> Iterator[Path | None]:
    """Claim ``tag`` across workers using an atomic marker-file creation.

    The context yields ``None`` when another process owns the claim. A claim is
    always released when the context exits; durable DONE/FAIL markers belong to
    the queue runner rather than this primitive.
    """

    if not _TAG.fullmatch(tag):
        raise ValueError(f"unsafe job tag: {tag!r}")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    claim = root / f"{tag}.claim"
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        yield None
        return
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        yield claim
    finally:
        claim.unlink(missing_ok=True)
