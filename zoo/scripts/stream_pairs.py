"""Disk-bounded, contract-aware runner for paired checkpoint readouts.

Each manifest row is ``TAG | PARENT | DESCENDANT | EXTRA``. Remote model
specifications may use ``repo[@revision][::subdir]``. We load from downloaded
paths but pass the original model identities to ``readout_pair.py`` so result
provenance never degrades into a machine-local cache path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", "/localscratch/zjin350/hf_cache")

from huggingface_hub import snapshot_download  # noqa: E402, I001


ZOO = Path(os.environ.get("ZOO", Path(__file__).resolve().parents[1])).resolve()
ROOT = ZOO.parent
LOG = ZOO / "logs"
RESULTS = ZOO / "results"
HF_HUB = Path(os.environ["HF_HOME"]) / "hub"
PYTHON = os.environ.get("ZOO_PY", sys.executable)
ALLOW = ("*.safetensors", "*.json", "*.txt", "*.jinja", "*.model", "*.py", "*.tiktoken")

sys.path.insert(0, str(ROOT / "src"))

from latent_vocabulary_tracing.manifest import atomic_claim, load_manifest  # noqa: E402
from latent_vocabulary_tracing.registry import load_edge_registry  # noqa: E402
from latent_vocabulary_tracing.summary import SummaryContract, load_summary  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    original: str
    repository: str | None
    revision: str | None
    subdirectory: str | None
    identity: str


@dataclass(frozen=True)
class FetchedModel:
    path: Path
    spec: ModelSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--keep", default="")
    parser.add_argument(
        "--min-free",
        type=float,
        default=60,
        help="minimum free GPU memory in GiB before loading a pair",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--contract-readout", choices=("J", "LL"))
    parser.add_argument("--require-categories", action="store_true")
    parser.add_argument("--require-fp32-store", action="store_true")
    parser.add_argument("--expected-probes-per-domain", type=int)
    parser.add_argument(
        "--edge-registry",
        type=Path,
        help="validate every manifest edge and bind its content hash into each summary",
    )
    args = parser.parse_args()
    if (
        args.require_categories
        or args.require_fp32_store
        or args.expected_probes_per_domain
    ) and not args.contract_readout:
        parser.error("contract requirements need --contract-readout")
    if args.expected_probes_per_domain is not None and args.expected_probes_per_domain <= 0:
        parser.error("--expected-probes-per-domain must be positive")
    if args.min_free <= 0:
        parser.error("--min-free must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def log(message: str) -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    line = (
        f"{time.strftime('%F %T')} "
        f"[gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}] {message}"
    )
    print(line, flush=True)
    with (LOG / "stream_pairs.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def parse_model_spec(value: str) -> ModelSpec:
    if Path(value).is_dir():
        return ModelSpec(value, None, None, None, value)
    repository_revision, separator, subdirectory = value.partition("::")
    repository, revision_separator, revision = repository_revision.partition("@")
    subdirectory = subdirectory if separator else None
    revision = revision if revision_separator else None
    identity = repository + (f"::{subdirectory}" if subdirectory else "")
    return ModelSpec(value, repository, revision, subdirectory, identity)


def gpu_free_gib() -> float:
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return float(output.strip()) / 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def wait_for_gpu(minimum_gib: float, poll_seconds: int) -> None:
    while True:
        free = gpu_free_gib()
        if free >= minimum_gib:
            return
        log(f"waiting for GPU memory ({free:.0f} GiB free < {minimum_gib:.0f} GiB)")
        time.sleep(poll_seconds)


def wait_for_disk(minimum_gib: float, poll_seconds: int) -> None:
    while True:
        free = shutil.disk_usage(ZOO).free / 2**30
        if free >= minimum_gib:
            return
        log(f"waiting for disk ({free:.0f} GiB free < {minimum_gib:.0f} GiB)")
        time.sleep(max(poll_seconds, 60))


def sanitize_config(path: Path) -> None:
    """Repair released configs that encode integral architecture fields as floats."""

    config_path = path / "config.json"
    if not config_path.is_file():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    changed = False

    def fix(value: object) -> None:
        nonlocal changed
        if isinstance(value, dict):
            for key, item in value.items():
                if (
                    isinstance(item, float)
                    and item.is_integer()
                    and any(
                        fragment in key
                        for fragment in (
                            "position",
                            "size",
                            "layers",
                            "heads",
                            "dim",
                            "vocab",
                            "token_id",
                            "window",
                            "length",
                        )
                    )
                ):
                    value[key] = int(item)
                    changed = True
                else:
                    fix(item)
        elif isinstance(value, list):
            for item in value:
                fix(item)

    fix(config)
    if changed:
        config_path.chmod(0o644)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        log(f"sanitized integral floats in {config_path}")


def fetch(spec: ModelSpec, minimum_disk_gib: float, poll_seconds: int) -> FetchedModel | None:
    if spec.repository is None:
        return FetchedModel(Path(spec.original).resolve(), spec)
    wait_for_disk(minimum_disk_gib, poll_seconds)
    for attempt in range(1, 4):
        try:
            if spec.subdirectory:
                allow_patterns = [f"{spec.subdirectory}/{pattern}" for pattern in ALLOW]
                allow_patterns.extend(("*.json", "*.txt", "*.jinja"))
            else:
                allow_patterns = list(ALLOW)
            snapshot = Path(
                snapshot_download(
                    spec.repository,
                    revision=spec.revision,
                    allow_patterns=allow_patterns,
                    max_workers=8,
                )
            )
            path = snapshot / spec.subdirectory if spec.subdirectory else snapshot
            if spec.subdirectory and not (path / "config.json").is_file():
                path.mkdir(parents=True, exist_ok=True)
                for pattern in ("*.json", "*.txt", "*.jinja"):
                    for source in snapshot.glob(pattern):
                        destination = path / source.name
                        if not destination.exists():
                            shutil.copy(source, destination)
            sanitize_config(path)
            return FetchedModel(path, spec)
        except Exception as error:  # network/model repositories fail heterogeneously
            log(
                f"download attempt {attempt}/3 failed for {spec.original}: "
                f"{type(error).__name__}: {str(error)[:180]}"
            )
            if attempt < 3:
                time.sleep(60)
    return None


def delete_download(model: FetchedModel, keep: set[str]) -> None:
    repository = model.spec.repository
    if repository is None or repository in keep:
        return
    target = model.path
    if not target.exists():
        return
    size_gib = sum(
        path.stat().st_size
        for path in target.rglob("*")
        if path.is_file() and not path.is_symlink()
    ) / 2**30
    shutil.rmtree(target)

    repository_root = HF_HUB / f"models--{repository.replace('/', '--')}"
    linked = {
        path.resolve()
        for path in (repository_root / "snapshots").glob("**/*")
        if path.is_symlink()
    }
    for blob in (repository_root / "blobs").glob("*"):
        if blob.resolve() not in linked:
            blob.unlink(missing_ok=True)
    log(f"deleted downloaded descendant {target} ({size_gib:.1f} GiB)")


def validate_result(path: Path, args: argparse.Namespace) -> None:
    if not args.contract_readout:
        if not path.is_file():
            raise ValueError("summary file was not written")
        return
    contract = SummaryContract(
        readout=args.contract_readout,
        require_category_statistics=args.require_categories,
        require_fp32_store=args.require_fp32_store,
        expected_probes_per_domain=args.expected_probes_per_domain,
        require_edge_registry=args.edge_registry is not None,
        edge_registry_hash=getattr(args, "edge_registry_hash", None),
    )
    load_summary(path, contract=contract)


def write_marker(path: Path, *, return_code: int, detail: str = "") -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(f"rc={return_code}\n{detail}", encoding="utf-8")
    os.replace(temporary, path)


def run_job(job, args: argparse.Namespace, keep: set[str]) -> None:
    done = LOG / f"ro_{job.tag}.DONE"
    failed = LOG / f"ro_{job.tag}.FAIL"
    if done.exists() or (failed.exists() and not args.retry_failed):
        return
    wait_for_gpu(args.min_free, args.poll_seconds)
    if done.exists() or (failed.exists() and not args.retry_failed):
        return

    with atomic_claim(LOG, f"ro_{job.tag}") as claim:
        if claim is None:
            return
        if args.retry_failed:
            failed.unlink(missing_ok=True)
        parent = fetch(parse_model_spec(job.parent), args.min_free, args.poll_seconds)
        descendant = fetch(parse_model_spec(job.descendant), args.min_free, args.poll_seconds)
        if parent is None or descendant is None:
            write_marker(failed, return_code=1, detail="download failed\n")
            log(f"FAIL {job.tag}: download failed")
            return

        wait_for_gpu(args.min_free, args.poll_seconds)
        summary = RESULTS / f"ro_{job.tag}_summary.json"
        command = [
            PYTHON,
            str(ZOO / "scripts" / "readout_pair.py"),
            str(parent.path),
            str(descendant.path),
            job.tag,
            "--model-a-id",
            parent.spec.identity,
            "--model-b-id",
            descendant.spec.identity,
        ]
        if args.edge_registry is not None:
            command.extend(("--edge-registry", str(args.edge_registry.resolve())))
        command.extend(job.extra_args)
        log(f"START {job.tag}: {job.parent} vs {job.descendant}")
        job_log = LOG / f"ro_{job.tag}.log"
        with job_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=ZOO,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        detail = ""
        valid = False
        if completed.returncode == 0:
            try:
                validate_result(summary, args)
                valid = True
            except (OSError, ValueError) as error:
                detail = f"contract error: {error}\n"
                log(f"contract failure for {job.tag}: {error}")
        marker = done if valid else failed
        write_marker(marker, return_code=completed.returncode, detail=detail)
        log(f"{'DONE' if valid else 'FAIL'} {job.tag} rc={completed.returncode}")
        delete_download(descendant, keep)


def main() -> None:
    args = parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    keep = {value for value in args.keep.split(",") if value}
    jobs = load_manifest(args.jobs)
    if args.edge_registry is not None:
        registry = load_edge_registry(args.edge_registry)
        for job in jobs:
            registry.require(job.tag, job.parent, job.descendant)
        args.edge_registry_hash = registry.digest
    for job in jobs:
        run_job(job, args, keep)
    log("jobs exhausted")


if __name__ == "__main__":
    main()
