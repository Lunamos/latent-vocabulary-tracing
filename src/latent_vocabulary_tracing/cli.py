"""Command-line entry point for repository-safe inspection utilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .manifest import load_manifest
from .registry import load_edge_registry
from .summary import SummaryContract, load_summary, summary_view
from .taxonomy import categorize_functional_token, categorize_token, categorize_trace_token


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lvt", description="Latent Vocabulary Tracing")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="work with pair manifests")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    check = manifest_commands.add_parser("check", help="validate a manifest")
    check.add_argument("path")
    check.add_argument(
        "--edge-registry",
        help="also require every tag and exact parent/descendant ID in this registry",
    )

    summary = commands.add_parser("summary", help="inspect a trace summary")
    summary.add_argument("path")
    summary.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    summary.add_argument(
        "--contract-readout",
        choices=("J", "LL"),
        help="also require the primary parent-anchored response-KL contract",
    )
    summary.add_argument("--require-categories", action="store_true")
    summary.add_argument("--require-fp32-store", action="store_true")
    summary.add_argument(
        "--edge-registry",
        help="also require the summary to be bound to this exact registry content",
    )

    token = commands.add_parser("token", help="work with decoded token strings")
    token_commands = token.add_subparsers(dest="token_command", required=True)
    classify = token_commands.add_parser("classify", help="classify literal token strings")
    classify.add_argument(
        "--scheme",
        choices=("structural", "functional", "trace"),
        default="structural",
        help="use the fine structural taxonomy or the reader-facing functional taxonomy",
    )
    classify.add_argument("tokens", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        jobs = load_manifest(args.path)
        if args.edge_registry:
            registry = load_edge_registry(args.edge_registry)
            for job in jobs:
                registry.require(job.tag, job.parent, job.descendant)
        print(f"valid manifest: {len(jobs)} unique jobs")
        return
    if args.command == "summary":
        if (
            args.require_categories or args.require_fp32_store or args.edge_registry
        ) and not args.contract_readout:
            raise SystemExit("--require-* and --edge-registry need --contract-readout")
        registry = load_edge_registry(args.edge_registry) if args.edge_registry else None
        contract = (
            SummaryContract(
                readout=args.contract_readout,
                require_category_statistics=args.require_categories,
                require_fp32_store=args.require_fp32_store,
                require_edge_registry=registry is not None,
                edge_registry_hash=registry.digest if registry else None,
            )
            if args.contract_readout
            else None
        )
        view = summary_view(load_summary(args.path, contract=contract))
        if args.json:
            print(json.dumps(view, indent=2, ensure_ascii=False))
        else:
            for key, value in view.items():
                print(f"{key}: {value}")
        return
    if args.command == "token":
        classifier = {
            "structural": categorize_token,
            "functional": categorize_functional_token,
            "trace": categorize_trace_token,
        }[args.scheme]
        for token in args.tokens:
            print(f"{token!r}\t{classifier(token)}")
        return
    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
