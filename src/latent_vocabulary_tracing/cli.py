"""Command-line entry point for repository-safe inspection utilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .manifest import load_manifest
from .summary import load_summary, summary_view
from .taxonomy import categorize_functional_token, categorize_token


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lvt", description="Latent Vocabulary Tracing")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="work with pair manifests")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    check = manifest_commands.add_parser("check", help="validate a manifest")
    check.add_argument("path")

    summary = commands.add_parser("summary", help="inspect a trace summary")
    summary.add_argument("path")
    summary.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    token = commands.add_parser("token", help="work with decoded token strings")
    token_commands = token.add_subparsers(dest="token_command", required=True)
    classify = token_commands.add_parser("classify", help="classify literal token strings")
    classify.add_argument(
        "--scheme",
        choices=("structural", "functional"),
        default="structural",
        help="use the fine structural taxonomy or the reader-facing functional taxonomy",
    )
    classify.add_argument("tokens", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "manifest":
        jobs = load_manifest(args.path)
        print(f"valid manifest: {len(jobs)} unique jobs")
        return
    if args.command == "summary":
        view = summary_view(load_summary(args.path))
        if args.json:
            print(json.dumps(view, indent=2, ensure_ascii=False))
        else:
            for key, value in view.items():
                print(f"{key}: {value}")
        return
    if args.command == "token":
        classifier = (
            categorize_functional_token if args.scheme == "functional" else categorize_token
        )
        for token in args.tokens:
            print(f"{token!r}\t{classifier(token)}")
        return
    raise AssertionError("unreachable")
