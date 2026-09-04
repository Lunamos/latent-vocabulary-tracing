#!/usr/bin/env python3
"""Report observed token-class composition without running a language model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import transformers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from latent_vocabulary_tracing.taxonomy import (  # noqa: E402
    TRACE_CATEGORIES,
    categorize_trace_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer", help="local path or Hugging Face tokenizer ID")
    parser.add_argument("probes", type=Path, help="JSONL with text, kind, and prompt_len")
    parser.add_argument("--max-length", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    counts: dict[str, Counter[str]] = {}
    for line_number, line in enumerate(
        args.probes.read_text(encoding="utf-8").splitlines(), start=1
    ):
        row = json.loads(line)
        for field in ("text", "kind", "prompt_len"):
            if field not in row:
                raise ValueError(f"{args.probes}:{line_number}: missing {field!r}")
        token_ids = tokenizer(
            row["text"], truncation=True, max_length=args.max_length
        ).input_ids
        response_ids = token_ids[int(row["prompt_len"]) :]
        domain_counts = counts.setdefault(row["kind"], Counter())
        for token_id in response_ids:
            piece = tokenizer.decode([token_id])
            domain_counts[categorize_trace_token(piece)] += 1

    payload = {
        "schema_version": 1,
        "tokenizer": args.tokenizer,
        "probes": str(args.probes),
        "max_length": args.max_length,
        "domains": {},
    }
    for domain, domain_counts in sorted(counts.items()):
        total = sum(domain_counts.values())
        payload["domains"][domain] = {
            "response_tokens": total,
            "category_percent": {
                category: 100.0 * domain_counts[category] / total
                for category in TRACE_CATEGORIES
            },
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
