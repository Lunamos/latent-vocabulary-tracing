#!/usr/bin/env python3
"""Convert legacy raw-Wikitext controls into chat-matched continuations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from latent_vocabulary_tracing.probes import select_spaced_text_records  # noqa: E402
from latent_vocabulary_tracing.spans import (  # noqa: E402
    infer_role_spans,
    validate_role_spans,
)

SYSTEM = "Continue the supplied passage in neutral expository prose."
USER_PREFIX = "Continue this encyclopedia passage:\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("probes", type=Path)
    parser.add_argument("tokenizer")
    parser.add_argument("--style", choices=("qwen", "olmo"), required=True)
    parser.add_argument("--context-tokens", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument(
        "--resample-wikitext",
        action="store_true",
        help="replace neutral rows with the frozen 30-document WikiText sample",
    )
    parser.add_argument("--source-start", type=int, default=1500)
    parser.add_argument("--source-stride", type=int, default=97)
    return parser.parse_args()


def render_prefix(tokenizer, *, context: str, style: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_PREFIX + context},
    ]
    if style == "qwen":
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    system = (
        SYSTEM
        + " You do not currently have access to any functions. <functions></functions>"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{USER_PREFIX}{context}<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )


def main() -> None:
    args = parse_args()
    if args.context_tokens <= 0:
        raise ValueError("--context-tokens must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [
        json.loads(line)
        for line in args.probes.read_text(encoding="utf-8").splitlines()
    ]
    neutral_count = sum(row.get("kind") == "neutral" for row in rows)
    sources = None
    if args.resample_wikitext:
        from datasets import load_dataset

        sources = select_spaced_text_records(
            load_dataset(
                "Salesforce/wikitext",
                "wikitext-103-raw-v1",
                split="train",
            ),
            count=neutral_count,
            start=args.source_start,
            stride=args.source_stride,
        )
    converted = 0
    for row in rows:
        if row.get("kind") != "neutral":
            continue
        source = sources[converted] if sources is not None else None
        if source is None and row.get("meta", {}).get("control") == "chat_matched_continuation":
            raise ValueError(f"{row.get('key')}: neutral row is already chat matched")
        if source is None and row.get("prompt_len") != 0:
            raise ValueError(f"{row.get('key')}: expected a legacy raw neutral row")
        source_text = source["text"] if source is not None else row["text"]
        source_ids = tokenizer.encode(source_text, add_special_tokens=False)
        if len(source_ids) <= args.context_tokens:
            raise ValueError(f"{row.get('key')}: source is too short for a continuation")
        context = tokenizer.decode(source_ids[: args.context_tokens])
        continuation = tokenizer.decode(source_ids[args.context_tokens :])
        prefix = render_prefix(tokenizer, context=context, style=args.style)
        row["text"] = prefix + continuation
        row["prompt_len"] = len(tokenizer.encode(prefix, add_special_tokens=False))
        row["meta"] = {
            **row.get("meta", {}),
            "idx": (
                source["qualifying_index"]
                if source is not None
                else row.get("meta", {}).get("idx")
            ),
            "article": source["article"] if source is not None else None,
            "control": "chat_matched_continuation",
            "source_context_tokens": args.context_tokens,
            "source_dataset": "Salesforce/wikitext:wikitext-103-raw-v1",
            "source_split": "train",
            "source_sampling": (
                "spaced_qualifying_records" if source is not None else "legacy_existing_rows"
            ),
            "source_start": args.source_start if source is not None else None,
            "source_stride": args.source_stride if source is not None else None,
        }
        encoded = tokenizer(
            row["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        row["n_tok"] = len(offsets)
        spans = infer_role_spans(
            row["text"],
            kind="neutral",
            prompt_len=row["prompt_len"],
            offsets=offsets,
        )
        validate_role_spans(spans, n_tokens=len(offsets))
        row["role_spans"] = {
            role: [list(span) for span in role_spans]
            for role, role_spans in spans.items()
        }
        converted += 1

    if converted == 0:
        raise ValueError("probe file contains no legacy neutral rows")
    temporary = args.probes.with_name(f"{args.probes.name}.tmp.{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, args.probes)
    print(f"converted {converted} neutral probes in {args.probes}")


if __name__ == "__main__":
    main()
