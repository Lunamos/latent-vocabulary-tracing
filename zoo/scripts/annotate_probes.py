"""Add deterministic token-role spans to an existing frozen probe file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from latent_vocabulary_tracing.spans import infer_role_spans, validate_role_spans  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("probes", type=Path)
    parser.add_argument("tokenizer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-len", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    output = args.output or args.probes
    rows = []
    for line in args.probes.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        encoded = tokenizer(
            row["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_len,
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        if len(offsets) != row["n_tok"]:
            raise ValueError(
                f"{row['key']}: stored n_tok={row['n_tok']} but tokenizer produced {len(offsets)}"
            )
        spans = infer_role_spans(
            row["text"],
            kind=row["kind"],
            prompt_len=row["prompt_len"],
            offsets=offsets,
        )
        validate_role_spans(spans, n_tokens=len(offsets))
        row["role_spans"] = {
            name: [list(span) for span in values] for name, values in spans.items()
        }
        rows.append(row)

    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"annotated {len(rows)} probes in {output}")


if __name__ == "__main__":
    main()
