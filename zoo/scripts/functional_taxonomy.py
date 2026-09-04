"""Estimate functional vocabulary shifts with prompt-clustered uncertainty.

For every response token and selected layer, this script decodes the union of
the parent and descendant top-k vocabulary supports and computes category-level probability-mass
changes

    delta_C = sum_{v in top-k union and category C} (p_B(v) - p_A(v)).

Point estimates average positions within each probe, layers within a declared
band, and then probes.  Confidence intervals resample probes, so a long answer
cannot dominate the result.  Promoted and suppressed masses are retained
separately because signed changes can cancel within a category.

The result is descriptive: categories state the linguistic role of changed
token pieces.  They do not by themselves identify why an optimization target
caused the change.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from latent_vocabulary_tracing.taxonomy import (  # noqa: E402
    FUNCTIONAL_CATEGORIES,
    categorize_functional_token,
)

DEFAULT_TAGS = (
    "q17_capsd_cap4k_fll",
    "q17_sft_ot3_fll",
    "q17_opd_dapo_fll",
    "q17_it_fll",
    "q17_opsa_fll",
    "q17_grpo_best_fll",
    "q17_ttrl_best_fll",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="*", default=DEFAULT_TAGS)
    parser.add_argument("--readout", choices=("LL", "J"), default="LL")
    parser.add_argument("--kind", default="math")
    parser.add_argument("--band", default="15-20")
    parser.add_argument("--support", choices=("union", "parent"), default="union")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--topn", type=int, default=5)
    return parser.parse_args()


def model_id(path: str) -> str:
    match = re.search(r"models--([^/]+)--([^/]+)/snapshots", path)
    return f"{match.group(1)}/{match.group(2)}" if match else path


def interval(values: np.ndarray, draws: int, seed: int) -> dict[str, list[float]]:
    """Mean and prompt-clustered percentile interval along axis zero."""

    rng = np.random.default_rng(seed)
    n_examples = values.shape[0]
    indices = rng.integers(0, n_examples, size=(draws, n_examples))
    boot = values[indices].mean(axis=1)
    return {
        "mean": values.mean(axis=0).tolist(),
        "ci_low": np.quantile(boot, 0.025, axis=0).tolist(),
        "ci_high": np.quantile(boot, 0.975, axis=0).tolist(),
    }


def readable_token(token: str) -> str:
    rendered = token.replace("\n", "↵").replace("\t", "⇥")
    if not rendered.strip():
        rendered = rendered.replace(" ", "·")
    return rendered


def analyze(tag: str, args: argparse.Namespace) -> dict:
    path = ROOT / "zoo" / "results" / f"ro_{tag}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id(data["model_a"]))
    layers = [int(layer) for layer in data["layers"]]
    lo, hi = (int(value) for value in args.band.split("-"))
    band_indices = [index for index, layer in enumerate(layers) if lo <= layer <= hi]
    if not band_indices:
        raise ValueError(f"band {args.band} does not intersect layers for {tag}")

    n_categories = len(FUNCTIONAL_CATEGORIES)
    category_cache: dict[int, int] = {}
    token_text: dict[int, str] = {}
    examples: list[np.ndarray] = []
    examples_up: list[np.ndarray] = []
    examples_down: list[np.ndarray] = []
    token_net: dict[int, float] = {}
    token_up: dict[int, float] = {}
    token_down: dict[int, float] = {}
    n_positions = 0

    for record in data["records"]:
        if record["kind"] != args.kind:
            continue
        store = data["store"][record["key"]]
        response_lo = record["prompt_len"] - store.get("store_offset", 0)
        response_hi = store["n_pos"] - store.get("store_offset", 0)
        if response_hi <= response_lo:
            continue
        n_response = response_hi - response_lo
        n_positions += n_response
        signed = np.zeros((len(layers), n_categories), dtype=np.float64)
        promoted = np.zeros_like(signed)
        suppressed = np.zeros_like(signed)

        for layer_index, layer in enumerate(layers):
            cell = store[args.readout][layer]
            if "b_at_a" not in cell:
                raise ValueError(f"{tag} layer {layer} lacks cross logits on parent support")
            ids_a = cell["top_a"][response_lo:response_hi].long().numpy()
            ids_b = cell["top_b"][response_lo:response_hi].long().numpy()
            parent_logits = cell["val_a"][response_lo:response_hi].float().numpy()
            descendant_logits = cell["b_at_a"][response_lo:response_hi].float().numpy()
            parent_lse = cell["lse_a"][response_lo:response_hi].float().numpy()[:, None]
            descendant_lse = cell["lse_b"][response_lo:response_hi].float().numpy()[:, None]
            parent_probability = np.exp(parent_logits - parent_lse)
            descendant_probability = np.exp(descendant_logits - descendant_lse)
            delta_a = descendant_probability - parent_probability

            if args.support == "union":
                if "a_at_b" not in cell:
                    raise ValueError(
                        f"{tag} layer {layer} lacks cross logits on descendant support"
                    )
                descendant_top_probability = np.exp(
                    cell["val_b"][response_lo:response_hi].float().numpy() - descendant_lse
                )
                parent_at_b_probability = np.exp(
                    cell["a_at_b"][response_lo:response_hi].float().numpy() - parent_lse
                )
                new_under_b = ~(ids_b[:, :, None] == ids_a[:, None, :]).any(axis=-1)
                ids = np.concatenate((ids_a.ravel(), ids_b[new_under_b]))
                delta = np.concatenate(
                    (
                        delta_a.ravel(),
                        (descendant_top_probability - parent_at_b_probability)[new_under_b],
                    )
                )
            else:
                ids = ids_a.ravel()
                delta = delta_a.ravel()

            flat_ids = ids
            missing = np.setdiff1d(np.unique(flat_ids), np.fromiter(category_cache, dtype=int))
            for token_id in missing:
                token_id = int(token_id)
                text = tokenizer.decode([token_id])
                token_text[token_id] = text
                category_cache[token_id] = FUNCTIONAL_CATEGORIES.index(
                    categorize_functional_token(text)
                )
            category_ids = np.fromiter(
                (category_cache[int(token_id)] for token_id in flat_ids),
                dtype=np.int64,
                count=flat_ids.size,
            )
            flat_delta = delta
            signed[layer_index] = np.bincount(
                category_ids, weights=flat_delta, minlength=n_categories
            ) / n_response
            promoted[layer_index] = np.bincount(
                category_ids, weights=np.maximum(flat_delta, 0), minlength=n_categories
            ) / n_response
            suppressed[layer_index] = np.bincount(
                category_ids, weights=np.maximum(-flat_delta, 0), minlength=n_categories
            ) / n_response

            if layer_index in band_indices:
                for token_id, value in zip(flat_ids, flat_delta, strict=True):
                    token_id = int(token_id)
                    value = float(value) / n_response / len(band_indices)
                    token_net[token_id] = token_net.get(token_id, 0.0) + value
                    token_up[token_id] = token_up.get(token_id, 0.0) + max(value, 0.0)
                    token_down[token_id] = token_down.get(token_id, 0.0) + max(-value, 0.0)

        examples.append(signed)
        examples_up.append(promoted)
        examples_down.append(suppressed)

    if not examples:
        raise ValueError(f"no {args.kind!r} records found for {tag}")

    signed_examples = np.stack(examples)
    promoted_examples = np.stack(examples_up)
    suppressed_examples = np.stack(examples_down)
    band_signed = signed_examples[:, band_indices].mean(axis=1)
    band_promoted = promoted_examples[:, band_indices].mean(axis=1)
    band_suppressed = suppressed_examples[:, band_indices].mean(axis=1)

    def top_rows(category_index: int, direction: str) -> list[dict]:
        sign = 1.0 if direction == "promoted" else -1.0
        candidates = []
        for token_id, signed_mass in token_net.items():
            directed_mass = sign * signed_mass
            if directed_mass > 0 and category_cache[token_id] == category_index:
                candidates.append((directed_mass / len(examples), token_id))
        rows = []
        for mass, token_id in sorted(candidates, reverse=True)[: args.topn]:
            rows.append(
                {
                    "id": token_id,
                    "token": readable_token(token_text[token_id]),
                    "probability_points": sign * 100 * mass,
                    "signed_probability_points": 100 * token_net[token_id] / len(examples),
                }
            )
        return rows

    result = {
        "tag": tag,
        "model_a": model_id(data["model_a"]),
        "model_b": model_id(data["model_b"]),
        "decoder_mode": data.get("decoder_mode", "native"),
        "readout": args.readout,
        "kind": args.kind,
        "layers": layers,
        "band": [lo, hi],
        "support": f"{args.support}_top_k",
        "categories": list(FUNCTIONAL_CATEGORIES),
        "n_examples": len(examples),
        "n_response_positions": n_positions,
        "estimand": (
            "100 * mean_examples(mean_layers(mean_response_positions("
            f"sum_{args.support}_topk_category(p_descendant-p_parent))))"
        ),
        "band_signed_probability_points": interval(
            100 * band_signed, args.bootstrap, args.seed
        ),
        "band_promoted_probability_points": interval(
            100 * band_promoted, args.bootstrap, args.seed + 1
        ),
        "band_suppressed_probability_points": interval(
            100 * band_suppressed, args.bootstrap, args.seed + 2
        ),
        "layer_signed_probability_points": {
            "mean": (100 * signed_examples.mean(axis=0)).tolist(),
        },
        "top_tokens": {
            category: {
                "promoted": top_rows(index, "promoted"),
                "suppressed": top_rows(index, "suppressed"),
            }
            for index, category in enumerate(FUNCTIONAL_CATEGORIES)
        },
    }
    del data
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "zoo" / "analysis" / "functional_taxonomy"
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for tag in args.tags:
        result = analyze(tag, args)
        output_path = output_dir / f"{tag}_{args.readout}_{args.kind}_{args.band}.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        combined.append(result)
        print(f"wrote {output_path}")
        print(
            tag,
            dict(
                zip(
                    result["categories"],
                    np.round(result["band_signed_probability_points"]["mean"], 3),
                    strict=True,
                )
            ),
        )
    combined_path = output_dir / f"q17_variants_{args.readout}_{args.kind}_{args.band}.json"
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {combined_path}")


if __name__ == "__main__":
    main()
