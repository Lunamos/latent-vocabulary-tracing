"""Unified vocabulary readout for one (reference, comparison) model pair.

One HF forward per model per probe (output_hidden_states), then for every lens layer:
  J  : Jacobian lens  = unembed(h_l @ J_l^T)   (frozen lens J fitted on the reference base)
  LL : logit lens     = unembed(h_l)
The default preserves legacy model-native decoders. ``--decoder parent`` applies
the reference model's final normalization and unembedding to both states, which
is the primary parent-anchored LVT estimand. ``final`` always compares the two
models' actual output logits. Metrics per position are:
  kl_ab, kl_ba, js, jaccard@10, top-50 ids/vals of both, cross logits (a_at_b, b_at_a), lse.
Plus hidden-state stats per layer (cos, linear CKA, dnorm_rel) as in opd/explore/hidden_cka.py, and
lens faithfulness KL(final_X || Jlens_X), KL(final_X || LL_X) for X in {A, B}.

Conventions match jlens: block-L output = hidden_states[L+1]; residual fp32 ->
@J^T (J cast to fp32) -> cast to lm_head dtype -> final_norm -> lm_head -> fp32.
Tokenization uses tok(text, truncation, max_length=640).

Usage: python readout_pair.py MODEL_A MODEL_B TAG [--lens PATH] [--probes PATH]
       [--out DIR] [--kinds math,code,agent,neutral]
Outputs: OUT/ro_<TAG>.pt (store) and OUT/ro_<TAG>_summary.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
import transformers

ZOO = os.environ.get("ZOO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(ZOO)
sys.path.insert(0, os.path.join(ROOT, "src"))

import latent_vocabulary_tracing.taxonomy as taxonomy_module  # noqa: E402
from latent_vocabulary_tracing.metrics import (  # noqa: E402
    benjamini_hochberg,
    deterministic_balanced_split,
    one_sided_sign_test,
)
from latent_vocabulary_tracing.provenance import model_config_hash  # noqa: E402
from latent_vocabulary_tracing.taxonomy import (  # noqa: E402
    TRACE_CATEGORIES,
    categorize_trace_token,
    is_displayable_trace_token,
)
from latent_vocabulary_tracing.torch_readout import (  # noqa: E402
    CELL_AA,
    CELL_AB,
    CELL_BB,
    FOUR_CELL_CONTRASTS,
    aggregate_category_statistics_by_spans,
    aggregate_position_metrics_by_spans,
    cell_distributions,
    four_cell_logits,
    four_cell_scalar_metrics,
    matched_faithfulness,
    net_token_direction_scores,
    pair_metrics,
    vocabulary_edit_statistics,
)

ap = argparse.ArgumentParser()
ap.add_argument("model_a")
ap.add_argument("model_b")
ap.add_argument("tag")
ap.add_argument("--model-a-id", default="", help="reported parent identity when loading locally")
ap.add_argument(
    "--model-b-id", default="", help="reported descendant identity when loading locally"
)
ap.add_argument("--lens", default=f"{ROOT}/repro/lenses/base_merged.pt")
ap.add_argument(
    "--lens-parent-id",
    default="",
    help="exact Hub repository used to fit --lens; required by confirmatory Jlens contracts",
)
ap.add_argument("--probes", default=f"{ZOO}/data/probes.jsonl")
ap.add_argument("--out", default=f"{ZOO}/results")
ap.add_argument("--kinds", default="math,code,agent,neutral")
ap.add_argument("--lmin", type=int, default=6)
ap.add_argument("--lmax", type=int, default=34)
ap.add_argument("--topk", type=int, default=10)
ap.add_argument("--max_len", type=int, default=640)
ap.add_argument(
    "--limit",
    type=int,
    default=0,
    help="process only the first N selected probes (smoke tests only)",
)
ap.add_argument(
    "--support_k",
    type=int,
    default=50,
    help="top-k per model used to form the stored union support",
)
ap.add_argument("--no_store", action="store_true", help="only write the summary json")
ap.add_argument(
    "--full_LL",
    action="store_true",
    help="store LL cross logits too (top-k, a_at_b/b_at_a) instead of the slim top-20 store",
)
ap.add_argument(
    "--store_fp32",
    action="store_true",
    help="retain floating support statistics in fp32 (required for confirmatory token analyses)",
)
ap.add_argument(
    "--category_stats",
    action="store_true",
    help="aggregate exact full-vocabulary trace-category statistics in fp32",
)
ap.add_argument(
    "--top_changes",
    type=int,
    default=20,
    help="exact full-vocabulary promoted/suppressed tokens retained per domain and layer",
)
ap.add_argument(
    "--no_J",
    action="store_true",
    help="no Jacobian lens for this architecture: logit lens + final + hidden only",
)
ap.add_argument(
    "--layers",
    default="",
    help="explicit layer list (e.g. 4,6,...,30); with Jlens, select a subset of its layers",
)
ap.add_argument(
    "--decoder",
    choices=("native", "parent"),
    default="native",
    help="decode each model natively (legacy) or decode both states with model A",
)
args = ap.parse_args()
model_a_id = args.model_a_id or args.model_a
model_b_id = args.model_b_id or args.model_b
dev = "cuda:0"
t0 = time.time()

if args.no_J:
    lens = {"n_prompts": 0}
    LAYERS = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(range(args.lmin, args.lmax + 1))
    )
    if not LAYERS:
        raise ValueError("no layers selected")
    J = None
    print(f"no J-lens; layers {LAYERS[0]}..{LAYERS[-1]} ({len(LAYERS)})", flush=True)
else:
    lens = torch.load(args.lens, map_location="cpu", weights_only=True)
    requested_layers = (
        {int(value) for value in args.layers.split(",")} if args.layers else None
    )
    LAYERS = [
        layer
        for layer in lens["source_layers"]
        if args.lmin <= layer <= args.lmax
        and (requested_layers is None or layer in requested_layers)
    ]
    if requested_layers is not None:
        missing_layers = requested_layers.difference(lens["source_layers"])
        if missing_layers:
            raise ValueError(
                f"requested layers absent from Jlens: {sorted(missing_layers)}"
            )
    if not LAYERS:
        raise ValueError("no Jlens layers remain after applying the layer/range selection")
    J = {layer: lens["J"][layer].to(dev, torch.float32) for layer in LAYERS}
    print(
        f"lens {args.lens} n_prompts={lens['n_prompts']} "
        f"layers {LAYERS[0]}..{LAYERS[-1]} ({len(LAYERS)})",
        flush=True,
    )
READOUTS = ("LL",) if args.no_J else ("J", "LL")


def load(name):
    m = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(dev).eval()
    return m


tok = transformers.AutoTokenizer.from_pretrained(args.model_a)
model_a, model_b = load(args.model_a), load(args.model_b)
tok_b = transformers.AutoTokenizer.from_pretrained(args.model_b)


def sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_state():
    revision = subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=normal"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "runner_hash": sha256_file(__file__),
    }


def text_config(model):
    getter = getattr(model.config, "get_text_config", None)
    return getter() if getter is not None else model.config


# Every model is read on the exact token ids emitted by the parent tokenizer.
# Compare id -> tokenizer piece for every id that actually occurs; comparing
# only shared piece -> id mappings misses parent-only ids reused by B.
all_probes = [json.loads(line) for line in open(args.probes)]
probe_ids = set()
for probe in all_probes:
    probe_ids.update(tok(probe["text"], truncation=True, max_length=args.max_len).input_ids)
piece_mismatches = []
for token_id in sorted(probe_ids):
    piece_a = tok.convert_ids_to_tokens(token_id)
    piece_b = tok_b.convert_ids_to_tokens(token_id)
    if piece_a != piece_b:
        piece_mismatches.append((token_id, piece_a, piece_b))
if piece_mismatches:
    raise AssertionError("tokenizer id->piece mismatch on probe ids: " + repr(piece_mismatches[:5]))

config_a, config_b = text_config(model_a), text_config(model_b)
assert config_a.vocab_size == config_b.vocab_size, "vocab_size mismatch"
assert config_a.hidden_size == config_b.hidden_size, "hidden_size mismatch"
va, vb = tok.get_vocab(), tok_b.get_vocab()
tok_note = {
    "checked_probe_token_ids": len(probe_ids),
    "id_piece_mismatches": 0,
    "shared_tokens": len(set(va) & set(vb)),
    "only_a_count": len(set(va) - set(vb)),
    "only_b_count": len(set(vb) - set(va)),
    "only_a_examples": sorted(set(va) - set(vb))[:20],
    "only_b_examples": sorted(set(vb) - set(va))[:20],
    "tokenizer_hash_a": sha256_json(va),
    "tokenizer_hash_b": sha256_json(vb),
}
print(f"models loaded ({time.time() - t0:.0f}s) tokenizer note: {tok_note}", flush=True)


def _parts(m):
    inner = getattr(m, "model", None) or getattr(m, "transformer", None)
    # VL / multimodal wrappers keep the text stack one level deeper
    for sub in ("language_model", "text_model"):
        if hasattr(inner, sub):
            inner = getattr(inner, sub)
            break
    norm = (
        getattr(inner, "norm", None)
        or getattr(inner, "final_layer_norm", None)
        or getattr(inner, "ln_f", None)
    )
    head = getattr(m, "lm_head", None) or getattr(m, "embed_out", None)
    layers = getattr(inner, "layers", None) or getattr(inner, "h", None)
    assert norm is not None and head is not None and layers is not None, (
        "cannot locate layers / final norm / lm_head"
    )
    return inner, layers, norm, head


def unembed(m, x):
    _, _, norm, head = _parts(m)
    return head(norm(x.to(head.weight.dtype))).float()


def forward_with_last_residual(model, input_ids):
    """Run a model and retain the last block output before its native norm."""

    _, layers, _, _ = _parts(model)
    captured = {}

    def save_last(_module, _inputs, output):
        captured["residual"] = output[0] if isinstance(output, tuple) else output

    handle = layers[-1].register_forward_hook(save_last)
    try:
        output = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()
    if "residual" not in captured:
        raise RuntimeError("last-layer hook did not capture a residual state")
    return output, captured["residual"]


def to_cpu_store(values):
    output = {}
    for key, value in values.items():
        value = value.detach().cpu()
        if value.dtype == torch.float32 and not args.store_fp32:
            value = value.half()
        output[key] = value
    return output


kinds = set(args.kinds.split(","))
probes = [probe for probe in all_probes if probe["kind"] in kinds]
if args.limit:
    probes = probes[: args.limit]
probe_splits = {}
for probe_kind in sorted(kinds):
    keys = [probe["key"] for probe in all_probes if probe["kind"] == probe_kind]
    split = deterministic_balanced_split(keys, salt=f"lvt-token-v1:{probe_kind}")
    probe_splits.update({f"{probe_kind}:{key}": value for key, value in split.items()})
for probe in probes:
    probe["inference_split"] = probe_splits[f"{probe['kind']}:{probe['key']}"]
print(f"{len(probes)} probes, kinds={sorted(kinds)}", flush=True)

category_ids = None
category_ids_cpu = None
displayable_ids = None
if args.category_stats:
    print(
        f"classifying {config_a.vocab_size} tokenizer pieces into "
        f"{len(TRACE_CATEGORIES)} trace categories",
        flush=True,
    )
    token_pieces = [tok.decode([token_id]) for token_id in range(config_a.vocab_size)]
    category_ids = torch.tensor(
        [TRACE_CATEGORIES.index(categorize_trace_token(piece)) for piece in token_pieces],
        dtype=torch.int64,
        device=dev,
    )
    displayable_ids = torch.tensor(
        [is_displayable_trace_token(piece) for piece in token_pieces], dtype=torch.bool
    )
    category_ids_cpu = category_ids.cpu()
    del token_pieces


def top_token_rows(values, signed_values, mask=None):
    ranked = values if mask is None else values.masked_fill(~mask, -torch.inf)
    top_values, top_ids = ranked.topk(min(args.top_changes, ranked.numel()))
    return [
        {
            "id": int(token_id),
            "token": tok.decode([int(token_id)]),
            "probability_points": 100.0 * float(value),
            "signed_probability_points": 100.0 * float(signed_values[token_id]),
        }
        for value, token_id in zip(top_values, top_ids, strict=True)
        if value > 0
    ]


def token_change_payload(changes, *, include_categories=False):
    signed = changes[0]
    promoted, suppressed = net_token_direction_scores(signed)
    payload = {
        "promoted": top_token_rows(promoted, signed),
        "suppressed": top_token_rows(suppressed, signed),
        "displayable_promoted": top_token_rows(promoted, signed, displayable_ids),
        "displayable_suppressed": top_token_rows(suppressed, signed, displayable_ids),
    }
    if include_categories:
        payload["displayable_by_category"] = {
            category: {
                "promoted": top_token_rows(
                    promoted,
                    signed,
                    displayable_ids & (category_ids_cpu == category_index),
                ),
                "suppressed": top_token_rows(
                    suppressed,
                    signed,
                    displayable_ids & (category_ids_cpu == category_index),
                ),
            }
            for category_index, category in enumerate(TRACE_CATEGORIES)
        }
    return payload


def heldout_token_payload(domain, readout, layer):
    discovery_key = (domain, readout, layer, "discovery")
    confirmation_key = (domain, readout, layer, "confirmation")
    n_discovery = token_split_counts.get(discovery_key, 0)
    n_confirmation = token_split_counts.get(confirmation_key, 0)
    payload = {
        "status": "complete" if n_discovery and n_confirmation else "insufficient_split",
        "discovery_probes": n_discovery,
        "confirmation_probes": n_confirmation,
        "test": "one_sided_exact_sign_test",
        "multiplicity": "benjamini_hochberg_across_displayed_candidates",
        "q_threshold": 0.05,
        "displayable_by_category": {},
    }
    if not n_discovery or not n_confirmation:
        return payload

    discovery_mean = token_split_sums[discovery_key] / n_discovery
    confirmation_sum = token_split_sums[confirmation_key]
    confirmation_mean = confirmation_sum / n_confirmation
    confirmation_square_sum = token_split_square_sums[confirmation_key]
    positive_counts = token_split_positive_counts[confirmation_key]
    negative_counts = token_split_negative_counts[confirmation_key]
    candidates = []
    for category_index, category in enumerate(TRACE_CATEGORIES):
        mask = displayable_ids & (category_ids_cpu == category_index)
        payload["displayable_by_category"][category] = {
            "promoted": [],
            "suppressed": [],
        }
        for label, direction in (("promoted", 1), ("suppressed", -1)):
            scores = (direction * discovery_mean).masked_fill(~mask, -torch.inf)
            top_values, top_ids = scores.topk(min(args.top_changes, scores.numel()))
            for rank, (score, token_id) in enumerate(
                zip(top_values, top_ids, strict=True), start=1
            ):
                if score <= 0:
                    continue
                token_id = int(token_id)
                positive = int(positive_counts[token_id])
                negative = int(negative_counts[token_id])
                alternative = "positive" if direction > 0 else "negative"
                confirmation_value = float(confirmation_mean[token_id])
                if n_confirmation > 1:
                    centered_sum = max(
                        0.0,
                        float(confirmation_square_sum[token_id])
                        - float(confirmation_sum[token_id]) ** 2 / n_confirmation,
                    )
                    standard_error = (centered_sum / (n_confirmation - 1) / n_confirmation) ** 0.5
                else:
                    standard_error = None
                row = {
                    "id": token_id,
                    "token": tok.decode([token_id]),
                    "discovery_rank": rank,
                    "discovery_probability_points": 100.0 * float(discovery_mean[token_id]),
                    "confirmation_probability_points": 100.0 * confirmation_value,
                    "confirmation_standard_error_points": (
                        100.0 * standard_error if standard_error is not None else None
                    ),
                    "confirmation_positive_probes": positive,
                    "confirmation_negative_probes": negative,
                    "confirmation_zero_probes": n_confirmation - positive - negative,
                    "one_sided_sign_p": one_sided_sign_test(
                        positive,
                        negative,
                        alternative=alternative,
                    ),
                    "direction": label,
                }
                payload["displayable_by_category"][category][label].append(row)
                candidates.append(row)
    q_values = benjamini_hochberg(
        [candidate["one_sided_sign_p"] for candidate in candidates]
    )
    for candidate, q_value in zip(candidates, q_values, strict=True):
        candidate["bh_q"] = float(q_value)
        expected_sign = 1 if candidate["direction"] == "promoted" else -1
        candidate["confirmed"] = bool(
            q_value <= payload["q_threshold"]
            and expected_sign * candidate["confirmation_probability_points"] > 0
        )
    payload["tested_candidates"] = len(candidates)
    payload["confirmed_candidates"] = sum(
        candidate["confirmed"] for candidate in candidates
    )
    return payload


def summarize_position_metrics(metrics, span, n_pos, prompt_len):
    output = {key: float(metrics[key].mean()) for key in ("kl_ab", "kl_ba", "js")}
    if "jaccard" in metrics:
        output["jaccard"] = float(metrics["jaccard"].mean())
    for key in (
        "kl_ab",
        "kl_ba",
        "js",
        "support_mass_a",
        "support_mass_b",
        "outside_mass_a",
        "outside_mass_b",
    ):
        if key not in metrics:
            continue
        output[key] = float(metrics[key].mean())
        output[f"{key}_resp"] = float(metrics[key][span].mean()) if n_pos > prompt_len else None
    return output


def summarize_faithfulness(faith, span, n_pos, prompt_len):
    output = {}
    for coordinate, values in faith.items():
        output[coordinate] = {}
        for model_key, per_position in values.items():
            output[coordinate][model_key] = float(per_position.mean())
            output[coordinate][f"{model_key}_resp"] = (
                float(per_position[span].mean()) if n_pos > prompt_len else None
            )
    return output


records, store = [], {}
token_change_sums = {}
token_change_counts = {}
token_split_sums = {}
token_split_square_sums = {}
token_split_positive_counts = {}
token_split_negative_counts = {}
token_split_counts = {}
with torch.no_grad():
    for i, p in enumerate(probes):
        ids = tok(
            p["text"], return_tensors="pt", truncation=True, max_length=args.max_len
        ).input_ids
        ids = ids.to(dev)
        oa, last_a = forward_with_last_residual(model_a, ids)
        ob, last_b = forward_with_last_residual(model_b, ids)
        fa, fb = oa.logits[0].float(), ob.logits[0].float()
        final_cells = four_cell_logits(
            last_a[0].float(),
            last_b[0].float(),
            lambda state: unembed(model_a, state),
            lambda state: unembed(model_b, state),
        )
        # Preserve the models' exact native logits on the diagonal.  The
        # hooked residual is used only to construct the two cross-decoder cells.
        final_cells[CELL_AA] = fa
        final_cells[CELL_BB] = fb
        final_distributions = cell_distributions(final_cells)
        final_four_cell_metrics = four_cell_scalar_metrics(
            final_cells, distributions=final_distributions
        )
        # The AB cell is needed for the final four-cell control but never for
        # matched native or parent-anchored faithfulness.  Release it before
        # iterating over layers; this matters for 248k-vocabulary models.
        del final_cells[CELL_AB], final_distributions[CELL_AB]
        n_pos = ids.shape[1]
        rec = {
            "key": p["key"],
            "kind": p["kind"],
            "inference_split": p["inference_split"],
            "prompt_len": p["prompt_len"],
            "meta": p["meta"],
            "n_pos": n_pos,
            "per_layer": {"J": {}, "LL": {}},
            "hidden": {},
            "four_cell": {"J": {}, "LL": {}},
            "faith": {"J": {}, "LL": {}},
            "faith_native": {"J": {}, "LL": {}},
            "faith_anchored": {"J": {}, "LL": {}},
            "categories": {"J": {}, "LL": {}},
            "role_categories": {"J": {}, "LL": {}},
            "role_metrics": {"J": {}, "LL": {}},
        }
        st = {
            "n_pos": n_pos,
            "input_ids": ids[0].cpu().int(),
            "J": {},
            "LL": {},
            "store_offset": p["prompt_len"],
        }
        span = slice(
            p["prompt_len"], n_pos
        )  # response span (the per-position store keeps ONLY this span)
        for layer in LAYERS:
            ha = oa.hidden_states[layer + 1][0].float()
            hb = ob.hidden_states[layer + 1][0].float()
            # hidden stats
            cos = float(F.cosine_similarity(ha, hb, dim=-1).mean())
            Xc, Yc = ha - ha.mean(0, keepdim=True), hb - hb.mean(0, keepdim=True)
            K, L2 = (Xc @ Xc.T).double(), (Yc @ Yc.T).double()
            cka = float((K * L2).sum() / (K.norm() * L2.norm() + 1e-12))
            dnr = float((hb - ha).norm(dim=-1).mean() / (ha.norm(dim=-1).mean() + 1e-12))
            rec["hidden"][str(layer)] = {"cos": cos, "cka": cka, "dnorm_rel": dnr}
            for kind, xa, xb in (
                (("J", ha @ J[layer].T, hb @ J[layer].T),) if J is not None else ()
            ) + (("LL", ha, hb),):
                cells = four_cell_logits(
                    xa,
                    xb,
                    lambda state: unembed(model_a, state),
                    lambda state: unembed(model_b, state),
                )
                primary_name = (
                    "state_parent_decoder" if args.decoder == "parent" else "native_total"
                )
                primary_a, primary_b = FOUR_CELL_CONTRASTS[primary_name]
                distributions = cell_distributions(cells)
                m = pair_metrics(
                    cells[primary_a],
                    cells[primary_b],
                    jaccard_k=args.topk,
                    support_k=args.support_k,
                    parent_distribution=distributions[primary_a],
                    descendant_distribution=distributions[primary_b],
                )
                rec["per_layer"][kind][str(layer)] = summarize_position_metrics(
                    m, span, n_pos, p["prompt_len"]
                )
                role_metrics = aggregate_position_metrics_by_spans(
                    m,
                    p.get("role_spans", {}),
                    fields=(
                        "kl_ab",
                        "kl_ba",
                        "js",
                        "jaccard",
                        "support_mass_a",
                        "support_mass_b",
                        "outside_mass_a",
                        "outside_mass_b",
                    ),
                )
                rec["role_metrics"][kind][str(layer)] = {
                    role: {metric: float(value) for metric, value in values.items()}
                    for role, values in role_metrics.items()
                }

                four_cell = four_cell_scalar_metrics(cells, distributions=distributions)
                rec["four_cell"][kind][str(layer)] = {
                    name: summarize_position_metrics(values, span, n_pos, p["prompt_len"])
                    for name, values in four_cell.items()
                }

                faith = summarize_faithfulness(
                    matched_faithfulness(
                        final_cells,
                        cells,
                        final_distributions=final_distributions,
                        readout_distributions=distributions,
                    ),
                    span,
                    n_pos,
                    p["prompt_len"],
                )
                rec["faith_native"][kind][str(layer)] = faith["native"]
                rec["faith_anchored"][kind][str(layer)] = faith["parent_anchored"]
                rec["faith"][kind][str(layer)] = (
                    faith["parent_anchored"] if args.decoder == "parent" else faith["native"]
                )

                if category_ids is not None:
                    category_values, probability_delta = vocabulary_edit_statistics(
                        cells[primary_a],
                        cells[primary_b],
                        category_ids,
                        n_categories=len(TRACE_CATEGORIES),
                        parent_probability=distributions[primary_a][1],
                        descendant_probability=distributions[primary_b][1],
                    )
                    if n_pos > p["prompt_len"]:
                        rec["categories"][kind][str(layer)] = {
                            metric: values[span].mean(dim=0).cpu().tolist()
                            for metric, values in category_values.items()
                        }
                        delta_response = probability_delta[span]
                        change_key = (p["kind"], kind, layer)
                        probe_changes = torch.stack(
                            (
                                delta_response.mean(dim=0),
                                delta_response.clamp_min(0).mean(dim=0),
                                (-delta_response).clamp_min(0).mean(dim=0),
                            )
                        ).cpu()
                        if change_key not in token_change_sums:
                            token_change_sums[change_key] = probe_changes
                            token_change_counts[change_key] = 1
                        else:
                            token_change_sums[change_key] += probe_changes
                            token_change_counts[change_key] += 1
                        split_key = (
                            p["kind"],
                            kind,
                            layer,
                            p["inference_split"],
                        )
                        signed_probe_change = probe_changes[0]
                        if split_key not in token_split_sums:
                            token_split_sums[split_key] = signed_probe_change.clone()
                            token_split_square_sums[split_key] = signed_probe_change.square()
                            token_split_positive_counts[split_key] = (
                                signed_probe_change > 0
                            ).to(torch.int16)
                            token_split_negative_counts[split_key] = (
                                signed_probe_change < 0
                            ).to(torch.int16)
                            token_split_counts[split_key] = 1
                        else:
                            token_split_sums[split_key] += signed_probe_change
                            token_split_square_sums[split_key] += signed_probe_change.square()
                            token_split_positive_counts[split_key] += (
                                signed_probe_change > 0
                            ).to(torch.int16)
                            token_split_negative_counts[split_key] += (
                                signed_probe_change < 0
                            ).to(torch.int16)
                            token_split_counts[split_key] += 1
                    rec["role_categories"][kind][str(layer)] = {}
                    role_values = aggregate_category_statistics_by_spans(
                        category_values, p.get("role_spans", {})
                    )
                    rec["role_categories"][kind][str(layer)] = {
                        role: {
                            metric: values.cpu().tolist() for metric, values in metrics.items()
                        }
                        for role, metrics in role_values.items()
                    }
                    del probability_delta, category_values
                if not args.no_store:
                    if (
                        kind == "LL" and not args.full_LL
                    ):  # slim store: top-20, no cross logits (disk budget)
                        slim_support_fields = {
                            "top_a",
                            "top_b",
                            "val_a",
                            "val_b",
                            "delta_logp_at_a",
                            "delta_logp_at_b",
                            "b_is_new",
                        }
                        m = {
                            k: (v[:, :20] if k in slim_support_fields else v)
                            for k, v in m.items()
                            if k not in ("a_at_b", "b_at_a")
                        }
                    st[kind][layer] = to_cpu_store({k: v[span] for k, v in m.items()})
                del cells, distributions, four_cell, m
        # final logits
        m = pair_metrics(
            fa,
            fb,
            jaccard_k=args.topk,
            support_k=args.support_k,
            parent_distribution=final_distributions[CELL_AA],
            descendant_distribution=final_distributions[CELL_BB],
        )
        rec["final"] = summarize_position_metrics(m, span, n_pos, p["prompt_len"])
        rec["final_role_metrics"] = {
            role: {metric: float(value) for metric, value in values.items()}
            for role, values in aggregate_position_metrics_by_spans(
                m,
                p.get("role_spans", {}),
                fields=("kl_ab", "kl_ba", "js", "jaccard"),
            ).items()
        }
        rec["final_four_cell"] = {
            name: summarize_position_metrics(values, span, n_pos, p["prompt_len"])
            for name, values in final_four_cell_metrics.items()
        }
        if not args.no_store:
            st["final"] = to_cpu_store({k: v[span] for k, v in m.items()})
        # Per-token log probability of the actual next token on the response span.
        tgt = ids[0, 1:]
        nll_a = -F.log_softmax(fa[:-1], -1).gather(1, tgt[:, None])[:, 0]
        nll_b = -F.log_softmax(fb[:-1], -1).gather(1, tgt[:, None])[:, 0]
        rs = slice(max(p["prompt_len"] - 1, 0), n_pos - 1)
        rec["nll"] = {"a": float(nll_a[rs].mean()), "b": float(nll_b[rs].mean())}
        st["nll_a"], st["nll_b"] = nll_a.cpu().half(), nll_b.cpu().half()
        records.append(rec)
        store[p["key"]] = st
        del (
            oa,
            ob,
            fa,
            fb,
            final_cells,
            final_distributions,
            final_four_cell_metrics,
            last_a,
            last_b,
            m,
        )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(probes)} ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(args.out, exist_ok=True)
# aggregate summary
agg = {}
for kind in sorted(kinds):
    rs = [r for r in records if r["kind"] == kind]
    if not rs:
        continue
    a = {
        "n": len(rs),
        "J": {},
        "LL": {},
        "hidden": {},
        "four_cell": {"J": {}, "LL": {}},
        "faith": {"J": {}, "LL": {}},
        "faith_native": {"J": {}, "LL": {}},
        "faith_anchored": {"J": {}, "LL": {}},
        "categories": {"J": {}, "LL": {}},
        "role_categories": {"J": {}, "LL": {}},
        "top_token_changes": {"J": {}, "LL": {}},
        "role_metrics": {"J": {}, "LL": {}},
    }
    for layer in LAYERS:
        for ro in READOUTS:
            scalar_fields = (
                "kl_ab",
                "kl_ba",
                "js",
                "jaccard",
                "support_mass_a",
                "support_mass_b",
                "outside_mass_a",
                "outside_mass_b",
            )
            a[ro][str(layer)] = {
                field: sum(r["per_layer"][ro][str(layer)][field] for r in rs) / len(rs)
                for field in scalar_fields
                if field in rs[0]["per_layer"][ro][str(layer)]
            }
            for field in scalar_fields:
                key = f"{field}_resp"
                values = [
                    r["per_layer"][ro][str(layer)].get(key)
                    for r in rs
                    if r["per_layer"][ro][str(layer)].get(key) is not None
                ]
                if values:
                    a[ro][str(layer)][key] = sum(values) / len(values)

            a["four_cell"][ro][str(layer)] = {}
            for contrast in FOUR_CELL_CONTRASTS:
                a["four_cell"][ro][str(layer)][contrast] = {}
                for field in ("kl_ab", "kl_ba", "js", "kl_ab_resp", "kl_ba_resp", "js_resp"):
                    values = [
                        r["four_cell"][ro][str(layer)][contrast].get(field)
                        for r in rs
                        if r["four_cell"][ro][str(layer)][contrast].get(field) is not None
                    ]
                    if values:
                        a["four_cell"][ro][str(layer)][contrast][field] = sum(values) / len(values)

            role_names = sorted(
                {
                    role
                    for r in rs
                    for role in r["role_metrics"][ro].get(str(layer), {})
                }
            )
            a["role_metrics"][ro][str(layer)] = {}
            for role in role_names:
                role_rows = [
                    r["role_metrics"][ro][str(layer)][role]
                    for r in rs
                    if role in r["role_metrics"][ro].get(str(layer), {})
                ]
                a["role_metrics"][ro][str(layer)][role] = {
                    metric: sum(row[metric] for row in role_rows) / len(role_rows)
                    for metric in role_rows[0]
                }

            for faith_key in ("faith_native", "faith_anchored"):
                a[faith_key][ro][str(layer)] = {}
                for field in ("a", "b", "a_resp", "b_resp"):
                    values = [
                        r[faith_key][ro][str(layer)].get(field)
                        for r in rs
                        if r[faith_key][ro][str(layer)].get(field) is not None
                    ]
                    if values:
                        a[faith_key][ro][str(layer)][field] = sum(values) / len(values)
            selected_faith = "faith_anchored" if args.decoder == "parent" else "faith_native"
            a["faith"][ro][str(layer)] = a[selected_faith][ro][str(layer)]

            if args.category_stats:
                category_rows = [
                    r["categories"][ro][str(layer)] for r in rs if str(layer) in r["categories"][ro]
                ]
                if category_rows:
                    a["categories"][ro][str(layer)] = {
                        metric: [
                            sum(row[metric][index] for row in category_rows) / len(category_rows)
                            for index in range(len(TRACE_CATEGORIES))
                        ]
                        for metric in category_rows[0]
                    }
                role_names = sorted(
                    {
                        role
                        for r in rs
                        for role in r["role_categories"][ro].get(str(layer), {})
                    }
                )
                a["role_categories"][ro][str(layer)] = {}
                for role in role_names:
                    role_rows = [
                        r["role_categories"][ro][str(layer)][role]
                        for r in rs
                        if role in r["role_categories"][ro].get(str(layer), {})
                    ]
                    a["role_categories"][ro][str(layer)][role] = {
                        metric: [
                            sum(row[metric][index] for row in role_rows) / len(role_rows)
                            for index in range(len(TRACE_CATEGORIES))
                        ]
                        for metric in role_rows[0]
                    }
                change_key = (kind, ro, layer)
                if change_key in token_change_sums:
                    changes = (
                        token_change_sums[change_key] / token_change_counts[change_key]
                    )
                    a["top_token_changes"][ro][str(layer)] = token_change_payload(changes)
        a["hidden"][str(layer)] = {
            k: sum(r["hidden"][str(layer)][k] for r in rs) / len(rs)
            for k in ("cos", "cka", "dnorm_rel")
        }
    if args.category_stats:
        n_model_layers = len(_parts(model_a)[1])
        summary_layers = [
            layer for layer in LAYERS if 0.50 <= (layer + 1) / n_model_layers <= 0.85
        ]
        for ro in READOUTS:
            changes = [
                token_change_sums[(kind, ro, layer)]
                / token_change_counts[(kind, ro, layer)]
                for layer in summary_layers
                if (kind, ro, layer) in token_change_sums
            ]
            if changes:
                depth_average = torch.stack(changes).mean(dim=0)
                a["top_token_changes"][ro]["depth_50_85"] = token_change_payload(
                    depth_average, include_categories=True
                )
            eligible_layers = [
                layer
                for layer in summary_layers
                if (kind, ro, layer) in token_change_sums
                and any(
                    r["inference_split"] == "discovery"
                    and r["per_layer"][ro][str(layer)]["kl_ba_resp"] is not None
                    for r in rs
                )
            ]
            if eligible_layers:
                discovery_response_kl = {
                    layer: sum(
                        r["per_layer"][ro][str(layer)]["kl_ba_resp"]
                        for r in rs
                        if r["inference_split"] == "discovery"
                    )
                    / sum(r["inference_split"] == "discovery" for r in rs)
                    for layer in eligible_layers
                }
                peak_layer = max(
                    eligible_layers,
                    key=discovery_response_kl.__getitem__,
                )
                peak_changes = (
                    token_change_sums[(kind, ro, peak_layer)]
                    / token_change_counts[(kind, ro, peak_layer)]
                )
                peak_payload = token_change_payload(
                    peak_changes, include_categories=True
                )
                peak_payload.update(
                    {
                        "layer": peak_layer,
                        "normalized_depth": (peak_layer + 1) / n_model_layers,
                        "response_kl_ba": a[ro][str(peak_layer)]["kl_ba_resp"],
                        "discovery_response_kl_ba": discovery_response_kl[peak_layer],
                        "heldout_inference": heldout_token_payload(kind, ro, peak_layer),
                    }
                )
                a["top_token_changes"][ro][
                    "peak_response_kl_50_85"
                ] = peak_payload
    a["final"] = {
        k: sum(r["final"][k] for r in rs) / len(rs) for k in ("kl_ab", "kl_ba", "js", "jaccard")
    }
    for metric in ("kl_ab", "kl_ba", "js"):
        key = f"{metric}_resp"
        vals = [r["final"][key] for r in rs if r["final"][key] is not None]
        a["final"][key] = sum(vals) / len(vals) if vals else None
    a["final_four_cell"] = {}
    for contrast in FOUR_CELL_CONTRASTS:
        a["final_four_cell"][contrast] = {}
        for field in ("kl_ab", "kl_ba", "js", "kl_ab_resp", "kl_ba_resp", "js_resp"):
            values = [
                r["final_four_cell"][contrast].get(field)
                for r in rs
                if r["final_four_cell"][contrast].get(field) is not None
            ]
            if values:
                a["final_four_cell"][contrast][field] = sum(values) / len(values)
    final_roles = sorted({role for r in rs for role in r["final_role_metrics"]})
    a["final_role_metrics"] = {}
    for role in final_roles:
        role_rows = [r["final_role_metrics"][role] for r in rs if role in r["final_role_metrics"]]
        a["final_role_metrics"][role] = {
            metric: sum(row[metric] for row in role_rows) / len(role_rows)
            for metric in role_rows[0]
        }
    a["nll"] = {x: sum(r["nll"][x] for r in rs) / len(rs) for x in ("a", "b")}
    agg[kind] = a


def checkpoint_revision(name, model):
    revision = getattr(model.config, "_commit_hash", None)
    if revision:
        return revision
    marker = f"{os.sep}snapshots{os.sep}"
    if marker in name:
        return name.split(marker, 1)[1].split(os.sep, 1)[0]
    return None


def model_metadata(identity, load_source, model):
    config = text_config(model)
    _, layers, norm, _ = _parts(model)
    rope = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None)
    return {
        "id": identity,
        "revision": checkpoint_revision(load_source, model),
        "architecture": type(model).__name__,
        "hidden_size": int(config.hidden_size),
        "n_layers": len(layers),
        "vocab_size": int(config.vocab_size),
        "norm_type": type(norm).__name__,
        "rope": rope,
        "config_hash": model_config_hash(config.to_dict()),
    }


summary = {
    "schema_version": 2,
    "model_a": model_a_id,
    "model_b": model_b_id,
    "models": {
        "a": model_metadata(model_a_id, args.model_a, model_a),
        "b": model_metadata(model_b_id, args.model_b, model_b),
    },
    "tag": args.tag,
    "lens": args.lens if J is not None else None,
    "lens_parent_id": args.lens_parent_id or None,
    "decoder_mode": "parent_anchored" if args.decoder == "parent" else "native_per_model",
    "primary_contrast": ("state_parent_decoder" if args.decoder == "parent" else "native_total"),
    "four_cell_contrasts": FOUR_CELL_CONTRASTS,
    "final_decoder_mode": "native_per_model_control",
    "faithfulness_coordinates": {
        "native": "final and layer readout both use each state's native decoder",
        "parent_anchored": "final and layer readout both use decoder A",
    },
    "lens_n_prompts": int(lens["n_prompts"]),
    "lens_hash": sha256_file(args.lens) if not args.no_J else None,
    "lens_source_layers": list(lens.get("source_layers", [])),
    "lens_d_model": lens.get("d_model"),
    "readouts": list(READOUTS),
    "layers": LAYERS,
    "probes": args.probes,
    "probe_hash": sha256_file(args.probes),
    "probe_inference_split": {
        "method": "sha256_rank_balanced_within_domain",
        "salt": "lvt-token-v1:<domain>",
        "mapping_hash": sha256_json(probe_splits),
        "counts": {
            probe_kind: {
                split: sum(
                    probe["kind"] == probe_kind and probe["inference_split"] == split
                    for probe in probes
                )
                for split in ("discovery", "confirmation")
            }
            for probe_kind in sorted(kinds)
        },
    },
    "analysis_provenance": repository_state(),
    "support_k": args.support_k,
    "analysis_dtype": "fp32",
    "stored_support_dtype": "none" if args.no_store else (
        "fp32" if args.store_fp32 else "fp16"
    ),
    "category_statistics": {
        "enabled": args.category_stats,
        "taxonomy": list(TRACE_CATEGORIES) if args.category_stats else None,
        "taxonomy_hash": (
            sha256_file(taxonomy_module.__file__) if args.category_stats else None
        ),
        "dtype": "fp32" if args.category_stats else None,
        "support": "full_vocabulary" if args.category_stats else None,
        "role_conditioned": args.category_stats,
        "averaging": "positions_within_probe_then_probes",
        "top_changes": args.top_changes if args.category_stats else None,
        "top_change_support": "full_vocabulary" if args.category_stats else None,
        "top_change_ranking": (
            "net_probability_delta_after_averaging" if args.category_stats else None
        ),
        "top_change_values": "percentage_points" if args.category_stats else None,
        "token_inference": (
            "discovery_selection_then_heldout_sign_test_bh"
            if args.category_stats
            else None
        ),
        "depth_summary": (
            {"minimum": 0.50, "maximum": 0.85} if args.category_stats else None
        ),
        "representative_layer_rule": (
            "maximum_discovery_response_kl_ba_within_depth_summary"
            if args.category_stats
            else None
        ),
    },
    "direction_statistics": {
        "quantity": "delta_logp",
        "dtype": "fp32" if args.store_fp32 and not args.no_store else None,
        "support": "union_parent_descendant_top_k" if not args.no_store else None,
        "support_coverage_reported": not args.no_store,
    },
    "seconds": time.time() - t0,
    "tokenizer_note": tok_note,
    "agg": agg,
    "records": records,
}
summary_path = os.path.join(args.out, f"ro_{args.tag}_summary.json")
summary_temporary = f"{summary_path}.tmp.{os.getpid()}"
with open(summary_temporary, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=1, ensure_ascii=False)
    handle.write("\n")
os.replace(summary_temporary, summary_path)
if not args.no_store:
    store_path = os.path.join(args.out, f"ro_{args.tag}.pt")
    store_temporary = f"{store_path}.tmp.{os.getpid()}"
    torch.save(
        {
            "records": records,
            "store": store,
            "layers": LAYERS,
            "model_a": model_a_id,
            "model_b": model_b_id,
            "lens": args.lens if J is not None else None,
            "decoder_mode": summary["decoder_mode"],
            "schema_version": summary["schema_version"],
            "primary_contrast": summary["primary_contrast"],
            "probe_hash": summary["probe_hash"],
            "support_k": args.support_k,
            "stored_support_dtype": summary["stored_support_dtype"],
        },
        store_temporary,
    )
    os.replace(store_temporary, store_path)


# one-line digest
def dig(kind):
    a = agg.get(kind)
    if not a:
        return ""
    ro = "J" if "J" in READOUTS else "LL"
    n_model_layers = len(_parts(model_a)[1])
    selected_layers = [
        layer for layer in LAYERS if 0.50 <= (layer + 1) / n_model_layers <= 0.85
    ] or [LAYERS[len(LAYERS) // 2]]
    response_kl = [
        a[ro][str(layer)]["kl_ba_resp"]
        for layer in selected_layers
        if a[ro][str(layer)]["kl_ba_resp"] is not None
    ]
    middle = str(22 if 22 in LAYERS else LAYERS[len(LAYERS) // 2])
    coordinate = "parent-anchored" if args.decoder == "parent" else "native"
    write_amount = sum(response_kl) / len(response_kl) if response_kl else float("nan")
    return (
        f"{kind}: {ro} mean response KL(B||A), {coordinate}, "
        f"depth 0.50-0.85={write_amount:.4f}; "
        f"native-final response KL(B||A)={a['final']['kl_ba_resp']:.4f}; L{middle} "
        f"cos={a['hidden'][middle]['cos']:.3f} "
        f"nll a/b={a['nll']['a']:.3f}/{a['nll']['b']:.3f}"
    )


print(
    f"DONE {args.tag} in {time.time() - t0:.0f}s | " + " | ".join(dig(k) for k in sorted(kinds)),
    flush=True,
)
