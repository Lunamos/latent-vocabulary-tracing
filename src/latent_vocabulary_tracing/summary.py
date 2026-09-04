"""Small, safe views over trace-summary JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SummaryContract:
    """Machine-checkable requirements for a confirmatory result.

    Plotting code should construct one of these explicitly instead of choosing
    a readout or decoder mode from whichever fields happen to be present.  The
    default contract is the paper's primary parent-anchored response-token KL.
    Native final logits remain a separately labelled control; they are not a
    continuation of the anchored layer curve.
    """

    readout: str
    domains: tuple[str, ...] = ("math", "code", "agent", "neutral")
    decoder_mode: str = "parent_anchored"
    primary_contrast: str = "state_parent_decoder"
    metric: str = "kl_ba_resp"
    minimum_schema_version: int = 2
    require_four_cell: bool = True
    require_role_metrics: bool = True
    require_provenance: bool = True
    require_category_statistics: bool = False
    require_fp32_store: bool = False

    def __post_init__(self) -> None:
        if self.readout not in {"J", "LL"}:
            raise ValueError("readout must be 'J' (Jlens) or 'LL'")
        if not self.domains:
            raise ValueError("domains cannot be empty")


def load_summary(
    path: str | Path, *, contract: SummaryContract | None = None
) -> dict[str, Any]:
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"tag", "model_a", "model_b", "layers"}
    missing = required.difference(summary)
    if missing:
        raise ValueError(f"summary is missing required keys: {sorted(missing)}")
    if contract is not None:
        validate_summary_contract(summary, contract)
    return summary


def validate_summary_contract(
    summary: dict[str, Any], contract: SummaryContract
) -> None:
    """Reject results that cannot support a declared confirmatory analysis."""

    errors: list[str] = []
    schema_version = summary.get("schema_version", 0)
    if not isinstance(schema_version, int) or schema_version < contract.minimum_schema_version:
        errors.append(
            f"schema_version={schema_version!r}, need >= {contract.minimum_schema_version}"
        )
    for field, expected in (
        ("decoder_mode", contract.decoder_mode),
        ("primary_contrast", contract.primary_contrast),
    ):
        if summary.get(field) != expected:
            errors.append(f"{field}={summary.get(field)!r}, need {expected!r}")

    readouts = summary.get("readouts", [])
    if contract.readout not in readouts:
        errors.append(f"readout {contract.readout!r} absent from {readouts!r}")

    aggregate = summary.get("agg")
    layers = summary.get("layers")
    if not isinstance(aggregate, dict):
        errors.append("agg is missing or is not an object")
    if not isinstance(layers, list) or not layers:
        errors.append("layers is missing or empty")
    if isinstance(aggregate, dict) and isinstance(layers, list):
        for domain in contract.domains:
            domain_result = aggregate.get(domain)
            if not isinstance(domain_result, dict):
                errors.append(f"domain {domain!r} is absent")
                continue
            readout_result = domain_result.get(contract.readout)
            if not isinstance(readout_result, dict):
                errors.append(f"domain {domain!r} lacks readout {contract.readout!r}")
                continue
            for layer in layers:
                layer_result = readout_result.get(str(layer))
                if not isinstance(layer_result, dict) or contract.metric not in layer_result:
                    errors.append(
                        f"domain {domain!r}, {contract.readout} L{layer} lacks "
                        f"metric {contract.metric!r}"
                    )
                    break
            if contract.require_four_cell:
                four_cell = domain_result.get("four_cell", {}).get(contract.readout, {})
                for layer in layers:
                    contrast = four_cell.get(str(layer), {}).get(contract.primary_contrast, {})
                    if contract.metric not in contrast:
                        errors.append(
                            f"domain {domain!r}, {contract.readout} L{layer} lacks four-cell "
                            f"{contract.primary_contrast}/{contract.metric}"
                        )
                        break
            if contract.require_role_metrics:
                role_layers = domain_result.get("role_metrics", {}).get(contract.readout, {})
                role_metric = contract.metric.removesuffix("_resp")
                for layer in layers:
                    role_result = role_layers.get(str(layer), {})
                    if not role_result or not any(
                        role_metric in values for values in role_result.values()
                    ):
                        errors.append(
                            f"domain {domain!r}, {contract.readout} L{layer} lacks "
                            f"role-conditioned {role_metric!r}"
                        )
                        break

    if summary.get("final_decoder_mode") != "native_per_model_control":
        errors.append("final logits are not labelled as a native-per-model control")

    if contract.require_category_statistics:
        category = summary.get("category_statistics", {})
        if not category.get("enabled"):
            errors.append("full-vocabulary category statistics are disabled")
        if category.get("dtype") != "fp32":
            errors.append(f"category dtype={category.get('dtype')!r}, need 'fp32'")
        if category.get("support") != "full_vocabulary":
            errors.append(
                f"category support={category.get('support')!r}, need 'full_vocabulary'"
            )
        if not category.get("role_conditioned"):
            errors.append("category statistics are not role-conditioned")
        if category.get("top_change_support") != "full_vocabulary":
            errors.append("exact token changes do not use full-vocabulary support")
        if category.get("top_change_ranking") != "net_probability_delta_after_averaging":
            errors.append("exact token changes are not ranked by net probability change")
    if contract.require_fp32_store:
        stored_dtype = summary.get("stored_support_dtype", summary.get("support_dtype"))
        if stored_dtype != "fp32":
            errors.append(f"stored support dtype={stored_dtype!r}, need 'fp32'")
        direction = summary.get("direction_statistics", {})
        if direction.get("quantity") != "delta_logp":
            errors.append("direction store does not contain direct delta_logp")
        if direction.get("dtype") != "fp32":
            errors.append(f"direction dtype={direction.get('dtype')!r}, need 'fp32'")
        if direction.get("support") != "union_parent_descendant_top_k":
            errors.append("direction support is not the parent/descendant top-k union")
        if not direction.get("support_coverage_reported"):
            errors.append("direction-support coverage is not reported")

    if contract.require_provenance:
        models = summary.get("models", {})
        for model_key in ("a", "b"):
            model = models.get(model_key, {})
            for field in (
                "id",
                "revision",
                "architecture",
                "hidden_size",
                "n_layers",
                "vocab_size",
                "norm_type",
                "config_hash",
            ):
                if model.get(field) in (None, ""):
                    errors.append(f"model {model_key!r} lacks provenance field {field!r}")
        if not summary.get("probe_hash"):
            errors.append("probe_hash is absent")
        tokenizer = summary.get("tokenizer_note", {})
        if tokenizer.get("id_piece_mismatches") != 0:
            errors.append("probe token ids were not verified identical across tokenizers")
        for field in ("tokenizer_hash_a", "tokenizer_hash_b"):
            if not tokenizer.get(field):
                errors.append(f"tokenizer provenance field {field!r} is absent")
        if contract.readout == "J":
            if not summary.get("lens_hash"):
                errors.append("Jlens hash is absent")
            lens_prompts = summary.get("lens_n_prompts")
            if not isinstance(lens_prompts, int) or lens_prompts <= 0:
                errors.append("Jlens fit prompt count is absent or zero")

    if errors:
        tag = summary.get("tag", "<unknown>")
        details = "\n  - ".join(errors)
        raise ValueError(f"summary {tag!r} violates its analysis contract:\n  - {details}")


def summary_view(summary: dict[str, Any]) -> dict[str, Any]:
    """Return metadata suitable for terminals and experiment registries."""

    probes = summary.get("probes")
    if isinstance(probes, list):
        probe_count: int | None = len(probes)
    elif isinstance(probes, int):
        probe_count = probes
    else:
        probe_count = None
    return {
        "tag": summary["tag"],
        "parent": summary["model_a"],
        "descendant": summary["model_b"],
        "layers": summary["layers"],
        "readouts": summary.get("readouts"),
        "decoder_mode": summary.get("decoder_mode"),
        "primary_contrast": summary.get("primary_contrast"),
        "probe_count": probe_count,
        "seconds": summary.get("seconds"),
    }
