import json

import pytest

from latent_vocabulary_tracing.summary import (
    SummaryContract,
    load_summary,
    validate_summary_contract,
)


def valid_summary() -> dict:
    domains = {}
    for domain in ("math", "code", "agent", "neutral"):
        domains[domain] = {
            "n": 1,
            "LL": {"4": {"kl_ba_resp": 0.2}},
            "role_metrics": {"LL": {"4": {"model_response": {"kl_ba": 0.2}}}},
            "four_cell": {
                "LL": {"4": {"state_parent_decoder": {"kl_ba_resp": 0.2}}}
            },
        }
    return {
        "schema_version": 2,
        "tag": "example",
        "edge_registry": {
            "hash": "registry",
            "hash_scheme": "sha256_canonical_json_v1",
            "entry": {
                "tag": "example",
                "parent": "org/parent",
                "descendant": "org/descendant",
            },
        },
        "model_a": "org/parent",
        "model_b": "org/descendant",
        "layers": [4],
        "readouts": ["LL"],
        "lens_parent_id": "org/parent",
        "lens_hash": "lens",
        "lens_n_prompts": 120,
        "lens_source_layers": [4],
        "lens_d_model": 8,
        "decoder_mode": "parent_anchored",
        "primary_contrast": "state_parent_decoder",
        "final_decoder_mode": "native_per_model_control",
        "transport_mode": {"LL": "identity", "J": "parent_anchored"},
        "four_cell_decoder_component": "final_norm_and_unembedding",
        "probe_protocol": {
            "neutral_control": "chat_matched_continuation",
            "neutral_source_context_tokens": 48,
            "neutral_source_dataset": "Salesforce/wikitext:wikitext-103-raw-v1",
            "neutral_source_split": "train",
            "neutral_source_sampling": "spaced_qualifying_records",
            "neutral_source_start": 1500,
            "neutral_source_stride": 97,
            "neutral_source_document_count": 30,
        },
        "category_statistics": {
            "enabled": True,
            "dtype": "fp32",
            "support": "full_vocabulary",
            "role_conditioned": True,
            "nonlinear_aggregation": (
                "derive_after_averaging_primitive_masses_within_probe"
            ),
            "top_change_support": "full_vocabulary",
            "top_change_ranking": "net_probability_delta_after_averaging",
            "turnover_enrichment": (
                "log2_turnover_composition_over_midpoint_category_probability_mass"
            ),
            "representative_layer_rule": (
                "maximum_discovery_response_kl_ba_within_depth_summary"
            ),
            "token_inference": "discovery_selection_then_heldout_sign_test_bh",
            "taxonomy_hash": "taxonomy",
        },
        "stored_support_dtype": "fp32",
        "models": {
            key: {
                "id": "org/parent" if key == "a" else "org/descendant",
                "revision": f"rev-{key}",
                "architecture": "Model",
                "hidden_size": 8,
                "n_layers": 5,
                "vocab_size": 16,
                "norm_type": "RMSNorm",
                "rope": {"rope_type": "default"},
                "config_hash": f"config-{key}",
                "config_hash_scheme": "canonical_model_config_v1",
            }
            for key in ("a", "b")
        },
        "probe_hash": "probes",
        "probe_inference_split": {
            "method": "sha256_rank_balanced_within_domain",
            "mapping_hash": "split",
        },
        "analysis_provenance": {
            "revision": "abc123",
            "dirty": False,
            "runner_hash": "runner",
        },
        "tokenizer_note": {
            "id_piece_mismatches": 0,
            "tokenizer_hash_a": "tokenizer",
            "tokenizer_hash_b": "tokenizer",
            "tokenizer_hash_scheme": "sha256_canonical_json_v1",
        },
        "direction_statistics": {
            "quantity": "delta_logp",
            "dtype": "fp32",
            "support": "union_parent_descendant_top_k",
            "support_coverage_reported": True,
        },
        "agg": domains,
        "records": [
            {"key": f"{domain}-0", "kind": domain}
            for domain in ("math", "code", "agent", "neutral")
        ],
    }


def test_confirmatory_contract_accepts_explicit_anchored_result():
    contract = SummaryContract(
        readout="LL",
        require_category_statistics=True,
        require_fp32_store=True,
    )
    validate_summary_contract(valid_summary(), contract)


def test_contract_rejects_native_or_silent_readout_fallback():
    summary = valid_summary()
    summary["decoder_mode"] = "native_per_model"
    with pytest.raises(ValueError, match="decoder_mode"):
        validate_summary_contract(summary, SummaryContract(readout="LL"))
    with pytest.raises(ValueError, match="readout 'J' absent"):
        validate_summary_contract(valid_summary(), SummaryContract(readout="J"))


def test_contract_rejects_raw_text_as_the_primary_neutral_control():
    summary = valid_summary()
    summary["probe_protocol"]["neutral_control"] = "raw_wikitext"
    with pytest.raises(ValueError, match="chat-matched"):
        validate_summary_contract(summary, SummaryContract(readout="LL"))


def test_contract_rejects_pseudoreplicated_neutral_documents():
    summary = valid_summary()
    summary["probe_protocol"]["neutral_source_document_count"] = 1
    with pytest.raises(ValueError, match="document_count"):
        validate_summary_contract(summary, SummaryContract(readout="LL"))


def test_contract_can_require_complete_per_domain_probe_records():
    validate_summary_contract(
        valid_summary(),
        SummaryContract(readout="LL", expected_probes_per_domain=1),
    )
    summary = valid_summary()
    summary["records"].pop()
    with pytest.raises(ValueError, match="records contain 0 'neutral'"):
        validate_summary_contract(
            summary,
            SummaryContract(readout="LL", expected_probes_per_domain=1),
        )


def test_jlens_contract_requires_named_matching_parent_and_geometry():
    summary = valid_summary()
    summary["readouts"] = ["J", "LL"]
    for domain in summary["agg"].values():
        domain["J"] = domain["LL"]
        domain["role_metrics"]["J"] = domain["role_metrics"]["LL"]
        domain["four_cell"]["J"] = domain["four_cell"]["LL"]
    validate_summary_contract(summary, SummaryContract(readout="J"))

    summary["lens_parent_id"] = "org/unrelated"
    with pytest.raises(ValueError, match="does not match model_a"):
        validate_summary_contract(summary, SummaryContract(readout="J"))


def test_contract_rejects_architecture_or_full_vocabulary_mismatch():
    summary = valid_summary()
    summary["models"]["b"]["rope"] = {"rope_type": "modified"}
    summary["tokenizer_note"]["tokenizer_hash_b"] = "another-tokenizer"
    with pytest.raises(ValueError, match="architecture field 'rope'"):
        validate_summary_contract(summary, SummaryContract(readout="LL"))
    with pytest.raises(ValueError, match="full tokenizer vocabularies differ"):
        validate_summary_contract(summary, SummaryContract(readout="LL"))


def test_contract_checks_directed_response_kl_in_four_cell_output():
    summary = valid_summary()
    del summary["agg"]["agent"]["four_cell"]["LL"]["4"]["state_parent_decoder"][
        "kl_ba_resp"
    ]
    with pytest.raises(ValueError, match="four-cell"):
        validate_summary_contract(summary, SummaryContract(readout="LL"))


def test_contract_distinguishes_transport_from_output_decoder():
    summary = valid_summary()
    summary["transport_mode"]["LL"] = "parent_anchored"
    with pytest.raises(ValueError, match="transport_mode"):
        validate_summary_contract(summary, SummaryContract(readout="LL"))

    summary = valid_summary()
    summary["four_cell_decoder_component"] = "whole_readout"
    with pytest.raises(ValueError, match="decoder component"):
        validate_summary_contract(summary, SummaryContract(readout="LL"))


def test_fp32_contract_rejects_reconstructed_direction_values():
    summary = valid_summary()
    summary["direction_statistics"]["quantity"] = "reconstructed_from_logits"
    with pytest.raises(ValueError, match="direct delta_logp"):
        validate_summary_contract(
            summary,
            SummaryContract(readout="LL", require_fp32_store=True),
        )


def test_load_summary_applies_contract(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(valid_summary()), encoding="utf-8")
    loaded = load_summary(path, contract=SummaryContract(readout="LL"))
    assert loaded["tag"] == "example"


def test_contract_can_bind_a_summary_to_an_exact_edge_registry():
    validate_summary_contract(
        valid_summary(),
        SummaryContract(
            readout="LL",
            require_edge_registry=True,
            edge_registry_hash="registry",
        ),
    )

    summary = valid_summary()
    summary["edge_registry"]["entry"]["parent"] = "org/unrelated"
    with pytest.raises(ValueError, match="entry parent"):
        validate_summary_contract(
            summary,
            SummaryContract(readout="LL", require_edge_registry=True),
        )

    summary = valid_summary()
    summary["edge_registry"] = None
    with pytest.raises(ValueError, match="edge registry hash"):
        validate_summary_contract(
            summary,
            SummaryContract(readout="LL", require_edge_registry=True),
        )
