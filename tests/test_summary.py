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
            "LL": {"4": {"kl_ba_resp": 0.2}},
            "role_metrics": {"LL": {"4": {"model_response": {"kl_ba": 0.2}}}},
            "four_cell": {
                "LL": {"4": {"state_parent_decoder": {"kl_ba_resp": 0.2}}}
            },
        }
    return {
        "schema_version": 2,
        "tag": "example",
        "model_a": "org/parent",
        "model_b": "org/descendant",
        "layers": [4],
        "readouts": ["LL"],
        "decoder_mode": "parent_anchored",
        "primary_contrast": "state_parent_decoder",
        "final_decoder_mode": "native_per_model_control",
        "category_statistics": {
            "enabled": True,
            "dtype": "fp32",
            "support": "full_vocabulary",
            "role_conditioned": True,
            "top_change_support": "full_vocabulary",
            "top_change_ranking": "net_probability_delta_after_averaging",
        },
        "stored_support_dtype": "fp32",
        "models": {
            key: {
                "id": f"org/{key}",
                "revision": f"rev-{key}",
                "architecture": "Model",
                "hidden_size": 8,
                "n_layers": 5,
                "vocab_size": 16,
                "norm_type": "RMSNorm",
                "config_hash": f"config-{key}",
            }
            for key in ("a", "b")
        },
        "probe_hash": "probes",
        "tokenizer_note": {
            "id_piece_mismatches": 0,
            "tokenizer_hash_a": "tokenizer",
            "tokenizer_hash_b": "tokenizer",
        },
        "direction_statistics": {
            "quantity": "delta_logp",
            "dtype": "fp32",
            "support": "union_parent_descendant_top_k",
            "support_coverage_reported": True,
        },
        "agg": domains,
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


def test_contract_checks_directed_response_kl_in_four_cell_output():
    summary = valid_summary()
    del summary["agg"]["agent"]["four_cell"]["LL"]["4"]["state_parent_decoder"][
        "kl_ba_resp"
    ]
    with pytest.raises(ValueError, match="four-cell"):
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
