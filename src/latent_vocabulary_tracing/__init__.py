"""Readable, layer-resolved comparisons between related language models."""

from .compatibility import validate_architecture_pair, validate_tokenizer_vocabularies
from .manifest import PairJob, load_manifest, parse_manifest_line
from .metrics import (
    benjamini_hochberg,
    category_probability_statistics,
    deterministic_balanced_split,
    domain_write_contrasts,
    jensen_shannon_from_logits,
    kl_divergence_from_logits,
    log_probability_delta,
    moved_probability_mass,
    normalized_depth,
    one_sided_sign_test,
    select_normalized_depth_layers,
    topk_jaccard,
    vocabulary_write_amount,
    weighted_direction_alignment,
)
from .provenance import (
    canonicalize_model_config,
    model_config_hash,
    snapshot_revision_from_path,
    stable_json_hash,
)
from .registry import EdgeRecord, EdgeRegistry, load_edge_registry
from .spans import infer_role_spans, validate_role_spans
from .summary import SummaryContract, load_summary, summary_view, validate_summary_contract
from .taxonomy import (
    CATEGORIES,
    FUNCTIONAL_CATEGORIES,
    TRACE_CATEGORIES,
    categorize_functional_token,
    categorize_token,
    categorize_trace_token,
    is_displayable_trace_token,
)

__all__ = [
    "benjamini_hochberg",
    "CATEGORIES",
    "EdgeRecord",
    "EdgeRegistry",
    "FUNCTIONAL_CATEGORIES",
    "TRACE_CATEGORIES",
    "PairJob",
    "SummaryContract",
    "category_probability_statistics",
    "canonicalize_model_config",
    "categorize_functional_token",
    "categorize_trace_token",
    "categorize_token",
    "domain_write_contrasts",
    "deterministic_balanced_split",
    "jensen_shannon_from_logits",
    "infer_role_spans",
    "is_displayable_trace_token",
    "kl_divergence_from_logits",
    "load_manifest",
    "load_edge_registry",
    "load_summary",
    "log_probability_delta",
    "moved_probability_mass",
    "model_config_hash",
    "normalized_depth",
    "one_sided_sign_test",
    "parse_manifest_line",
    "stable_json_hash",
    "topk_jaccard",
    "summary_view",
    "select_normalized_depth_layers",
    "snapshot_revision_from_path",
    "validate_summary_contract",
    "validate_architecture_pair",
    "validate_role_spans",
    "validate_tokenizer_vocabularies",
    "vocabulary_write_amount",
    "weighted_direction_alignment",
]

__version__ = "0.1.0"
