"""Readable, layer-resolved comparisons between related language models."""

from .manifest import PairJob, load_manifest, parse_manifest_line
from .metrics import (
    category_probability_statistics,
    domain_write_contrasts,
    jensen_shannon_from_logits,
    kl_divergence_from_logits,
    log_probability_delta,
    moved_probability_mass,
    normalized_depth,
    select_normalized_depth_layers,
    topk_jaccard,
    vocabulary_write_amount,
    weighted_direction_alignment,
)
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
    "CATEGORIES",
    "FUNCTIONAL_CATEGORIES",
    "TRACE_CATEGORIES",
    "PairJob",
    "SummaryContract",
    "category_probability_statistics",
    "categorize_functional_token",
    "categorize_trace_token",
    "categorize_token",
    "domain_write_contrasts",
    "jensen_shannon_from_logits",
    "infer_role_spans",
    "is_displayable_trace_token",
    "kl_divergence_from_logits",
    "load_manifest",
    "load_summary",
    "log_probability_delta",
    "moved_probability_mass",
    "normalized_depth",
    "parse_manifest_line",
    "topk_jaccard",
    "summary_view",
    "select_normalized_depth_layers",
    "validate_summary_contract",
    "validate_role_spans",
    "vocabulary_write_amount",
    "weighted_direction_alignment",
]

__version__ = "0.1.0"
