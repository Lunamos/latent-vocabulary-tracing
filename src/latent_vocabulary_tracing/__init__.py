"""Readable, layer-resolved comparisons between related language models."""

from .manifest import PairJob, load_manifest, parse_manifest_line
from .metrics import (
    jensen_shannon_from_logits,
    kl_divergence_from_logits,
    log_probability_delta,
    moved_probability_mass,
    topk_jaccard,
    vocabulary_write_amount,
    weighted_direction_alignment,
)
from .taxonomy import CATEGORIES, categorize_token

__all__ = [
    "CATEGORIES",
    "PairJob",
    "categorize_token",
    "jensen_shannon_from_logits",
    "kl_divergence_from_logits",
    "load_manifest",
    "log_probability_delta",
    "moved_probability_mass",
    "parse_manifest_line",
    "topk_jaccard",
    "vocabulary_write_amount",
    "weighted_direction_alignment",
]

__version__ = "0.1.0"
