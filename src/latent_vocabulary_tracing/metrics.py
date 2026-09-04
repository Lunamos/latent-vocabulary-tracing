"""Numerically stable statistics used by Latent Vocabulary Tracing.

All public functions operate on NumPy arrays and keep the vocabulary dimension
explicit. This module intentionally has no model-loading or filesystem logic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_matching_float_arrays(a: ArrayLike, b: ArrayLike) -> tuple[NDArray, NDArray]:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    if left.ndim == 0:
        raise ValueError("expected at least one dimension")
    return left, right


def _log_softmax(x: NDArray, axis: int = -1) -> NDArray:
    peak = np.max(x, axis=axis, keepdims=True)
    shifted = x - peak
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def kl_divergence_from_logits(
    logits_p: ArrayLike, logits_q: ArrayLike, axis: int = -1
) -> NDArray:
    """Return ``D_KL(p || q)`` in nats along ``axis``.

    Argument order follows the mathematical expression.  For LVT's primary
    parent-to-descendant write amount, pass descendant logits first and parent
    logits second, or use :func:`vocabulary_write_amount`.
    """

    p_logits, q_logits = _as_matching_float_arrays(logits_p, logits_q)
    log_p = _log_softmax(p_logits, axis=axis)
    log_q = _log_softmax(q_logits, axis=axis)
    return np.sum(np.exp(log_p) * (log_p - log_q), axis=axis)


def vocabulary_write_amount(
    logits_parent: ArrayLike, logits_descendant: ArrayLike, axis: int = -1
) -> NDArray:
    """Return LVT write amount ``D_KL(p_descendant || p_parent)`` in nats.

    The logits must already be expressed in the same vocabulary coordinate
    system, normally by applying a decoder anchored to the parent model.
    """

    return kl_divergence_from_logits(logits_descendant, logits_parent, axis=axis)


def jensen_shannon_from_logits(
    logits_parent: ArrayLike, logits_descendant: ArrayLike, axis: int = -1
) -> NDArray:
    """Return Jensen--Shannon divergence in nats along ``axis``.

    The calculation remains in log space until the final weighted sums and is
    symmetric by construction. Identical distributions return zero up to
    floating-point precision.
    """

    parent, descendant = _as_matching_float_arrays(logits_parent, logits_descendant)
    log_parent = _log_softmax(parent, axis=axis)
    log_descendant = _log_softmax(descendant, axis=axis)
    log_mixture = np.logaddexp(log_parent, log_descendant) - np.log(2.0)
    prob_parent = np.exp(log_parent)
    prob_descendant = np.exp(log_descendant)
    kl_parent = np.sum(prob_parent * (log_parent - log_mixture), axis=axis)
    kl_descendant = np.sum(prob_descendant * (log_descendant - log_mixture), axis=axis)
    return 0.5 * (kl_parent + kl_descendant)


def topk_jaccard(logits_parent: ArrayLike, logits_descendant: ArrayLike, k: int = 10) -> NDArray:
    """Jaccard overlap between the two top-``k`` vocabulary sets."""

    parent, descendant = _as_matching_float_arrays(logits_parent, logits_descendant)
    vocab_size = parent.shape[-1]
    if not 1 <= k <= vocab_size:
        raise ValueError(f"k must be in [1, {vocab_size}], got {k}")
    parent_ids = np.argpartition(parent, -k, axis=-1)[..., -k:]
    descendant_ids = np.argpartition(descendant, -k, axis=-1)[..., -k:]
    intersection = (
        (parent_ids[..., :, None] == descendant_ids[..., None, :]).any(axis=-1).sum(axis=-1)
    )
    return intersection / (2 * k - intersection)


def log_probability_delta(
    logits_parent: ArrayLike,
    logits_descendant: ArrayLike,
    support: ArrayLike | None = None,
) -> NDArray:
    """Return ``log p_descendant - log p_parent`` on a fixed support.

    If ``support`` is omitted, the full vocabulary is returned. Otherwise its
    final dimension indexes vocabulary ids and all leading dimensions must
    broadcast with the logits' leading dimensions.
    """

    parent, descendant = _as_matching_float_arrays(logits_parent, logits_descendant)
    delta = _log_softmax(descendant) - _log_softmax(parent)
    if support is None:
        return delta
    ids = np.asarray(support, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= parent.shape[-1]):
        raise ValueError("support contains an out-of-vocabulary id")
    return np.take_along_axis(delta, ids, axis=-1)


def moved_probability_mass(
    probability_parent: ArrayLike, probability_descendant: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Return promoted and suppressed probability mass for every observation."""

    parent, descendant = _as_matching_float_arrays(probability_parent, probability_descendant)
    if np.any(parent < 0) or np.any(descendant < 0):
        raise ValueError("probabilities must be non-negative")
    difference = descendant - parent
    promoted = np.maximum(difference, 0).sum(axis=-1)
    suppressed = np.maximum(-difference, 0).sum(axis=-1)
    return promoted, suppressed


def weighted_direction_alignment(delta_a: ArrayLike, delta_b: ArrayLike) -> float:
    """Weighted mean Pearson alignment between token-change directions.

    Each row is a matched layer/position observation and the final dimension is
    vocabulary support. Rows are weighted by the geometric mean of their two
    change magnitudes, matching the research pipeline's direction statistic.
    """

    left, right = _as_matching_float_arrays(delta_a, delta_b)
    if left.ndim == 1:
        left = left[None, :]
        right = right[None, :]
    left = left.reshape(-1, left.shape[-1])
    right = right.reshape(-1, right.shape[-1])
    left_centered = left - left.mean(axis=-1, keepdims=True)
    right_centered = right - right.mean(axis=-1, keepdims=True)
    left_norm = np.linalg.norm(left_centered, axis=-1)
    right_norm = np.linalg.norm(right_centered, axis=-1)
    valid = (left_norm > 0) & (right_norm > 0)
    if not np.any(valid):
        return float("nan")
    correlations = np.sum(left_centered * right_centered, axis=-1) / (
        left_norm * right_norm + 1e-12
    )
    weights = np.sqrt(np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1))
    valid &= np.isfinite(correlations) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.average(correlations[valid], weights=weights[valid]))
