"""Numerically stable statistics used by Latent Vocabulary Tracing.

All public functions operate on NumPy arrays and keep the vocabulary dimension
explicit. This module intentionally has no model-loading or filesystem logic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalized_depth(layer: int, n_layers: int) -> float:
    """Return depth after zero-indexed block ``layer`` as a fraction of blocks."""

    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer {layer} is outside [0, {n_layers})")
    return (layer + 1) / n_layers


def select_normalized_depth_layers(
    layers: ArrayLike,
    *,
    n_layers: int,
    lower: float = 0.50,
    upper: float = 0.85,
) -> NDArray:
    """Select zero-indexed blocks whose completed depth lies in ``[lower, upper]``."""

    if not 0 <= lower <= upper <= 1:
        raise ValueError("require 0 <= lower <= upper <= 1")
    values = np.asarray(layers)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("layers must be a one-dimensional integer array")
    if np.any(values < 0) or np.any(values >= n_layers):
        raise ValueError(f"layers must lie in [0, {n_layers})")
    depths = (values.astype(np.float64) + 1.0) / n_layers
    return values[(depths >= lower) & (depths <= upper)]


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


def category_probability_statistics(
    probability_parent: ArrayLike,
    probability_descendant: ArrayLike,
    category_ids: ArrayLike,
    *,
    n_categories: int | None = None,
    epsilon: float = 1e-12,
) -> dict[str, NDArray]:
    """Decompose a probability edit into interpretable category statistics.

    ``category_ids[v]`` assigns vocabulary item ``v`` to exactly one category.
    The leading dimensions of the two probability arrays are retained and the
    final vocabulary dimension is replaced by a category dimension.

    The returned quantities deliberately separate three questions that a
    signed category sum alone conflates:

    ``turnover``
        Half the absolute probability movement carried by a category,
        ``0.5 * sum_C |p_descendant - p_parent|``.
    ``composition``
        A category's share of total turnover.  This is scale-free across edits
        of different overall magnitude.
    ``balance``
        Net signed change divided by absolute movement in the category.  It is
        in ``[-1, 1]``; zero can mean balanced promotion and suppression, not
        absence of an edit.

    Parent/descendant category mass and their smoothed log ratio are also
    returned.  The function expects full distributions when exact totals are
    required; callers using a truncated support must report its coverage.
    """

    parent, descendant = _as_matching_float_arrays(
        probability_parent, probability_descendant
    )
    if np.any(parent < 0) or np.any(descendant < 0):
        raise ValueError("probabilities must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    ids = np.asarray(category_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.shape[0] != parent.shape[-1]:
        raise ValueError(
            "category_ids must be one-dimensional with one entry per vocabulary item"
        )
    inferred_categories = int(ids.max()) + 1 if ids.size else 0
    if n_categories is None:
        n_categories = inferred_categories
    if n_categories <= 0:
        raise ValueError("n_categories must be positive")
    if np.any(ids < 0) or np.any(ids >= n_categories):
        raise ValueError("category_ids contains an invalid category index")

    flat_parent = parent.reshape(-1, parent.shape[-1])
    flat_descendant = descendant.reshape(-1, descendant.shape[-1])
    leading_shape = parent.shape[:-1]
    output_shape = (*leading_shape, n_categories)

    parent_mass = np.zeros((flat_parent.shape[0], n_categories), dtype=np.float64)
    descendant_mass = np.zeros_like(parent_mass)
    promoted = np.zeros_like(parent_mass)
    suppressed = np.zeros_like(parent_mass)
    difference = flat_descendant - flat_parent
    for category in range(n_categories):
        mask = ids == category
        parent_mass[:, category] = flat_parent[:, mask].sum(axis=-1)
        descendant_mass[:, category] = flat_descendant[:, mask].sum(axis=-1)
        category_difference = difference[:, mask]
        promoted[:, category] = np.maximum(category_difference, 0).sum(axis=-1)
        suppressed[:, category] = np.maximum(-category_difference, 0).sum(axis=-1)

    signed = promoted - suppressed
    absolute = promoted + suppressed
    turnover = 0.5 * absolute
    total_turnover = turnover.sum(axis=-1, keepdims=True)
    composition = np.divide(
        turnover,
        total_turnover,
        out=np.zeros_like(turnover),
        where=total_turnover > 0,
    )
    balance = np.divide(
        signed,
        absolute,
        out=np.zeros_like(signed),
        where=absolute > 0,
    )
    log_mass_ratio = np.log(descendant_mass + epsilon) - np.log(parent_mass + epsilon)

    return {
        "parent_mass": parent_mass.reshape(output_shape),
        "descendant_mass": descendant_mass.reshape(output_shape),
        "promoted": promoted.reshape(output_shape),
        "suppressed": suppressed.reshape(output_shape),
        "signed": signed.reshape(output_shape),
        "turnover": turnover.reshape(output_shape),
        "composition": composition.reshape(output_shape),
        "balance": balance.reshape(output_shape),
        "log_mass_ratio": log_mass_ratio.reshape(output_shape),
    }


def domain_write_contrasts(
    write_amounts: dict[str, ArrayLike],
    *,
    neutral: str = "neutral",
    noise_floor: float,
) -> dict[str, dict[str, NDArray]]:
    """Return raw, excess, and scale-free write amounts by probe domain.

    The log enrichment for domain ``d`` is
    ``log2((W_d + noise_floor) / (W_neutral + noise_floor))``.  The fixed floor
    prevents two numerically negligible writes from producing an arbitrarily
    large ratio.  Raw nats are retained because enrichment is a direction/
    locality statistic, not a replacement for edit magnitude.
    """

    if noise_floor <= 0:
        raise ValueError("noise_floor must be positive")
    if neutral not in write_amounts:
        raise ValueError(f"missing neutral domain {neutral!r}")

    raw = {key: np.asarray(value, dtype=np.float64) for key, value in write_amounts.items()}
    neutral_value = raw[neutral]
    output: dict[str, dict[str, NDArray]] = {}
    for domain, value in raw.items():
        try:
            excess = value - neutral_value
        except ValueError as exc:
            raise ValueError(
                f"domain {domain!r} does not broadcast with neutral domain {neutral!r}"
            ) from exc
        enrichment = np.log2(
            (value + noise_floor) / (neutral_value + noise_floor)
        )
        output[domain] = {
            "raw": value,
            "excess_over_neutral": excess,
            "log2_enrichment_over_neutral": enrichment,
        }
    return output


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
