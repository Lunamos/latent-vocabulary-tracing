"""Torch helpers for GPU vocabulary readouts and output-decoder audits.

This module is an optional research dependency: importing the core LVT package
does not import torch.  The helpers keep the four state-by-decoder cells
explicit so an anchored hidden-state edit is never silently mixed with a
final-norm/unembedding edit. Any upstream transport (such as a parent Jlens)
is applied before these helpers and must be recorded separately.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

CELL_AA = "state_a_decoder_a"
CELL_BA = "state_b_decoder_a"
CELL_AB = "state_a_decoder_b"
CELL_BB = "state_b_decoder_b"

FOUR_CELL_CONTRASTS = {
    "state_parent_decoder": (CELL_AA, CELL_BA),
    "state_descendant_decoder": (CELL_AB, CELL_BB),
    "decoder_parent_state": (CELL_AA, CELL_AB),
    "decoder_descendant_state": (CELL_BA, CELL_BB),
    "native_total": (CELL_AA, CELL_BB),
}


def four_cell_logits(
    state_a: torch.Tensor,
    state_b: torch.Tensor,
    decoder_a: Callable[[torch.Tensor], torch.Tensor],
    decoder_b: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Apply both output decoders to both transported states."""

    if state_a.shape != state_b.shape:
        raise ValueError(f"state shape mismatch: {tuple(state_a.shape)} != {tuple(state_b.shape)}")
    cells = {
        CELL_AA: decoder_a(state_a),
        CELL_BA: decoder_a(state_b),
        CELL_AB: decoder_b(state_a),
        CELL_BB: decoder_b(state_b),
    }
    shapes = {tuple(value.shape) for value in cells.values()}
    if len(shapes) != 1:
        raise ValueError(f"decoder output shape mismatch: {sorted(shapes)}")
    return cells


def _check_logits(parent: torch.Tensor, descendant: torch.Tensor) -> None:
    if parent.shape != descendant.shape:
        raise ValueError(f"shape mismatch: {tuple(parent.shape)} != {tuple(descendant.shape)}")
    if parent.ndim < 2:
        raise ValueError("expected logits with a position and vocabulary dimension")


def _distribution(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probability = F.log_softmax(logits.float(), dim=-1)
    return log_probability, log_probability.exp()


def cell_distributions(
    logits: Mapping[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute each named cell's fp32 log-probability and probability once."""

    return {name: _distribution(values) for name, values in logits.items()}


def kl_from_log_probabilities(
    log_probability_p: torch.Tensor,
    probability_p: torch.Tensor,
    log_probability_q: torch.Tensor,
) -> torch.Tensor:
    """Return ``KL(p || q)`` without recomputing either distribution."""

    zero = torch.zeros((), device=probability_p.device)
    return torch.where(
        probability_p > 0,
        probability_p * (log_probability_p - log_probability_q),
        zero,
    ).sum(dim=-1)


def scalar_pair_metrics_from_distributions(
    log_parent: torch.Tensor,
    probability_parent: torch.Tensor,
    log_descendant: torch.Tensor,
    probability_descendant: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute full-vocabulary divergences from cached distributions."""

    log_mixture = torch.logaddexp(log_parent, log_descendant) - torch.log(
        torch.tensor(2.0, device=log_parent.device)
    )
    kl_ab = kl_from_log_probabilities(log_parent, probability_parent, log_descendant)
    kl_ba = kl_from_log_probabilities(log_descendant, probability_descendant, log_parent)
    js = 0.5 * (
        kl_from_log_probabilities(log_parent, probability_parent, log_mixture)
        + kl_from_log_probabilities(log_descendant, probability_descendant, log_mixture)
    )
    return {"kl_ab": kl_ab, "kl_ba": kl_ba, "js": js}


def pair_metrics(
    parent_logits: torch.Tensor,
    descendant_logits: torch.Tensor,
    *,
    jaccard_k: int = 10,
    support_k: int = 50,
    parent_distribution: tuple[torch.Tensor, torch.Tensor] | None = None,
    descendant_distribution: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return divergences and an auditable union-top-k support representation.

    Floating-point support values and log-normalizers remain fp32.  The
    returned support coverage is the exact mass of the union of the parent and
    descendant top-``support_k`` sets at each position.
    """

    _check_logits(parent_logits, descendant_logits)
    if parent_logits.ndim != 2:
        raise ValueError("pair_metrics currently expects [position, vocabulary] logits")
    vocab_size = parent_logits.shape[-1]
    if not 1 <= jaccard_k <= support_k <= vocab_size:
        raise ValueError(
            f"require 1 <= jaccard_k <= support_k <= {vocab_size}, got {jaccard_k} and {support_k}"
        )

    parent = parent_logits.float()
    descendant = descendant_logits.float()
    if (parent_distribution is None) != (descendant_distribution is None):
        raise ValueError("parent and descendant distributions must be supplied together")
    if parent_distribution is None:
        log_parent, probability_parent = _distribution(parent)
        log_descendant, probability_descendant = _distribution(descendant)
    else:
        log_parent, probability_parent = parent_distribution
        log_descendant, probability_descendant = descendant_distribution
        expected_shape = tuple(parent.shape)
        if any(
            tuple(values.shape) != expected_shape
            for values in (
                log_parent,
                probability_parent,
                log_descendant,
                probability_descendant,
            )
        ):
            raise ValueError("cached distribution shape does not match logits")
    output = scalar_pair_metrics_from_distributions(
        log_parent,
        probability_parent,
        log_descendant,
        probability_descendant,
    )

    value_a, top_a = parent.topk(support_k, dim=-1)
    value_b, top_b = descendant.topk(support_k, dim=-1)
    intersection = top_a[:, :jaccard_k, None] == top_b[:, None, :jaccard_k]
    intersection_count = intersection.any(dim=-1).sum(dim=-1).float()
    jaccard = intersection_count / (2 * jaccard_k - intersection_count)

    parent_at_b = torch.gather(parent, 1, top_b)
    descendant_at_a = torch.gather(descendant, 1, top_a)
    probability_a_at_a = torch.gather(probability_parent, 1, top_a)
    probability_b_at_a = torch.gather(probability_descendant, 1, top_a)
    probability_a_at_b = torch.gather(probability_parent, 1, top_b)
    probability_b_at_b = torch.gather(probability_descendant, 1, top_b)
    b_is_new = ~(top_b[:, :, None] == top_a[:, None, :]).any(dim=-1)
    delta_log_probability = log_descendant - log_parent
    support_mass_a = probability_a_at_a.sum(dim=-1) + (probability_a_at_b * b_is_new).sum(dim=-1)
    support_mass_b = probability_b_at_a.sum(dim=-1) + (probability_b_at_b * b_is_new).sum(dim=-1)

    output.update(
        {
            "jaccard": jaccard,
            "top_a": top_a.int(),
            "top_b": top_b.int(),
            "val_a": value_a,
            "val_b": value_b,
            "a_at_b": parent_at_b,
            "b_at_a": descendant_at_a,
            "delta_logp_at_a": torch.gather(delta_log_probability, 1, top_a),
            "delta_logp_at_b": torch.gather(delta_log_probability, 1, top_b),
            "b_is_new": b_is_new,
            "lse_a": torch.logsumexp(parent, dim=-1),
            "lse_b": torch.logsumexp(descendant, dim=-1),
            "support_mass_a": support_mass_a,
            "support_mass_b": support_mass_b,
            "outside_mass_a": 1.0 - support_mass_a,
            "outside_mass_b": 1.0 - support_mass_b,
        }
    )
    return output


def four_cell_scalar_metrics(
    logits: Mapping[str, torch.Tensor],
    *,
    distributions: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute the five named contrasts from four state-by-decoder cells."""

    required = {CELL_AA, CELL_BA, CELL_AB, CELL_BB}
    missing = required.difference(logits)
    if missing:
        raise ValueError(f"missing four-cell logits: {sorted(missing)}")
    shapes = {tuple(logits[cell].shape) for cell in required}
    if len(shapes) != 1:
        raise ValueError(f"four-cell shape mismatch: {sorted(shapes)}")

    if distributions is None:
        distributions = cell_distributions(logits)
    elif required.difference(distributions):
        raise ValueError(
            f"missing four-cell distributions: {sorted(required.difference(distributions))}"
        )
    output = {}
    for name, (parent_cell, descendant_cell) in FOUR_CELL_CONTRASTS.items():
        log_a, probability_a = distributions[parent_cell]
        log_b, probability_b = distributions[descendant_cell]
        output[name] = scalar_pair_metrics_from_distributions(
            log_a,
            probability_a,
            log_b,
            probability_b,
        )
    return output


def matched_faithfulness(
    final_logits: Mapping[str, torch.Tensor],
    readout_logits: Mapping[str, torch.Tensor],
    *,
    final_distributions: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    readout_distributions: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Measure faithfulness without crossing output-decoder coordinates.

    Native-output-decoder faithfulness compares AA and BB with final logits
    expressed through their respective output decoders. Parent-anchored
    faithfulness compares AA and BA with both final states decoded by decoder
    A. The descendant's native final logits never enter the anchored quantity.
    This helper does not know whether an upstream transport was identity,
    parent anchored, or native; callers must label it independently.
    """

    needed = {CELL_AA, CELL_BA, CELL_BB}
    for name, mapping in (("final", final_logits), ("readout", readout_logits)):
        missing = needed.difference(mapping)
        if missing:
            raise ValueError(f"missing {name} logits: {sorted(missing)}")

    if final_distributions is None:
        final_distributions = cell_distributions(final_logits)
    if readout_distributions is None:
        readout_distributions = cell_distributions(readout_logits)

    def faith(final_cell: str, readout_cell: str) -> torch.Tensor:
        log_final, probability_final = final_distributions[final_cell]
        log_readout, _ = readout_distributions[readout_cell]
        return kl_from_log_probabilities(log_final, probability_final, log_readout)

    parent = faith(CELL_AA, CELL_AA)
    return {
        "native_output_decoder": {
            "a": parent,
            "b": faith(CELL_BB, CELL_BB),
        },
        "parent_anchored": {
            "a": parent,
            "b": faith(CELL_BA, CELL_BA),
        },
    }


def matched_readout_diagnostics(
    final_logits: Mapping[str, torch.Tensor],
    readout_logits: Mapping[str, torch.Tensor],
    malformed_token_mask: torch.Tensor,
    *,
    topk: int = 10,
    final_distributions: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    readout_distributions: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    faithfulness: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Measure readout quality within each output-decoder coordinate.

    In addition to final-to-readout KL, report top-k Jaccard overlap and the
    readout probability assigned to objectively malformed tokenizer pieces
    (replacement characters or disallowed control characters). The same
    coordinate pairings as :func:`matched_faithfulness` are used throughout.
    """

    if topk <= 0:
        raise ValueError("topk must be positive")
    needed = {CELL_AA, CELL_BA, CELL_BB}
    for name, mapping in (("final", final_logits), ("readout", readout_logits)):
        missing = needed.difference(mapping)
        if missing:
            raise ValueError(f"missing {name} logits: {sorted(missing)}")

    vocab_size = final_logits[CELL_AA].shape[-1]
    if malformed_token_mask.dtype != torch.bool:
        raise TypeError("malformed_token_mask must be boolean")
    if malformed_token_mask.ndim != 1 or malformed_token_mask.shape[0] != vocab_size:
        raise ValueError("malformed_token_mask must have one entry per vocabulary item")
    if final_distributions is None:
        final_distributions = cell_distributions(final_logits)
    if readout_distributions is None:
        readout_distributions = cell_distributions(readout_logits)

    if faithfulness is None:
        faithfulness = matched_faithfulness(
            final_logits,
            readout_logits,
            final_distributions=final_distributions,
            readout_distributions=readout_distributions,
        )
    pairings = {
        "native_output_decoder": {"a": (CELL_AA, CELL_AA), "b": (CELL_BB, CELL_BB)},
        "parent_anchored": {"a": (CELL_AA, CELL_AA), "b": (CELL_BA, CELL_BA)},
    }
    k = min(topk, vocab_size)
    output: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for coordinate, models in pairings.items():
        output[coordinate] = {}
        for model_key, (final_cell, readout_cell) in models.items():
            final_ids = final_logits[final_cell].topk(k, dim=-1).indices
            readout_ids = readout_logits[readout_cell].topk(k, dim=-1).indices
            intersection = (
                (final_ids.unsqueeze(-1) == readout_ids.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
            )
            union = 2 * k - intersection
            probability_readout = readout_distributions[readout_cell][1]
            output[coordinate][model_key] = {
                "faith_kl": faithfulness[coordinate][model_key],
                "topk_jaccard": intersection.float() / union.float(),
                "malformed_token_mass": probability_readout[..., malformed_token_mask].sum(dim=-1),
            }
    return output


def _category_statistics_from_probabilities(
    parent: torch.Tensor,
    descendant: torch.Tensor,
    category_ids: torch.Tensor,
    *,
    n_categories: int,
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    if category_ids.ndim != 1 or category_ids.shape[0] != parent.shape[-1]:
        raise ValueError("category_ids must contain one entry per vocabulary item")
    if n_categories <= 0:
        raise ValueError("n_categories must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if int(category_ids.min()) < 0 or int(category_ids.max()) >= n_categories:
        raise ValueError("category_ids contains an invalid category index")

    delta = descendant - parent
    category_ids = category_ids.to(parent.device, dtype=torch.int64)
    indices = category_ids.unsqueeze(0).expand(parent.shape[0], -1)

    def reduce(values: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            parent.shape[0], n_categories, dtype=torch.float32, device=parent.device
        )
        return output.scatter_add_(1, indices, values.float())

    parent_mass = reduce(parent)
    descendant_mass = reduce(descendant)
    promoted = reduce(delta.clamp_min(0))
    suppressed = reduce((-delta).clamp_min(0))
    signed = promoted - suppressed
    absolute = promoted + suppressed
    turnover = 0.5 * absolute
    total_turnover = turnover.sum(dim=-1, keepdim=True)
    composition = torch.where(total_turnover > 0, turnover / total_turnover, 0.0)
    balance = torch.where(absolute > 0, signed / absolute, 0.0)
    reference_mass = 0.5 * (parent_mass + descendant_mass)
    turnover_enrichment_bits = torch.where(
        total_turnover > 0,
        torch.log2((composition + epsilon) / (reference_mass + epsilon)),
        0.0,
    )
    log_mass_ratio = torch.log(descendant_mass + epsilon) - torch.log(parent_mass + epsilon)
    return {
        "parent_mass": parent_mass,
        "descendant_mass": descendant_mass,
        "promoted": promoted,
        "suppressed": suppressed,
        "signed": signed,
        "turnover": turnover,
        "composition": composition,
        "balance": balance,
        "turnover_enrichment_bits": turnover_enrichment_bits,
        "log_mass_ratio": log_mass_ratio,
    }


def category_statistics(
    parent_logits: torch.Tensor,
    descendant_logits: torch.Tensor,
    category_ids: torch.Tensor,
    *,
    n_categories: int,
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """GPU category decomposition matching ``category_probability_statistics``."""

    _check_logits(parent_logits, descendant_logits)
    if parent_logits.ndim != 2:
        raise ValueError("category_statistics currently expects [position, vocabulary] logits")
    _, parent = _distribution(parent_logits)
    _, descendant = _distribution(descendant_logits)
    return _category_statistics_from_probabilities(
        parent,
        descendant,
        category_ids,
        n_categories=n_categories,
        epsilon=epsilon,
    )


def vocabulary_edit_statistics(
    parent_logits: torch.Tensor,
    descendant_logits: torch.Tensor,
    category_ids: torch.Tensor,
    *,
    n_categories: int,
    epsilon: float = 1e-12,
    parent_probability: torch.Tensor | None = None,
    descendant_probability: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return exact category statistics and per-position FP32 token changes.

    The second return value is ``p_descendant - p_parent`` on the full
    vocabulary.  Keeping it long enough for the caller to average within a
    probe permits exact promoted/suppressed token rankings without assuming
    that an important new token was already in the parent's top-k support.
    """

    _check_logits(parent_logits, descendant_logits)
    if parent_logits.ndim != 2:
        raise ValueError("vocabulary_edit_statistics expects [position, vocabulary] logits")
    if (parent_probability is None) != (descendant_probability is None):
        raise ValueError("parent and descendant probabilities must be supplied together")
    if parent_probability is None:
        _, parent = _distribution(parent_logits)
        _, descendant = _distribution(descendant_logits)
    else:
        parent, descendant = parent_probability, descendant_probability
        if parent.shape != parent_logits.shape or descendant.shape != descendant_logits.shape:
            raise ValueError("cached probability shape does not match logits")
    categories = _category_statistics_from_probabilities(
        parent,
        descendant,
        category_ids,
        n_categories=n_categories,
        epsilon=epsilon,
    )
    return categories, descendant - parent


def aggregate_category_statistics(
    statistics: Mapping[str, torch.Tensor],
    position_indices: slice | torch.Tensor | None = None,
    *,
    epsilon: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Average primitive masses over positions, then derive category ratios.

    Composition, balance, and enrichment are nonlinear.  Computing them after
    the primitive parent/descendant/promoted/suppressed masses have been
    averaged prevents tiny-edit positions from receiving the same influence as
    positions that carry substantial probability movement.
    """

    primitive_names = ("parent_mass", "descendant_mass", "promoted", "suppressed")
    missing = set(primitive_names).difference(statistics)
    if missing:
        raise ValueError(f"missing primitive category statistics: {sorted(missing)}")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    shapes = {tuple(statistics[name].shape) for name in primitive_names}
    if len(shapes) != 1:
        raise ValueError(f"category primitive shape mismatch: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("category statistics must have [position, category] shape")

    selected = {
        name: values if position_indices is None else values[position_indices]
        for name, values in statistics.items()
        if name in primitive_names
    }
    if not selected["parent_mass"].shape[0]:
        raise ValueError("cannot aggregate an empty set of positions")
    parent_mass = selected["parent_mass"].mean(dim=0)
    descendant_mass = selected["descendant_mass"].mean(dim=0)
    promoted = selected["promoted"].mean(dim=0)
    suppressed = selected["suppressed"].mean(dim=0)
    signed = promoted - suppressed
    absolute = promoted + suppressed
    turnover = 0.5 * absolute
    total_turnover = turnover.sum()
    composition = torch.where(total_turnover > 0, turnover / total_turnover, 0.0)
    balance = torch.where(absolute > 0, signed / absolute, 0.0)
    reference_mass = 0.5 * (parent_mass + descendant_mass)
    turnover_enrichment_bits = torch.where(
        total_turnover > 0,
        torch.log2((composition + epsilon) / (reference_mass + epsilon)),
        0.0,
    )
    log_mass_ratio = torch.log(descendant_mass + epsilon) - torch.log(parent_mass + epsilon)
    return {
        "parent_mass": parent_mass,
        "descendant_mass": descendant_mass,
        "promoted": promoted,
        "suppressed": suppressed,
        "signed": signed,
        "turnover": turnover,
        "composition": composition,
        "balance": balance,
        "turnover_enrichment_bits": turnover_enrichment_bits,
        "log_mass_ratio": log_mass_ratio,
    }


def aggregate_category_statistics_by_spans(
    statistics: Mapping[str, torch.Tensor],
    role_spans: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Average per-position category statistics within each sequence role.

    A role may contain multiple disjoint half-open spans.  Positions are
    concatenated before averaging, so each token position has equal weight
    within a probe; the caller can then average probe-level vectors so long
    traces do not dominate the experiment.
    """

    if not statistics:
        raise ValueError("statistics cannot be empty")
    first = next(iter(statistics.values()))
    if first.ndim != 2:
        raise ValueError("category statistics must have [position, category] shape")
    n_positions = first.shape[0]
    output: dict[str, dict[str, torch.Tensor]] = {}
    for role, spans in role_spans.items():
        valid_spans: list[tuple[int, int]] = []
        for span in spans:
            if len(span) != 2:
                raise ValueError(f"role {role!r} contains a malformed span")
            start, end = int(span[0]), int(span[1])
            if start < 0 or end <= start:
                raise ValueError(f"role {role!r} contains invalid span {(start, end)}")
            if start < n_positions:
                valid_spans.append((start, min(end, n_positions)))
        if not valid_spans:
            continue
        indices = torch.cat(
            [torch.arange(start, end, device=first.device) for start, end in valid_spans]
        )
        output[role] = aggregate_category_statistics(statistics, indices)
    return output


def aggregate_position_metrics_by_spans(
    metrics: Mapping[str, torch.Tensor],
    role_spans: Mapping[str, Sequence[Sequence[int]]],
    *,
    fields: Sequence[str],
) -> dict[str, dict[str, torch.Tensor]]:
    """Average one-dimensional per-position metrics within protocol roles."""

    missing = set(fields).difference(metrics)
    if missing:
        raise ValueError(f"missing position metrics: {sorted(missing)}")
    lengths = {metrics[field].shape for field in fields}
    if len(lengths) != 1 or len(next(iter(lengths))) != 1:
        raise ValueError("position metric fields must share shape [position]")
    n_positions = next(iter(lengths))[0]
    output: dict[str, dict[str, torch.Tensor]] = {}
    for role, spans in role_spans.items():
        indices = []
        for span in spans:
            if len(span) != 2:
                raise ValueError(f"role {role!r} contains a malformed span")
            start, end = int(span[0]), int(span[1])
            if start < 0 or end <= start:
                raise ValueError(f"role {role!r} contains invalid span {(start, end)}")
            if start < n_positions:
                indices.append((start, min(end, n_positions)))
        if not indices:
            continue
        output[role] = {
            field: torch.cat([metrics[field][start:end] for start, end in indices]).mean()
            for field in fields
        }
    return output


def net_token_direction_scores(
    signed_probability_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint promotion and suppression scores for exact tokens.

    Category turnover deliberately preserves positive and negative movement
    before cancellation. Exact token labels answer a different question:
    whether a token increased or decreased after position/probe/depth
    averaging. Ranking gross movement can otherwise put one token in both
    lists, or call a token promoted despite a negative net change.
    """

    if signed_probability_delta.ndim != 1:
        raise ValueError("signed_probability_delta must be a vocabulary vector")
    return (
        signed_probability_delta.clamp_min(0),
        (-signed_probability_delta).clamp_min(0),
    )
