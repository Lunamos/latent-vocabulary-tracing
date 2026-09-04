import numpy as np
import pytest

from latent_vocabulary_tracing.metrics import (
    jensen_shannon_from_logits,
    kl_divergence_from_logits,
    log_probability_delta,
    moved_probability_mass,
    topk_jaccard,
    vocabulary_write_amount,
    weighted_direction_alignment,
)


def test_write_amount_is_descendant_to_parent_kl_in_nats():
    parent_probability = np.array([[0.8, 0.2]])
    descendant_probability = np.array([[0.5, 0.5]])
    parent_logits = np.log(parent_probability)
    descendant_logits = np.log(descendant_probability)
    expected = np.sum(
        descendant_probability * np.log(descendant_probability / parent_probability), axis=-1
    )
    assert vocabulary_write_amount(parent_logits, descendant_logits) == pytest.approx(expected)
    assert vocabulary_write_amount(parent_logits, parent_logits) == pytest.approx(np.array([0.0]))
    assert vocabulary_write_amount(parent_logits, descendant_logits) == pytest.approx(
        kl_divergence_from_logits(descendant_logits, parent_logits)
    )


def test_js_is_zero_for_identical_logits_and_symmetric():
    left = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    right = np.array([[0.0, 2.0, -1.0], [1.0, -1.0, 0.0]])
    assert np.allclose(jensen_shannon_from_logits(left, left), 0.0, atol=1e-12)
    assert np.allclose(
        jensen_shannon_from_logits(left, right),
        jensen_shannon_from_logits(right, left),
    )


def test_topk_jaccard_and_support_delta():
    left = np.array([[3.0, 2.0, 1.0, 0.0]])
    right = np.array([[3.0, 0.0, 2.0, 1.0]])
    assert topk_jaccard(left, right, k=2) == pytest.approx(np.array([1 / 3]))
    support = np.array([[0, 2]])
    assert log_probability_delta(left, right, support).shape == (1, 2)


def test_moved_mass_balances_for_distributions():
    parent = np.array([[0.6, 0.3, 0.1]])
    descendant = np.array([[0.2, 0.5, 0.3]])
    promoted, suppressed = moved_probability_mass(parent, descendant)
    assert promoted == pytest.approx(np.array([0.4]))
    assert suppressed == pytest.approx(np.array([0.4]))


def test_direction_alignment_recovers_same_and_opposite_directions():
    delta = np.array([[1.0, 0.0, -1.0], [0.5, -0.5, 0.0]])
    assert weighted_direction_alignment(delta, delta) == pytest.approx(1.0)
    assert weighted_direction_alignment(delta, -delta) == pytest.approx(-1.0)
