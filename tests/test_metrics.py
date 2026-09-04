import numpy as np
import pytest

from latent_vocabulary_tracing.metrics import (
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


def test_category_statistics_separate_turnover_composition_and_balance():
    parent = np.array([[0.40, 0.20, 0.30, 0.10]])
    descendant = np.array([[0.25, 0.35, 0.10, 0.30]])
    # The first category has equal promotion and suppression: signed change is
    # zero even though it carries substantial probability turnover.
    stats = category_probability_statistics(
        parent,
        descendant,
        category_ids=np.array([0, 0, 1, 1]),
    )
    assert stats["signed"] == pytest.approx(np.array([[0.0, 0.0]]))
    assert stats["turnover"] == pytest.approx(np.array([[0.15, 0.20]]))
    assert stats["composition"] == pytest.approx(np.array([[3 / 7, 4 / 7]]))
    assert stats["balance"] == pytest.approx(np.array([[0.0, 0.0]]))
    assert stats["parent_mass"] == pytest.approx(np.array([[0.60, 0.40]]))
    assert stats["descendant_mass"] == pytest.approx(np.array([[0.60, 0.40]]))
    assert stats["turnover_enrichment_bits"] == pytest.approx(
        np.log2(np.array([[5 / 7, 10 / 7]]))
    )


def test_category_statistics_can_show_direction_within_category():
    parent = np.array([0.7, 0.2, 0.1])
    descendant = np.array([0.4, 0.2, 0.4])
    stats = category_probability_statistics(parent, descendant, [0, 0, 1])
    assert stats["balance"] == pytest.approx(np.array([-1.0, 1.0]))
    assert stats["composition"].sum() == pytest.approx(1.0)


def test_domain_write_contrasts_keep_raw_values_and_bound_specificity():
    contrasts = domain_write_contrasts(
        {
            "math": np.array([0.05]),
            "agent": np.array([0.01]),
            "neutral": np.array([0.01]),
        },
    )
    assert contrasts["math"]["raw"] == pytest.approx(np.array([0.05]))
    assert contrasts["math"]["excess_over_neutral"] == pytest.approx(np.array([0.04]))
    assert contrasts["math"]["normalized_specificity"] == pytest.approx(
        np.array([2.0 / 3.0])
    )
    assert contrasts["neutral"]["normalized_specificity"] == pytest.approx(
        np.array([0.0])
    )
    zero = domain_write_contrasts({"math": 0.0, "neutral": 0.0})
    assert zero["math"]["normalized_specificity"] == pytest.approx(0.0)


def test_category_and_domain_normalizations_reject_invalid_inputs():
    with pytest.raises(ValueError, match="one entry per vocabulary"):
        category_probability_statistics([0.5, 0.5], [0.4, 0.6], [0])
    with pytest.raises(ValueError, match="nonnegative"):
        domain_write_contrasts({"math": -0.1, "neutral": 0.0})


def test_normalized_depth_uses_completed_block_fraction():
    assert normalized_depth(0, 4) == pytest.approx(0.25)
    assert normalized_depth(3, 4) == pytest.approx(1.0)
    selected = select_normalized_depth_layers([0, 1, 2, 3], n_layers=4, lower=0.5, upper=0.85)
    assert selected.tolist() == [1, 2]


def test_normalized_depth_selection_rejects_invalid_layers():
    with pytest.raises(ValueError, match="must lie"):
        select_normalized_depth_layers([4], n_layers=4)


def test_discovery_confirmation_split_is_balanced_stable_and_salted():
    keys = [f"probe-{index}" for index in range(9)]
    first = deterministic_balanced_split(keys, salt="domain-a")
    repeated = deterministic_balanced_split(list(reversed(keys)), salt="domain-a")
    other = deterministic_balanced_split(keys, salt="domain-b")
    assert first == repeated
    assert list(first.values()).count("discovery") == 5
    assert list(first.values()).count("confirmation") == 4
    assert first != other


def test_exact_sign_test_and_bh_correction():
    assert one_sided_sign_test(10, 0, alternative="positive") == pytest.approx(1 / 1024)
    assert one_sided_sign_test(0, 10, alternative="negative") == pytest.approx(1 / 1024)
    assert one_sided_sign_test(5, 5, alternative="positive") > 0.5
    q_values = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
    assert q_values == pytest.approx([0.02, 0.04, 0.04, 0.008])


def test_direction_alignment_recovers_same_and_opposite_directions():
    delta = np.array([[1.0, 0.0, -1.0], [0.5, -0.5, 0.0]])
    assert weighted_direction_alignment(delta, delta) == pytest.approx(1.0)
    assert weighted_direction_alignment(delta, -delta) == pytest.approx(-1.0)
