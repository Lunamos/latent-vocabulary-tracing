import pytest

torch = pytest.importorskip("torch")

from latent_vocabulary_tracing.torch_readout import (  # noqa: E402
    CELL_AA,
    CELL_AB,
    CELL_BA,
    CELL_BB,
    aggregate_category_statistics,
    aggregate_category_statistics_by_spans,
    aggregate_position_metrics_by_spans,
    category_statistics,
    cell_distributions,
    four_cell_logits,
    four_cell_scalar_metrics,
    matched_faithfulness,
    matched_readout_diagnostics,
    net_token_direction_scores,
    pair_metrics,
    vocabulary_edit_statistics,
)


def test_pair_metrics_reports_union_support_coverage_in_fp32():
    parent = torch.log(torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float32))
    descendant = torch.log(torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32))
    metrics = pair_metrics(parent, descendant, jaccard_k=1, support_k=1)
    assert metrics["jaccard"].item() == pytest.approx(0.0)
    assert metrics["support_mass_a"].item() == pytest.approx(0.7)
    assert metrics["support_mass_b"].item() == pytest.approx(0.7)
    assert metrics["outside_mass_a"].item() == pytest.approx(0.3)
    assert metrics["val_a"].dtype == torch.float32
    assert metrics["lse_a"].dtype == torch.float32
    assert metrics["delta_logp_at_a"].dtype == torch.float32
    assert metrics["delta_logp_at_a"].item() == pytest.approx(torch.log(torch.tensor(1 / 3)))


def test_four_cell_contrasts_have_unambiguous_semantics():
    cells = {
        CELL_AA: torch.tensor([[2.0, 0.0]]),
        CELL_BA: torch.tensor([[0.0, 2.0]]),
        CELL_AB: torch.tensor([[1.0, 0.0]]),
        CELL_BB: torch.tensor([[0.0, 1.0]]),
    }
    metrics = four_cell_scalar_metrics(cells)
    assert set(metrics) == {
        "state_parent_decoder",
        "state_descendant_decoder",
        "decoder_parent_state",
        "decoder_descendant_state",
        "native_total",
    }
    assert metrics["state_parent_decoder"]["kl_ba"].item() > 0
    assert metrics["decoder_parent_state"]["kl_ba"].item() > 0


def test_four_cell_logits_applies_each_decoder_to_each_state():
    state_a = torch.tensor([[1.0, 2.0]])
    state_b = torch.tensor([[3.0, 4.0]])
    cells = four_cell_logits(
        state_a,
        state_b,
        lambda state: state + 10,
        lambda state: state * 2,
    )
    assert torch.equal(cells[CELL_AA], torch.tensor([[11.0, 12.0]]))
    assert torch.equal(cells[CELL_BA], torch.tensor([[13.0, 14.0]]))
    assert torch.equal(cells[CELL_AB], torch.tensor([[2.0, 4.0]]))
    assert torch.equal(cells[CELL_BB], torch.tensor([[6.0, 8.0]]))


def test_faithfulness_never_mixes_native_and_anchored_descendant_logits():
    final = {
        CELL_AA: torch.tensor([[2.0, 0.0]]),
        CELL_BA: torch.tensor([[0.0, 2.0]]),
        CELL_BB: torch.tensor([[1.0, 1.0]]),
    }
    readout = {
        CELL_AA: final[CELL_AA].clone(),
        CELL_BA: final[CELL_BA].clone(),
        CELL_BB: final[CELL_BB].clone(),
    }
    faith = matched_faithfulness(final, readout)
    assert faith["native_output_decoder"]["b"].item() == pytest.approx(0.0, abs=1e-7)
    assert faith["parent_anchored"]["b"].item() == pytest.approx(0.0, abs=1e-7)

    # Perturbing only the native output-decoder readout changes its
    # faithfulness but cannot leak into the parent-anchored number.
    readout[CELL_BB] = torch.tensor([[4.0, 0.0]])
    changed = matched_faithfulness(final, readout)
    assert changed["native_output_decoder"]["b"].item() > 0
    assert changed["parent_anchored"]["b"].item() == pytest.approx(0.0, abs=1e-7)


def test_readout_diagnostics_use_matched_coordinates_and_objective_junk_mask():
    final = {
        CELL_AA: torch.tensor([[3.0, 2.0, 0.0]]),
        CELL_BA: torch.tensor([[0.0, 3.0, 2.0]]),
        CELL_BB: torch.tensor([[3.0, 0.0, 2.0]]),
    }
    readout = {name: value.clone() for name, value in final.items()}
    readout[CELL_BB] = torch.tensor([[0.0, 2.0, 3.0]])
    malformed = torch.tensor([False, True, False])

    diagnostics = matched_readout_diagnostics(final, readout, malformed, topk=1)
    assert diagnostics["parent_anchored"]["b"]["faith_kl"].item() == pytest.approx(0.0, abs=1e-7)
    assert diagnostics["parent_anchored"]["b"]["topk_jaccard"].item() == 1.0
    assert diagnostics["native_output_decoder"]["b"]["faith_kl"].item() > 0
    assert diagnostics["native_output_decoder"]["b"]["topk_jaccard"].item() == 0.0
    expected_mass = torch.softmax(readout[CELL_BB], dim=-1)[0, 1].item()
    assert diagnostics["native_output_decoder"]["b"][
        "malformed_token_mass"
    ].item() == pytest.approx(expected_mass)


def test_torch_category_statistics_expose_cancelled_turnover():
    parent = torch.log(torch.tensor([[0.4, 0.2, 0.3, 0.1]]))
    descendant = torch.log(torch.tensor([[0.25, 0.35, 0.1, 0.3]]))
    stats = category_statistics(
        parent,
        descendant,
        torch.tensor([0, 0, 1, 1]),
        n_categories=2,
    )
    assert stats["signed"][0].cpu().numpy() == pytest.approx([0.0, 0.0], abs=1e-7)
    assert stats["turnover"][0].cpu().numpy() == pytest.approx([0.15, 0.20])
    assert stats["composition"].sum().item() == pytest.approx(1.0)
    assert stats["turnover_enrichment_bits"][0].cpu().numpy() == pytest.approx(
        torch.log2(torch.tensor([5 / 7, 10 / 7])).numpy()
    )


def test_category_aggregation_derives_ratios_after_averaging_primitive_masses():
    parent = torch.log(torch.tensor([[0.3, 0.2, 0.3, 0.2], [0.3, 0.2, 0.3, 0.2]]))
    descendant = torch.log(
        torch.tensor([[0.2, 0.3, 0.3, 0.2], [0.3, 0.2, 0.0, 0.5]]).clamp_min(1e-12)
    )
    statistics = category_statistics(
        parent,
        descendant,
        torch.tensor([0, 0, 1, 1]),
        n_categories=2,
    )
    aggregate = aggregate_category_statistics(statistics)
    assert statistics["composition"].mean(dim=0).tolist() == pytest.approx([0.5, 0.5])
    assert aggregate["composition"].tolist() == pytest.approx([0.25, 0.75])


def test_role_aggregation_weights_positions_then_returns_probe_vectors():
    parent = torch.log(torch.full((4, 4), 0.25))
    descendant = torch.log(
        torch.tensor(
            [
                [0.20, 0.30, 0.25, 0.25],
                [0.15, 0.35, 0.25, 0.25],
                [0.25, 0.25, 0.10, 0.40],
                [0.25, 0.25, 0.05, 0.45],
            ]
        )
    )
    statistics = category_statistics(
        parent,
        descendant,
        torch.tensor([0, 0, 1, 1]),
        n_categories=2,
    )
    roles = aggregate_category_statistics_by_spans(
        statistics,
        {"observation": [[0, 2]], "action": [[2, 3], [3, 9]]},
    )
    assert roles["observation"]["turnover"].tolist() == pytest.approx([0.075, 0.0])
    assert roles["action"]["turnover"].tolist() == pytest.approx([0.0, 0.175])


def test_position_metric_role_aggregation_handles_disjoint_spans():
    roles = aggregate_position_metrics_by_spans(
        {"kl_ba": torch.tensor([1.0, 3.0, 5.0, 7.0]), "js": torch.ones(4)},
        {"observation": [[0, 2]], "action": [[2, 3], [3, 9]]},
        fields=("kl_ba", "js"),
    )
    assert roles["observation"]["kl_ba"].item() == pytest.approx(2.0)
    assert roles["action"]["kl_ba"].item() == pytest.approx(6.0)


def test_full_vocabulary_edit_exposes_newly_promoted_tokens():
    parent = torch.log(torch.tensor([[0.70, 0.20, 0.09, 0.01]]))
    descendant = torch.log(torch.tensor([[0.45, 0.20, 0.09, 0.26]]))
    categories, delta = vocabulary_edit_statistics(
        parent,
        descendant,
        torch.tensor([0, 0, 1, 1]),
        n_categories=2,
    )
    assert delta.argmax().item() == 3
    assert delta[0, 3].item() == pytest.approx(0.25)
    assert categories["promoted"][0, 1].item() == pytest.approx(0.25)


def test_exact_token_directions_use_net_change_after_averaging():
    signed = torch.tensor([0.30, -0.20, 0.0, -0.10])
    promoted, suppressed = net_token_direction_scores(signed)
    assert promoted.tolist() == pytest.approx([0.30, 0.0, 0.0, 0.0])
    assert suppressed.tolist() == pytest.approx([0.0, 0.20, 0.0, 0.10])
    assert not torch.logical_and(promoted > 0, suppressed > 0).any()


def test_cached_distributions_reproduce_all_metrics():
    cells = {
        CELL_AA: torch.tensor([[2.0, 0.0, -1.0]]),
        CELL_BA: torch.tensor([[0.0, 2.0, -1.0]]),
        CELL_AB: torch.tensor([[1.5, 0.0, -0.5]]),
        CELL_BB: torch.tensor([[0.0, 1.5, -0.5]]),
    }
    distributions = cell_distributions(cells)
    fresh = four_cell_scalar_metrics(cells)
    cached = four_cell_scalar_metrics(cells, distributions=distributions)
    for contrast in fresh:
        for metric in fresh[contrast]:
            assert cached[contrast][metric] == pytest.approx(fresh[contrast][metric])

    pair = pair_metrics(
        cells[CELL_AA],
        cells[CELL_BA],
        jaccard_k=1,
        support_k=2,
        parent_distribution=distributions[CELL_AA],
        descendant_distribution=distributions[CELL_BA],
    )
    assert pair["kl_ba"] == pytest.approx(fresh["state_parent_decoder"]["kl_ba"])
