"""Tests for deterministic rank fusion: RRF, RSF, and the combine entry point."""

from __future__ import annotations

import pytest

from omnex.kernel.fusion import combine, rrf, rsf


def _ranked(*pairs: tuple[str, int]) -> list[str]:
    """Build a lane placing each ``(id, rank)`` at its 1-based rank with fillers."""
    size = max(rank for _, rank in pairs)
    lane = [f"_pad{i}" for i in range(1, size + 1)]
    for unit_id, rank in pairs:
        lane[rank - 1] = unit_id
    return lane


def test_rrf_fuses_two_rankings_with_id_tiebreak() -> None:
    # a and b tie on score (rank 1+2 vs 2+1); c and d tie (rank 3 each).
    # Ties break by ascending id, so the order is fully determined.
    assert rrf([["a", "b", "c"], ["b", "a", "d"]]) == ["a", "b", "c", "d"]


def test_rrf_tiebreak_holds_across_three_or_more_lanes() -> None:
    # "a" and "b" share the rank multiset {5, 15, 17} across these lanes, so
    # their RRF scores are mathematically equal. Naive left-to-right float
    # summation can make them differ by 1 ULP and override the id tie-break;
    # an order-insensitive sum must keep "a" (smaller id) ahead of "b".
    lanes = [
        _ranked(("a", 17), ("b", 15)),
        _ranked(("a", 5)),
        _ranked(("b", 5)),
        _ranked(("a", 15), ("b", 17)),
    ]
    order = rrf(lanes)
    assert order.index("a") < order.index("b")


def test_rrf_single_ranking_passes_through_unchanged() -> None:
    assert rrf([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_is_order_stable_across_lane_order() -> None:
    a = rrf([["a", "b", "c"], ["b", "a", "d"]])
    b = rrf([["b", "a", "d"], ["a", "b", "c"]])
    assert a == b


def test_rrf_dedupes_to_best_rank_within_a_lane() -> None:
    # "a" appears twice; only its best (first) rank counts, so it stays ahead.
    assert rrf([["a", "b", "a"]]) == ["a", "b"]


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        rrf([["a"]], k=0)


def test_rsf_normalizes_and_sums_lanes() -> None:
    # lane1 normalized: a=1.0, b=0.5, c=0.0; lane2 normalized: c=1.0, a=0.0.
    # totals: a=1.0, c=1.0, b=0.5 -> a before c by id tiebreak, then b.
    assert rsf([[("a", 10.0), ("b", 5.0), ("c", 0.0)], [("c", 8.0), ("a", 4.0)]]) == ["a", "c", "b"]


def test_rsf_single_lane_passes_through_in_score_order() -> None:
    assert rsf([[("x", 3.0), ("y", 2.0), ("z", 1.0)]]) == ["x", "y", "z"]


def test_rsf_all_equal_lane_normalizes_to_one() -> None:
    # Equal scores normalize to 1.0 each; order falls back to id tiebreak.
    assert rsf([[("b", 2.0), ("a", 2.0)]]) == ["a", "b"]


def test_combine_single_lane_is_exact_passthrough() -> None:
    assert combine([["p", "q", "r"]]) == ["p", "q", "r"]


def test_combine_drops_empty_lanes() -> None:
    assert combine([[], ["a", "b"]]) == ["a", "b"]
    assert combine([[], []]) == []


def test_combine_multiple_lanes_uses_rrf() -> None:
    assert combine([["a", "b", "c"], ["b", "a", "d"]]) == rrf([["a", "b", "c"], ["b", "a", "d"]])


def test_rrf_empty_input_is_empty() -> None:
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rsf_empty_input_is_empty() -> None:
    assert rsf([]) == []
    assert rsf([[], []]) == []


def test_rsf_is_order_stable_across_lane_order() -> None:
    lane_a = [("a", 2.0), ("c", 1.0)]
    lane_b = [("b", 3.0), ("c", 1.0)]
    assert rsf([lane_a, lane_b]) == rsf([lane_b, lane_a])


def test_rsf_first_score_wins_for_duplicate_id_in_lane() -> None:
    # If the later "a" score (0.0) won, min-max would rank a last; first-wins
    # keeps a at the top of the lane.
    assert rsf([[("a", 10.0), ("a", 0.0), ("b", 5.0)]]) == ["a", "b"]


def test_rsf_handles_negative_scores() -> None:
    assert rsf([[("a", -1.0), ("b", -5.0)]]) == ["a", "b"]


def test_combine_single_lane_dedupes_duplicate_ids() -> None:
    assert combine([["a", "b", "a", "c"]]) == ["a", "b", "c"]
