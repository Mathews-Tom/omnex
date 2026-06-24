"""Benchmark metric math: tokens-at-recall, F1, and p95.

These tests pin the arithmetic on hand-computed inputs, prove recall is held
equal in every token comparison, and prove the computation is deterministic.
"""

from __future__ import annotations

from omnex.bench.baselines import full_dump_baseline
from omnex.bench.metrics import (
    RetrievedItem,
    f1,
    p95_latency,
    precision,
    recall,
    tokens_at_recall,
)
from omnex.bench.report import Comparison, PathResult, render, render_report, verdict

_ABC = frozenset({"a", "b", "c"})


def _items(*pairs: tuple[int, set[str]]) -> list[RetrievedItem]:
    return [RetrievedItem(tokens, frozenset(cov)) for tokens, cov in pairs]


def test_tokens_at_recall_full_recall_sums_until_last_label() -> None:
    results = _items((10, {"a"}), (5, {"b"}), (3, {"c"}))
    assert tokens_at_recall(results, _ABC, 1.0) == 18


def test_tokens_at_recall_partial_target_rounds_up_label_count() -> None:
    results = _items((10, {"a"}), (5, {"b"}), (3, {"c"}))
    # ceil(0.5 * 3) = 2 labels -> stops after the second item.
    assert tokens_at_recall(results, _ABC, 0.5) == 15
    # ceil(1/3 * 3) = 1 label -> stops after the first item.
    assert tokens_at_recall(results, _ABC, 1.0 / 3.0) == 10


def test_tokens_at_recall_ignores_irrelevant_coverage() -> None:
    # The second item covers only a distractor, so it adds tokens but no recall;
    # the target is reached only at the third item.
    results = _items((4, {"a"}), (6, {"x"}), (2, {"b"}))
    assert tokens_at_recall(results, frozenset({"a", "b"}), 1.0) == 12


def test_tokens_at_recall_zero_target_costs_nothing() -> None:
    results = _items((10, {"a"}))
    assert tokens_at_recall(results, _ABC, 0.0) == 0


def test_tokens_at_recall_zero_target_on_empty_ranking_is_zero() -> None:
    # The pre-loop guard returns before consuming any item, so a zero target
    # costs nothing even with nothing to retrieve.
    assert tokens_at_recall([], _ABC, 0.0) == 0


def test_tokens_at_recall_returns_none_when_recall_unreached() -> None:
    # No item ever covers "c": full recall is impossible, so no token figure is
    # reported -- the caller must not compare tokens at unequal recall.
    results = _items((10, {"a"}), (5, {"b"}))
    assert tokens_at_recall(results, _ABC, 1.0) is None


def test_tokens_at_recall_is_deterministic_on_repeat() -> None:
    results = _items((7, {"a"}), (4, {"b"}), (9, {"c"}))
    first = tokens_at_recall(results, _ABC, 1.0)
    second = tokens_at_recall(results, _ABC, 1.0)
    assert first == second == 20


def test_recall_is_held_equal_across_two_rankings_before_comparing_tokens() -> None:
    # Two paths reach the SAME recall (1.0) at the target; only then are their
    # token totals comparable. The metric makes that explicit: both are non-None.
    labels = frozenset({"a", "b"})
    cheap = _items((3, {"a"}), (2, {"b"}))
    dear = _items((30, {"a"}), (20, {"b"}))
    cheap_tokens = tokens_at_recall(cheap, labels, 1.0)
    dear_tokens = tokens_at_recall(dear, labels, 1.0)
    assert cheap_tokens is not None and dear_tokens is not None  # equal recall reached
    assert recall({"a", "b"}, labels) == 1.0
    assert cheap_tokens < dear_tokens


def test_tokens_at_recall_rejects_out_of_range_target() -> None:
    results = _items((1, {"a"}))
    for bad in (-0.1, 1.1):
        try:
            tokens_at_recall(results, _ABC, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for target_recall={bad}")


def test_tokens_at_recall_rejects_empty_labels() -> None:
    try:
        tokens_at_recall(_items((1, {"a"})), frozenset(), 1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty labels")


def test_recall_precision_f1_on_known_sets() -> None:
    assert recall({"a", "b"}, {"a", "b", "c"}) == 2 / 3
    assert precision({"a", "b"}, {"a", "b", "c"}) == 1.0
    assert f1({"a", "b"}, {"a", "b", "c"}) == 0.8
    # Half the retrieval is a distractor: precision and recall both 0.5 -> F1 0.5.
    assert f1({"a", "x"}, {"a", "b"}) == 0.5


def test_recall_and_precision_handle_empty_inputs() -> None:
    assert recall(set(), set()) == 1.0  # vacuously full
    assert recall(set(), {"a"}) == 0.0
    assert precision(set(), {"a"}) == 0.0
    assert f1(set(), {"a"}) == 0.0


def test_p95_latency_nearest_rank_on_known_inputs() -> None:
    assert p95_latency([float(i) for i in range(1, 101)]) == 95.0
    assert p95_latency([0.1, 0.2, 0.3, 0.4]) == 0.4
    assert p95_latency([0.5]) == 0.5


def test_p95_latency_is_order_independent() -> None:
    forward = [float(i) for i in range(1, 21)]
    assert p95_latency(forward) == p95_latency(list(reversed(forward)))


def test_p95_latency_rejects_empty() -> None:
    try:
        p95_latency([])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty latencies")


def test_full_dump_baseline_is_the_whole_corpus_in_one_passage() -> None:
    passages = full_dump_baseline(["alpha beta", "gamma"])
    assert passages == ["alpha beta\ngamma"]


def test_report_demotes_upper_bound_and_reports_delta_only_at_equal_recall() -> None:
    comparison = Comparison(
        task="create_payment",
        recall_target=1.0,
        subject=PathResult("omnex T1", 210),
        headline=PathResult("chunk-and-embed", 917),
        upper_bound=PathResult("full-dump", 938),
    )
    text = render(comparison)
    assert "upper bound (demoted)  full-dump" in text
    assert "omnex T1 210 <= chunk-and-embed 917" in verdict(comparison)


def test_report_marks_pending_headline_and_refuses_a_delta() -> None:
    comparison = Comparison(
        task="create_payment",
        recall_target=1.0,
        subject=PathResult("omnex T1", 210),
        headline=PathResult("chunk-and-embed", None, available=False),
        upper_bound=PathResult("full-dump", 938),
    )
    assert "pending" in render(comparison)
    assert "no equal-recall delta yet" in verdict(comparison)


def test_report_refuses_a_delta_when_a_path_is_unreached() -> None:
    comparison = Comparison(
        task="dispatch_shipment",
        recall_target=1.0,
        subject=PathResult("omnex T1", 259),
        headline=PathResult("chunk-and-embed", None),  # available but unreached
        upper_bound=PathResult("full-dump", 938),
    )
    assert "unreached at this recall" in render(comparison)
    assert "no equal-recall delta" in verdict(comparison)


def test_report_states_a_loss_when_subject_spends_more_than_headline() -> None:
    # The honesty property must surface a loss, not silently render "<=".
    comparison = Comparison(
        task="coherent_closure",
        recall_target=1.0,
        subject=PathResult("omnex T1", 1000),
        headline=PathResult("chunk-and-embed", 917),
        upper_bound=PathResult("full-dump", 938),
    )
    line = verdict(comparison)
    assert "omnex T1 1000 > chunk-and-embed 917" in line
    assert "(1.09x of headline)" in line


def test_render_report_underlines_the_title_and_separates_blocks() -> None:
    comparison = Comparison(
        task="create_payment",
        recall_target=1.0,
        subject=PathResult("omnex T1", 210),
        headline=PathResult("chunk-and-embed", 917),
        upper_bound=PathResult("full-dump", 938),
    )
    text = render_report("Spec family", [comparison])
    # The underline sits directly beneath the title (no blank line between them).
    assert text.startswith("Spec family\n===========\n\n")
    assert render(comparison) in text
