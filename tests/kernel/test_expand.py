"""Tests for T0 bounded graph expansion: budgets, decay, determinism, guards."""

from __future__ import annotations

import pytest

from omnex.ir.graph import StructureGraph, build_graph
from omnex.ir.types import Reference, Span, Unit
from omnex.kernel.expand import graph_expand


def _unit(uid: str) -> Unit:
    return Unit(
        id=uid,
        document_id="doc:1",
        span=Span(0, 1),
        text=uid,
        token_count=1,
        title=None,
        breadcrumb=(),
        kind="SECTION",
        summary=None,
        protect=False,
    )


def _ref(src: str, tgt: str, kind: str = "CONTAINS", confidence: float = 1.0) -> Reference:
    return Reference(
        source_id=src,
        target_id=tgt,
        kind=kind,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=(),
    )


def _graph() -> StructureGraph:
    # a -CONTAINS-> b -CONTAINS-> c ; a -REFERENCES-> d -REFERENCES-> e
    units = [_unit(x) for x in ("a", "b", "c", "d", "e")]
    refs = [
        _ref("a", "b"),
        _ref("b", "c"),
        _ref("a", "d", "REFERENCES"),
        _ref("d", "e", "REFERENCES"),
    ]
    return build_graph(units, refs)


def test_expansion_honors_per_kind_hop_budget() -> None:
    graph = _graph()
    one_hop = {(h.unit_id, h.depth) for h in graph_expand(["a"], graph, {"CONTAINS": 1})}
    assert one_hop == {("a", 0), ("b", 1)}  # c is two CONTAINS hops away: excluded

    two_hop = {(h.unit_id, h.depth) for h in graph_expand(["a"], graph, {"CONTAINS": 2})}
    assert two_hop == {("a", 0), ("b", 1), ("c", 2)}


def test_distinct_kinds_use_independent_budgets() -> None:
    graph = _graph()
    reached = {
        (h.unit_id, h.depth) for h in graph_expand(["a"], graph, {"CONTAINS": 2, "REFERENCES": 2})
    }
    assert reached == {("a", 0), ("b", 1), ("c", 2), ("d", 1), ("e", 2)}


def test_unbudgeted_kind_is_not_followed() -> None:
    graph = _graph()
    # Only REFERENCES budgeted: the CONTAINS chain (b, c) must not be followed.
    reached = {h.unit_id for h in graph_expand(["a"], graph, {"REFERENCES": 2})}
    assert reached == {"a", "d", "e"}


def test_decay_lowers_far_neighbor_weight() -> None:
    graph = _graph()
    confidence = {
        h.unit_id: h.confidence
        for h in graph_expand(["a"], graph, {"CONTAINS": 2}, confidence_decay=0.5)
    }
    assert confidence["a"] == 1.0  # seed
    assert confidence["b"] == pytest.approx(0.5)  # one hop
    assert confidence["c"] == pytest.approx(0.25)  # two hops
    assert confidence["c"] < confidence["b"] < confidence["a"]


def test_expansion_is_order_independent() -> None:
    units = [_unit(x) for x in ("a", "b", "c", "d", "e")]
    refs = [
        _ref("a", "b"),
        _ref("b", "c"),
        _ref("a", "d", "REFERENCES"),
        _ref("d", "e", "REFERENCES"),
    ]
    budgets = {"CONTAINS": 2, "REFERENCES": 2}
    # Output must depend only on the IR, not on the order units/references were
    # supplied: a graph built from reversed inputs yields identical Hops.
    forward = graph_expand(["a"], build_graph(units, refs), budgets, confidence_decay=0.9)
    reordered = graph_expand(
        ["a"], build_graph(units[::-1], refs[::-1]), budgets, confidence_decay=0.9
    )
    assert forward == reordered


def test_empty_seeds_expand_to_nothing() -> None:
    assert graph_expand([], _graph(), {"CONTAINS": 1}) == []


def test_empty_budget_yields_only_seeds() -> None:
    # No budgeted kind: nothing is followed, only seeds return at depth 0.
    result = [(h.unit_id, h.depth, h.confidence) for h in graph_expand(["a"], _graph(), {})]
    assert result == [("a", 0, 1.0)]


def test_missing_seed_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        graph_expand(["nope"], _graph(), {"CONTAINS": 1})


def test_bare_str_seed_ids_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="sequence of ids"):
        graph_expand("ab", _graph(), {"CONTAINS": 1})


def test_rejects_decay_outside_unit_interval() -> None:
    graph = _graph()
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="confidence_decay"):
            graph_expand(["a"], graph, {"CONTAINS": 1}, confidence_decay=bad)


def test_rejects_negative_hop_budget() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        graph_expand(["a"], _graph(), {"CONTAINS": -1})
