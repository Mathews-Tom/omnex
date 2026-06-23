"""Tests for the StructureGraph build, neighbors, and traversal primitive."""

from __future__ import annotations

import itertools

import pytest

from omnex.ir.graph import Hop, build_graph
from omnex.ir.types import Reference, Span, Unit, UnitKind


def _unit(uid: str, kind: UnitKind = "SECTION") -> Unit:
    return Unit(
        id=uid,
        document_id="doc:1",
        span=Span(0, 1),
        text=uid,
        token_count=1,
        title=None,
        breadcrumb=(),
        kind=kind,
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


# --- build ---


def test_build_graph_nodes_and_edges() -> None:
    units = [_unit("a"), _unit("b"), _unit("c")]
    refs = [_ref("a", "b"), _ref("b", "c", kind="REFERENCES")]
    graph = build_graph(units, refs)
    assert len(graph) == 3
    assert graph.nodes() == ["a", "b", "c"]
    assert "a" in graph
    assert graph.unit("a").id == "a"


def test_build_graph_groups_multiple_kinds_on_one_edge() -> None:
    units = [_unit("a"), _unit("b")]
    refs = [_ref("a", "b", kind="CONTAINS"), _ref("a", "b", kind="REFERENCES")]
    graph = build_graph(units, refs)
    assert graph.edge_kinds("a", "b") == frozenset({"CONTAINS", "REFERENCES"})


def test_build_graph_rejects_dangling_reference() -> None:
    with pytest.raises(ValueError, match="target not in units"):
        build_graph([_unit("a")], [_ref("a", "missing")])


def test_build_graph_rejects_conflicting_duplicate_unit() -> None:
    dup = Unit(
        id="a",
        document_id="doc:1",
        span=Span(0, 1),
        text="different",
        token_count=2,
        title=None,
        breadcrumb=(),
        kind="SECTION",
        summary=None,
        protect=False,
    )
    with pytest.raises(ValueError, match="duplicate unit id"):
        build_graph([_unit("a"), dup], [])


def test_unit_lookup_missing_raises() -> None:
    graph = build_graph([_unit("a")], [])
    with pytest.raises(KeyError):
        graph.unit("nope")


# --- neighbors ---


def test_neighbors_filters_by_kind() -> None:
    units = [_unit("a"), _unit("b"), _unit("c")]
    refs = [_ref("a", "b", kind="CONTAINS"), _ref("a", "c", kind="REFERENCES")]
    graph = build_graph(units, refs)
    assert graph.neighbors("a") == ["b", "c"]
    assert graph.neighbors("a", {"CONTAINS"}) == ["b"]
    assert graph.neighbors("a", {"REFERENCES"}) == ["c"]


def test_neighbors_direction() -> None:
    units = [_unit("a"), _unit("b")]
    graph = build_graph(units, [_ref("a", "b")])
    assert graph.neighbors("a", direction="out") == ["b"]
    assert graph.neighbors("a", direction="in") == []
    assert graph.neighbors("b", direction="in") == ["a"]
    assert graph.neighbors("b", direction="both") == ["a"]


# --- traverse ---


def test_traverse_includes_seed_at_depth_zero() -> None:
    graph = build_graph([_unit("a")], [])
    assert graph.traverse(["a"], kinds=None, hop_budget={}) == [Hop("a", 0, 1.0)]


def test_traverse_empty_seeds_returns_empty() -> None:
    graph = build_graph([_unit("a")], [])
    assert graph.traverse([], None, {"CONTAINS": 1}) == []


def test_traverse_missing_seed_raises() -> None:
    graph = build_graph([_unit("a")], [])
    with pytest.raises(KeyError):
        graph.traverse(["ghost"], None, {"CONTAINS": 1})


def test_traverse_respects_per_kind_hop_budget() -> None:
    # a -> b -> c -> d, all CONTAINS edges.
    units = [_unit(x) for x in ("a", "b", "c", "d")]
    refs = [_ref("a", "b"), _ref("b", "c"), _ref("c", "d")]
    graph = build_graph(units, refs)
    reached = [h.unit_id for h in graph.traverse(["a"], None, {"CONTAINS": 2})]
    assert reached == ["a", "b", "c"]  # d is one hop past the budget
    full = [h.unit_id for h in graph.traverse(["a"], None, {"CONTAINS": 3})]
    assert full == ["a", "b", "c", "d"]


def test_traverse_kind_filter_excludes_unlisted_kinds() -> None:
    units = [_unit("a"), _unit("b"), _unit("c")]
    refs = [_ref("a", "b", kind="CONTAINS"), _ref("a", "c", kind="REFERENCES")]
    graph = build_graph(units, refs)
    reached = [
        h.unit_id for h in graph.traverse(["a"], {"CONTAINS"}, {"CONTAINS": 5, "REFERENCES": 5})
    ]
    assert reached == ["a", "b"]


def test_traverse_dedups_shared_target() -> None:
    # Diamond: a -> b, a -> c, b -> d, c -> d. d must appear exactly once.
    units = [_unit(x) for x in ("a", "b", "c", "d")]
    refs = [_ref("a", "b"), _ref("a", "c"), _ref("b", "d"), _ref("c", "d")]
    graph = build_graph(units, refs)
    reached = [h.unit_id for h in graph.traverse(["a"], None, {"CONTAINS": 5})]
    assert reached.count("d") == 1
    assert sorted(reached) == ["a", "b", "c", "d"]


def test_traverse_applies_confidence_decay() -> None:
    units = [_unit("a"), _unit("b")]
    graph = build_graph(units, [_ref("a", "b", confidence=0.8)])
    hops = {h.unit_id: h for h in graph.traverse(["a"], None, {"CONTAINS": 1}, 0.5)}
    assert hops["a"].confidence == 1.0
    assert hops["b"].confidence == pytest.approx(0.5 * 0.8)


def test_traverse_is_complete_under_multi_kind_budget() -> None:
    # T is reachable only via seed -> B -> P -> T, which spends CONTAINS:2 and
    # REFERENCES:1; the shorter seed -> A -> P path spends the single REFERENCES
    # before reaching T. A first-reach-only BFS would wrongly drop T.
    units = [_unit(x) for x in ("seed", "A", "B", "P", "T")]
    refs = [
        _ref("seed", "A", kind="REFERENCES"),
        _ref("seed", "B", kind="CONTAINS"),
        _ref("A", "P", kind="CONTAINS"),
        _ref("B", "P", kind="CONTAINS"),
        _ref("P", "T", kind="REFERENCES"),
    ]
    graph = build_graph(units, refs)
    reached = {h.unit_id for h in graph.traverse(["seed"], None, {"CONTAINS": 2, "REFERENCES": 1})}
    assert reached == {"seed", "A", "B", "P", "T"}


def test_traverse_branches_over_multi_kind_edge() -> None:
    # a -> b carries both CONTAINS and REFERENCES; b -> c is CONTAINS. With only
    # one CONTAINS hop, c is reachable only by spending REFERENCES on a -> b.
    units = [_unit(x) for x in ("a", "b", "c")]
    refs = [
        _ref("a", "b", kind="CONTAINS"),
        _ref("a", "b", kind="REFERENCES"),
        _ref("b", "c", kind="CONTAINS"),
    ]
    graph = build_graph(units, refs)
    reached = {h.unit_id for h in graph.traverse(["a"], None, {"CONTAINS": 1, "REFERENCES": 5})}
    assert reached == {"a", "b", "c"}


def test_traverse_terminates_on_cycles_and_self_loops() -> None:
    units = [_unit("a"), _unit("b")]
    refs = [_ref("a", "b"), _ref("b", "a"), _ref("a", "a")]
    graph = build_graph(units, refs)
    reached = [h.unit_id for h in graph.traverse(["a"], None, {"CONTAINS": 5})]
    assert sorted(reached) == ["a", "b"]


# --- determinism ---


def test_edges_are_canonically_ordered() -> None:
    units = [_unit(x) for x in ("c", "a", "b")]
    refs = [_ref("b", "c"), _ref("a", "b", kind="REFERENCES"), _ref("a", "b")]
    graph = build_graph(units, refs)
    assert graph.edges() == [
        ("a", "b", "CONTAINS"),
        ("a", "b", "REFERENCES"),
        ("b", "c", "CONTAINS"),
    ]


def test_build_is_order_invariant() -> None:
    units = [_unit(x) for x in ("a", "b", "c", "d")]
    refs = [_ref("a", "b"), _ref("a", "c"), _ref("b", "d"), _ref("c", "d")]
    canonical = build_graph(units, refs)
    for unit_perm in itertools.permutations(units):
        for ref_perm in (refs, list(reversed(refs))):
            graph = build_graph(list(unit_perm), ref_perm)
            assert graph.nodes() == canonical.nodes()
            assert graph.edges() == canonical.edges()


def test_traverse_output_is_byte_identical_across_input_order() -> None:
    units = [_unit(x) for x in ("a", "b", "c", "d", "e")]
    refs = [
        _ref("a", "b"),
        _ref("a", "c"),
        _ref("b", "d"),
        _ref("c", "d"),
        _ref("d", "e"),
    ]
    budget = {"CONTAINS": 5}
    canonical = repr(build_graph(units, refs).traverse(["a"], None, budget))
    for ref_perm in itertools.permutations(refs):
        graph = build_graph(list(reversed(units)), list(ref_perm))
        assert repr(graph.traverse(["a"], None, budget)) == canonical
