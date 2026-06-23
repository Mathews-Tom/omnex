"""Tests for the StructureGraph build, neighbors, and traversal primitive."""

from __future__ import annotations

import pytest

from omnex.ir.graph import build_graph
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
