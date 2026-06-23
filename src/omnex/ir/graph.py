"""StructureGraph: a deterministic directed graph over the IR.

Wraps ``networkx.DiGraph``. Nodes are unit ids carrying their ``Unit``; each
edge aggregates the typed ``Reference`` values between an ordered
``(source_id, target_id)`` pair plus the frozenset of their kinds. Construction
is fail-loud: a reference naming a unit absent from the input is rejected, so
the graph never fabricates structure.

All iteration is canonically ordered (unit ids sorted, edges grouped and
sorted), so the same IR always yields byte-identical output regardless of the
order units and references were supplied.

This module imports ``networkx`` but performs no model load, network, or
file-system access on import.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import networkx as nx

from omnex.ir.types import Reference, Unit


class StructureGraph:
    """A deterministic directed graph over IR units and references."""

    __slots__ = ("_g",)

    def __init__(self, graph: nx.DiGraph | None = None) -> None:
        self._g = graph if graph is not None else nx.DiGraph()

    def __len__(self) -> int:
        return int(self._g.number_of_nodes())

    def __contains__(self, unit_id: object) -> bool:
        return bool(self._g.has_node(unit_id))

    def nodes(self) -> list[str]:
        """All unit ids in canonical (sorted) order."""
        return sorted(self._g.nodes)

    def unit(self, unit_id: str) -> Unit:
        """Return the ``Unit`` stored at ``unit_id``."""
        if unit_id not in self._g:
            raise KeyError(unit_id)
        return cast(Unit, self._g.nodes[unit_id]["unit"])

    def edge_kinds(self, source: str, target: str) -> frozenset[str]:
        """Return the set of reference kinds on the edge ``source -> target``."""
        return cast("frozenset[str]", self._g.edges[source, target]["kinds"])


def build_graph(units: Iterable[Unit], references: Iterable[Reference]) -> StructureGraph:
    """Build a ``StructureGraph`` from IR units and references.

    Nodes are inserted in canonical (id-sorted) order. References are grouped by
    ``(source_id, target_id)`` and stored as a deterministically ordered tuple on
    the edge, alongside the frozenset of edge kinds. Raises ``ValueError`` if a
    reference names a unit absent from ``units``, or if two units share an id but
    differ in content.
    """
    units_by_id: dict[str, Unit] = {}
    for unit in units:
        existing = units_by_id.get(unit.id)
        if existing is not None and existing != unit:
            raise ValueError(f"duplicate unit id with differing content: {unit.id}")
        units_by_id[unit.id] = unit

    g = nx.DiGraph()
    for uid in sorted(units_by_id):
        g.add_node(uid, unit=units_by_id[uid])

    grouped: dict[tuple[str, str], list[Reference]] = {}
    for ref in references:
        if ref.source_id not in units_by_id:
            raise ValueError(f"reference source not in units: {ref.source_id}")
        if ref.target_id not in units_by_id:
            raise ValueError(f"reference target not in units: {ref.target_id}")
        grouped.setdefault((ref.source_id, ref.target_id), []).append(ref)

    for pair in sorted(grouped):
        refs = tuple(sorted(grouped[pair], key=lambda r: (r.kind, -r.confidence, r.evidence)))
        kinds = frozenset(r.kind for r in refs)
        g.add_edge(pair[0], pair[1], references=refs, kinds=kinds)

    return StructureGraph(g)
