"""StructureGraph: a deterministic directed graph over the IR.

Wraps ``networkx.DiGraph``. Nodes are unit ids carrying their ``Unit``; each
edge aggregates the typed ``Reference`` values between an ordered
``(source_id, target_id)`` pair plus the frozenset of their kinds. Construction
is fail-loud: a reference naming a unit absent from the input is rejected, so
the graph never fabricates structure.

All iteration is canonically ordered (unit ids sorted, edges grouped and
sorted), so the same IR always yields byte-identical output regardless of the
order units and references were supplied.

``traverse`` is the shared basis for later T0 bounded expansion and T1 closure.
It returns every unit reachable under the per-kind hop budgets (complete: a unit
is returned iff some budget-respecting path exists), deduplicated by id. It stays
complete and terminating even with multiple kinds and cycles by tracking, per
node, the non-dominated budget vectors already expanded.

This module imports ``networkx`` but performs no model load, network, or
file-system access on import.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import networkx as nx

from omnex.ir.types import Reference, Unit

Direction = Literal["out", "in", "both"]


@dataclass(frozen=True, slots=True)
class Hop:
    """A reached unit in a traversal: its id, hop depth, and decayed confidence.

    ``depth`` and ``confidence`` reflect the first (breadth-first, then
    lowest-sorted kind) path by which the unit was reached, not necessarily the
    maximum-confidence path; relevance ranking is the kernel's job, not the
    traversal primitive's.
    """

    unit_id: str
    depth: int
    confidence: float


@dataclass(frozen=True, slots=True)
class _Frontier:
    """Internal BFS frontier entry tracking per-kind hop budget consumed so far."""

    unit_id: str
    depth: int
    confidence: float
    used: tuple[tuple[str, int], ...]


def _dominates(left: Mapping[str, int], right: Mapping[str, int]) -> bool:
    """Return True if ``left`` spends no more than ``right`` on every kind.

    A dominating (more budget-efficient) vector can reach everything the
    dominated one can, so a dominated frontier state is never worth expanding.
    Equal vectors dominate each other, which is what prevents re-expanding an
    identical state and guarantees termination.
    """
    keys = set(left) | set(right)
    return all(left.get(kind, 0) <= right.get(kind, 0) for kind in keys)


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

    def neighbors(
        self,
        unit_id: str,
        kinds: Iterable[str] | None = None,
        *,
        direction: Direction = "out",
    ) -> list[str]:
        """Return canonically-ordered unit ids reachable from ``unit_id`` in one hop.

        ``kinds`` filters edges by reference kind (all kinds if ``None``).
        ``direction`` selects outgoing edges (``"out"``), incoming edges
        (``"in"``), or both.
        """
        if unit_id not in self._g:
            raise KeyError(unit_id)
        allowed = frozenset(kinds) if kinds is not None else None
        found: set[str] = set()
        if direction in ("out", "both"):
            for target in self._g.successors(unit_id):
                if allowed is None or self.edge_kinds(unit_id, target) & allowed:
                    found.add(target)
        if direction in ("in", "both"):
            for source in self._g.predecessors(unit_id):
                if allowed is None or self.edge_kinds(source, unit_id) & allowed:
                    found.add(source)
        return sorted(found)

    def _edge_refs(self, source: str, target: str) -> tuple[Reference, ...]:
        return cast("tuple[Reference, ...]", self._g.edges[source, target]["references"])

    def _edge_confidence(self, source: str, target: str, kind: str) -> float:
        """Strongest confidence among references of ``kind`` on the edge."""
        return max(r.confidence for r in self._edge_refs(source, target) if r.kind == kind)

    def traverse(
        self,
        seed_ids: Iterable[str],
        kinds: Iterable[str] | None,
        hop_budget: Mapping[str, int],
        confidence_decay: float = 1.0,
    ) -> list[Hop]:
        """Breadth-first typed expansion returning every budget-reachable unit.

        Follows edges whose reference kind is in ``kinds`` (defaulting to the keys
        of ``hop_budget``), consuming one unit from that kind's ``hop_budget`` per
        hop taken along a path. A kind without a ``hop_budget`` entry is treated as
        budget 0 and is never followed, so callers should keep ``kinds`` and the
        ``hop_budget`` keys aligned. An edge with several allowed kinds is explored
        once per kind, since each choice spends a different budget.

        The result is complete: a unit is returned iff some budget-respecting path
        reaches it. Each unit is emitted once (deduplicated by id), in
        deterministic breadth-first, id-sorted order; seeds are emitted at depth 0
        with confidence 1.0. Confidence decays multiplicatively by
        ``confidence_decay`` and the edge's per-kind confidence. Hop budgets are
        expected to be small.
        """
        allowed = frozenset(kinds) if kinds is not None else frozenset(hop_budget)
        seeds = sorted(set(seed_ids))
        for seed in seeds:
            if seed not in self._g:
                raise KeyError(seed)

        emitted: set[str] = set()
        results: list[Hop] = []
        frontier_seen: dict[str, list[dict[str, int]]] = {}
        queue: deque[_Frontier] = deque()

        for seed in seeds:
            emitted.add(seed)
            results.append(Hop(seed, 0, 1.0))
            frontier_seen[seed] = [{}]
            queue.append(_Frontier(seed, 0, 1.0, ()))

        while queue:
            current = queue.popleft()
            used = dict(current.used)
            for target in sorted(self._g.successors(current.unit_id)):
                for kind in sorted(self.edge_kinds(current.unit_id, target) & allowed):
                    if used.get(kind, 0) >= hop_budget.get(kind, 0):
                        continue
                    next_used = dict(used)
                    next_used[kind] = next_used.get(kind, 0) + 1
                    if any(_dominates(seen, next_used) for seen in frontier_seen.get(target, ())):
                        continue
                    frontier_seen.setdefault(target, []).append(next_used)
                    confidence = (
                        current.confidence
                        * confidence_decay
                        * self._edge_confidence(current.unit_id, target, kind)
                    )
                    if target not in emitted:
                        emitted.add(target)
                        results.append(Hop(target, current.depth + 1, confidence))
                    queue.append(
                        _Frontier(
                            target,
                            current.depth + 1,
                            confidence,
                            tuple(sorted(next_used.items())),
                        )
                    )
        return results


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
