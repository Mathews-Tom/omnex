"""T0 bounded graph expansion over the StructureGraph.

``graph_expand`` is the kernel-facing seam for the T0 retrieval tier: starting
from lexical seeds it pulls a *bounded* structural neighborhood, following typed
edges only as far as the per-kind hop budgets allow and weighting reached units
by multiplicative confidence decay. It builds directly on the generic IR
primitive ``StructureGraph.traverse``; the allowed edge kinds are exactly the
keys of ``hop_budget_by_kind``, so callers configure expansion with a single
budget mapping.

``graph_expand`` is deliberately *bounded*, not transitive. ``closure_expand`` is
the deterministic T1 reference closure: the unbounded transitive closure over a
chosen set of reference edge kinds, terminating on cycles. Both return whatever
``traverse`` gives — complete, deduplicated by id, and in deterministic order —
so the same inputs always yield byte-identical output.

No model load, network, or file-system access.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from omnex.ir.graph import Hop, StructureGraph


def graph_expand(
    seed_ids: Sequence[str],
    graph: StructureGraph,
    hop_budget_by_kind: Mapping[str, int],
    confidence_decay: float = 1.0,
) -> list[Hop]:
    """Expand ``seed_ids`` into their bounded structural neighborhood.

    Follows edges whose kind appears in ``hop_budget_by_kind``, spending one unit
    of that kind's budget per hop. Each reached unit carries its hop ``depth``
    (graph distance from the nearest seed) and a ``confidence`` that decays
    multiplicatively by ``confidence_decay`` and per-edge confidence, so far
    neighbors weigh less than near ones. Seeds are returned at depth 0 with
    confidence 1.0.

    ``confidence_decay`` must lie in ``(0.0, 1.0]`` and every hop budget must be
    non-negative; both are caller errors otherwise. A bare ``str`` for
    ``seed_ids`` raises ``TypeError`` (it must be a sequence of ids, never one
    string). Raises ``KeyError`` (from the traversal) if a seed id is absent from
    ``graph``.
    """
    if isinstance(seed_ids, str):
        raise TypeError(f"seed_ids must be a sequence of ids, not a single str: {seed_ids!r}")
    if not 0.0 < confidence_decay <= 1.0:
        raise ValueError(f"confidence_decay must be in (0.0, 1.0], got {confidence_decay}")
    for kind, budget in hop_budget_by_kind.items():
        if budget < 0:
            raise ValueError(f"hop budget for {kind!r} must be non-negative, got {budget}")
    return graph.traverse(
        seed_ids,
        kinds=None,
        hop_budget=hop_budget_by_kind,
        confidence_decay=confidence_decay,
    )


def closure_expand(
    seed_ids: Sequence[str],
    graph: StructureGraph,
    ref_kinds: Iterable[str],
    confidence_decay: float = 1.0,
) -> list[Hop]:
    """Compute the deterministic transitive closure over reference edges.

    Starting from ``seed_ids``, follow every edge whose kind is in ``ref_kinds``
    (the hard reference edges: REFERENCES, FOREIGN_KEY, IMPORTS, CALLS) to a
    fixpoint, returning every reachable unit deduplicated by id. Unlike
    ``graph_expand`` this is *unbounded*: it is expressed as a per-kind hop budget
    equal to the node count, which is at least the longest simple path, so the
    result is the complete transitive closure. Cycles terminate via the
    traversal's per-node domination and emitted-set guards.

    Seeds are returned at depth 0 with confidence 1.0; reached units carry their
    closure depth and decayed confidence. ``confidence_decay`` must lie in
    ``(0.0, 1.0]``; a bare ``str`` for ``seed_ids`` raises ``TypeError``; a seed
    absent from ``graph`` raises ``KeyError``. The output is byte-identical for
    identical inputs.
    """
    if isinstance(seed_ids, str):
        raise TypeError(f"seed_ids must be a sequence of ids, not a single str: {seed_ids!r}")
    if not 0.0 < confidence_decay <= 1.0:
        raise ValueError(f"confidence_decay must be in (0.0, 1.0], got {confidence_decay}")
    kinds = frozenset(ref_kinds)
    budget = max(len(graph), 1)
    hop_budget = dict.fromkeys(kinds, budget)
    return graph.traverse(
        seed_ids,
        kinds=kinds,
        hop_budget=hop_budget,
        confidence_decay=confidence_decay,
    )
