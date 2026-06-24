"""T0 bounded graph expansion over the StructureGraph.

``graph_expand`` is the kernel-facing seam for the T0 retrieval tier: starting
from lexical seeds it pulls a *bounded* structural neighborhood, following typed
edges only as far as the per-kind hop budgets allow and weighting reached units
by multiplicative confidence decay. It builds directly on the generic IR
primitive ``StructureGraph.traverse``; the allowed edge kinds are exactly the
keys of ``hop_budget_by_kind``, so callers configure expansion with a single
budget mapping.

This is deliberately *bounded*, not transitive: the deterministic T1 reference
closure is a separate, later seam. The result is whatever ``traverse`` returns —
complete under the budgets, deduplicated by id, and in deterministic order — so
the same seeds, graph, budgets, and decay always yield byte-identical output.

No model load, network, or file-system access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
