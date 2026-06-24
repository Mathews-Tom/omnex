"""Deterministic rank fusion over unit ids.

Fusion is intentionally boring infrastructure: reciprocal rank fusion (RRF) over
ordered id lists and relative-score fusion (RSF) over scored id lists. The kernel
calls :func:`combine` to merge retrieval lanes; in the T0 floor there is a single
lexical lane, so fusion is an explicit passthrough.

Every function is total and order-stable: results are sorted by descending fused
score with ties broken by ascending unit id, so identical inputs always yield
identical output. No model load, network, or file-system access.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_DEFAULT_RRF_K = 60


def _dedupe_best_rank(ranking: Sequence[str]) -> dict[str, int]:
    """Map each id to its best (smallest) 1-based rank in ``ranking``."""
    best: dict[str, int] = {}
    for position, unit_id in enumerate(ranking, start=1):
        if unit_id not in best:
            best[unit_id] = position
    return best


def rrf(rankings: Sequence[Sequence[str]], k: int = _DEFAULT_RRF_K) -> list[str]:
    """Reciprocal rank fusion of ordered id ``rankings``.

    Each id contributes ``1 / (k + rank)`` per ranking it appears in, using its
    best rank within that ranking. ``k`` damps the influence of low ranks and
    must be positive. The fused order sorts by descending total score, breaking
    ties by ascending unit id. A single ranking passes through unchanged (its
    scores strictly decrease with rank), and empty rankings contribute nothing.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")
    terms: dict[str, list[float]] = {}
    for ranking in rankings:
        for unit_id, rank in _dedupe_best_rank(ranking).items():
            terms.setdefault(unit_id, []).append(1.0 / (k + rank))
    # math.fsum is order-insensitive and exact, so ids that share the same
    # multiset of rank terms score bit-identically and the ascending-id
    # tie-break holds regardless of lane order or count.
    scores = {unit_id: math.fsum(unit_terms) for unit_id, unit_terms in terms.items()}
    return sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))
