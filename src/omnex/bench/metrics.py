"""Benchmark metrics: tokens-at-fixed-recall, F1, and latency percentiles.

These are the numbers the token-efficiency benchmark reports. They are pure
functions over already-graded retrieval results, so they hold no model, open no
socket, and read no file. The headline metric is :func:`tokens_at_recall`, which
makes every token comparison honest by construction: it returns the tokens a
retrieval path spends *to reach a fixed recall*, and ``None`` when a path never
reaches that recall, so a caller can refuse to report a token delta at unequal
recall.

Benchmark-only. Nothing under ``omnex.kernel`` or ``omnex.adapters`` imports this
package; the dependency runs one way, from the benchmark harness into the
product, never back.
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Set
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievedItem:
    """One ranked retrieval result: its token cost and the gold labels it covers.

    ``tokens`` is the item's cost measured in the same whitespace ledger the
    product reports (``omnex.kernel.packer.count_tokens``), so token totals across
    retrieval paths are denominated identically. ``covered`` is the set of
    gold-label ids whose identifying text this item contains; the runner computes
    it once, here it is taken as given.
    """

    tokens: int
    covered: frozenset[str]


def recall(retrieved: Set[str], relevant: Set[str]) -> float:
    """Fraction of ``relevant`` labels present in ``retrieved``.

    Returns ``1.0`` when nothing is relevant (an empty gold set is vacuously
    fully recalled), so the metric never divides by zero.
    """
    if not relevant:
        return 1.0
    return len(retrieved & relevant) / len(relevant)


def precision(retrieved: Set[str], relevant: Set[str]) -> float:
    """Fraction of ``retrieved`` labels that are relevant; ``0.0`` if none retrieved."""
    if not retrieved:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)


def f1(retrieved: Set[str], relevant: Set[str]) -> float:
    """Harmonic mean of :func:`precision` and :func:`recall`; ``0.0`` when either is ``0``.

    F1 is the secondary metric: at equal recall a path that drags in irrelevant
    labels (distractors) scores lower precision and therefore lower F1, which is
    how the benchmark surfaces precision differences the token number alone hides.
    """
    p = precision(retrieved, relevant)
    r = recall(retrieved, relevant)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def tokens_at_recall(
    results: Sequence[RetrievedItem],
    labels: Set[str],
    target_recall: float,
) -> int | None:
    """Minimum cumulative tokens a ranked retrieval spends to reach ``target_recall``.

    Walks ``results`` in rank order, accumulating both the union of covered gold
    ``labels`` and the token cost, and returns the cumulative token total at the
    first item whose cumulative recall reaches ``target_recall``. Returns ``None``
    when the full ranking never reaches that recall, so the caller refuses to
    report a token delta at unequal recall. Deterministic: it reads only its
    arguments and breaks no ties (the ranking is the caller's responsibility).
    """
    if not 0.0 <= target_recall <= 1.0:
        raise ValueError(f"target_recall must be in [0.0, 1.0], got {target_recall}")
    if not labels:
        raise ValueError("tokens_at_recall requires a non-empty label set")
    need = math.ceil(target_recall * len(labels))
    covered: set[str] = set()
    tokens = 0
    if len(covered) >= need:  # target_recall == 0.0: satisfied before any spend
        return tokens
    for item in results:
        tokens += item.tokens
        covered |= item.covered & labels
        if len(covered) >= need:
            return tokens
    return None


def p95_latency(latencies: Sequence[float]) -> float:
    """95th-percentile latency by the nearest-rank method.

    Sorts ``latencies`` ascending and returns the sample at 1-based rank
    ``ceil(0.95 * n)`` -- the smallest observation at or above which the slowest
    5% of runs sit. Deterministic for a fixed input. Raises on empty input rather
    than inventing a percentile of nothing.
    """
    if not latencies:
        raise ValueError("p95_latency requires at least one sample")
    ordered = sorted(latencies)
    rank = math.ceil(0.95 * len(ordered))
    return ordered[rank - 1]
