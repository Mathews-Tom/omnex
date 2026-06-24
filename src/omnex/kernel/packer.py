"""The efficiency packer: choose whole-unit representations under a token budget.

The packer is the substantial part of the kernel. It scores candidates by
relevance per token with a graph-distance penalty, then walks a strict
representation chain under a token budget:

- ``INCLUDE``: emit the full unit text.
- ``COMPRESS``: emit a deterministic shorter representation.
- ``ELIDE``: keep only the identity scaffold (breadcrumb and title).
- ``SKIP``: omit the unit entirely.

``protect=True`` is a hard guard: a protected unit is only ever ``INCLUDE`` or
``SKIP``, never ``COMPRESS`` or ``ELIDE``. In the byte-exact tiers (T0 and T1)
``COMPRESS`` is explicitly deterministic and model-free: a heading-plus-lead
representation built with pure string operations. The packer fails loud for any
other tier, since a model-backed ``COMPRESS`` does not exist here.

No model load, network, or file-system access.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from omnex.ir.types import Unit

RepresentationMode = Literal["INCLUDE", "COMPRESS", "ELIDE", "SKIP"]


def count_tokens(text: str) -> int:
    """Deterministic, offline token measure: count of whitespace-separated runs.

    This is the budget ledger's unit of account. It is intentionally model-free
    and stable so the packed total is byte-exact and reproducible.
    """
    return len(text.split())


@dataclass(frozen=True, slots=True)
class Candidate:
    """A unit considered for packing.

    ``score`` must already fold in the graph-distance penalty (it is what
    ``score_candidate`` returns); packing orders candidates by ``score`` alone.
    ``graph_distance`` is retained for receipt auditing, not re-applied here.
    """

    unit: Unit
    score: float
    graph_distance: int


@dataclass(frozen=True, slots=True)
class Representation:
    """The packer's chosen representation of one unit, with its emitted token cost.

    ``SKIP`` representations carry empty text and zero tokens; they are recorded
    so the receipt can audit what was dropped.
    """

    unit_id: str
    mode: RepresentationMode
    text: str
    token_count: int


def score_candidate(unit: Unit, query_signals: Mapping[str, float], graph_distance: int) -> float:
    """Score a unit by relevance per token with a graph-distance penalty.

    Relevance is ``query_signals[unit.id]`` (0.0 if absent). It is divided by the
    unit's size in the same ``count_tokens`` ledger the packer budgets against, so
    a smaller unit at equal relevance scores higher, then by ``1 + graph_distance``
    so a far neighbor scores below a near one at equal density. Using one token
    measure keeps scoring and budgeting denominated identically. The result is
    non-negative for non-negative signals.
    """
    relevance = query_signals.get(unit.id, 0.0)
    per_token = relevance / max(count_tokens(unit.text), 1)
    return per_token / (1.0 + graph_distance)
