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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from omnex.ir.types import Unit
from omnex.kernel.config import KernelConfig

RepresentationMode = Literal["INCLUDE", "COMPRESS", "ELIDE", "SKIP"]

# Tiers whose packing (including COMPRESS) is deterministic and model-free. The T2
# vector lane changes only which candidates arrive, never how they are packed, so
# its packing stays deterministic; T3 model extraction is not packable here.
_DETERMINISTIC_TIERS = frozenset({"T0", "T1", "T2"})


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


def _compress(unit: Unit) -> str:
    """Deterministic, model-free COMPRESS: heading plus lead paragraph.

    Uses the unit's title (falling back to its deepest breadcrumb) as the heading
    and the first non-empty paragraph as the lead. Pure string operations only.
    """
    heading = unit.title or (unit.breadcrumb[-1] if unit.breadcrumb else "")
    paragraphs = [block.strip() for block in unit.text.split("\n\n") if block.strip()]
    lead = paragraphs[0] if paragraphs else unit.text.strip()
    if heading and lead:
        return f"{heading}\n\n{lead}"
    return heading or lead


def _elide(unit: Unit) -> str:
    """Identity scaffold for continuity: breadcrumb and title, else the unit id."""
    parts = [*unit.breadcrumb]
    if unit.title:
        parts.append(unit.title)
    scaffold = " / ".join(part for part in parts if part)
    return scaffold or unit.id


def pack_efficiently(
    candidates: Sequence[Candidate],
    budget: int,
    config: KernelConfig,
) -> list[Representation]:
    """Pack candidates into ``budget`` tokens via the strict representation chain.

    Candidates are ordered by descending score, ties broken by ascending unit id.
    Each descends INCLUDE -> COMPRESS -> ELIDE -> SKIP until a representation fits
    the remaining budget; a protected unit skips the COMPRESS and ELIDE rungs
    entirely, so it is only ever INCLUDE or SKIP. The emitted total never exceeds
    ``budget``. ``config.tier`` must be a deterministic-packing tier (T0/T1/T2);
    the T2 vector lane only changes which candidates arrive, so packing stays
    model-free. T3 model extraction is not packable here and fails loud.
    """
    if config.tier not in _DETERMINISTIC_TIERS:
        raise NotImplementedError(
            f"deterministic packing supports only tiers {sorted(_DETERMINISTIC_TIERS)}; "
            f"tier {config.tier!r} would require model-backed extraction"
        )
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    ordered = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.unit.id))
    remaining = budget
    representations: list[Representation] = []
    for candidate in ordered:
        unit = candidate.unit
        full_tokens = count_tokens(unit.text)
        if full_tokens <= remaining:
            representations.append(Representation(unit.id, "INCLUDE", unit.text, full_tokens))
            remaining -= full_tokens
            continue
        if not unit.protect:
            compressed = _compress(unit)
            compressed_tokens = count_tokens(compressed)
            # Only emit COMPRESS when it is genuinely smaller than the full text;
            # a single-paragraph unit compresses to heading + whole body, which is
            # not smaller, so it falls through to ELIDE instead.
            if compressed_tokens < full_tokens and compressed_tokens <= remaining:
                representations.append(
                    Representation(unit.id, "COMPRESS", compressed, compressed_tokens)
                )
                remaining -= compressed_tokens
                continue
            elided = _elide(unit)
            elided_tokens = count_tokens(elided)
            if elided_tokens <= remaining:
                representations.append(Representation(unit.id, "ELIDE", elided, elided_tokens))
                remaining -= elided_tokens
                continue
        representations.append(Representation(unit.id, "SKIP", "", 0))
    return representations
