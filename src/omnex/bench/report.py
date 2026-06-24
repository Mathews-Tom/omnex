"""Two-number honesty framing for token comparisons.

The full-document dump is an upper bound, not a competitor: it reaches full
recall by returning everything, so leading with "we beat full-dump" overstates
the result. This module fixes that framing. A comparison has three roles -- the
*subject* (omnex), the *headline* baseline (the realistic chunk-and-embed
retrieval omnex must actually beat), and the *upper bound* (full-dump, demoted
and labeled as such). The headline is what the verdict is judged against; the
upper bound is shown only to bound waste.

Two rules keep the rendering honest:

- The upper bound is always labeled demoted, never presented as the thing beaten.
- A token delta is reported only when both compared paths reached the recall
  target. A path that never reached it renders as ``unreached`` and yields no
  ratio, so the report never compares tokens at unequal recall.

Until the headline baseline exists it renders as ``pending``. Benchmark-only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PathResult:
    """One retrieval path's standing in a comparison.

    ``tokens`` is the path's cost to reach the comparison's recall target, or
    ``None`` when the path never reached it. ``available`` is ``False`` only for a
    headline baseline that does not exist yet (rendered ``pending``), which is
    distinct from an available path that failed to reach the recall (``None``
    tokens, rendered ``unreached``).
    """

    name: str
    tokens: int | None
    available: bool = True


@dataclass(frozen=True, slots=True)
class Comparison:
    """Tokens at one equal-recall target across the three honesty roles."""

    task: str
    recall_target: float
    subject: PathResult
    headline: PathResult
    upper_bound: PathResult


def _tokens_phrase(result: PathResult) -> str:
    if not result.available:
        return "pending (baseline not yet available)"
    if result.tokens is None:
        return "unreached at this recall"
    return f"{result.tokens} tokens"


def verdict(comparison: Comparison) -> str:
    """One-line verdict of subject vs the headline at equal recall.

    Returns a delta only when both the subject and the headline reached the recall
    target; otherwise it states that no equal-recall delta is available, so the
    verdict never implies a comparison at unequal recall.
    """
    subject, headline = comparison.subject, comparison.headline
    if not headline.available:
        return f"{subject.name}: headline baseline pending; no equal-recall delta yet"
    if subject.tokens is None or headline.tokens is None:
        return f"{subject.name}: recall target unreached by a compared path; no equal-recall delta"
    relation = "<=" if subject.tokens <= headline.tokens else ">"
    ratio = subject.tokens / headline.tokens if headline.tokens else float("inf")
    return (
        f"{subject.name} {subject.tokens} {relation} {headline.name} {headline.tokens} "
        f"at recall {comparison.recall_target:.2f} ({ratio:.2f}x of headline)"
    )


def render(comparison: Comparison) -> str:
    """Render one comparison with the upper bound demoted and headline foremost."""
    lines = [
        f"{comparison.task} @ recall={comparison.recall_target:.2f}",
        f"  subject     {comparison.subject.name}: {_tokens_phrase(comparison.subject)}",
        f"  headline    {comparison.headline.name}: {_tokens_phrase(comparison.headline)}",
        f"  upper bound (demoted)  {comparison.upper_bound.name}: "
        f"{_tokens_phrase(comparison.upper_bound)}",
        f"  verdict     {verdict(comparison)}",
    ]
    return "\n".join(lines)


def render_report(title: str, comparisons: Sequence[Comparison]) -> str:
    """Render a titled block of comparisons, each in the two-number framing."""
    blocks = [f"{title}\n{'=' * len(title)}"]
    blocks.extend(render(comparison) for comparison in comparisons)
    return "\n\n".join(blocks)
