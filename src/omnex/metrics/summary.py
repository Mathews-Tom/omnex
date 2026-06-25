"""Build and render a usage-metrics summary, including the CLI-vs-MCP split.

The summary aggregates recorded events into overall savings and a per-surface
breakdown, so an operator can see both the total token savings and whether agents
actually route context through omnex (the MCP surface) versus calling the CLI by
hand. Rendering keeps the honest labels the savings layer defines: the two
headline figures stay headline, and the whole-corpus dump stays a demoted upper
bound.

Pure data and string shaping -- no I/O, no network.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from omnex.metrics.savings import TARGETED_READ_MULTIPLE, Savings, compute_savings
from omnex.metrics.store import UsageEvent


@dataclass(frozen=True, slots=True)
class SurfaceBreakdown:
    """Event count and savings for one surface (e.g. ``cli`` or ``mcp``)."""

    surface: str
    events: int
    savings: Savings


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """The full summary: enable state, totals, overall savings, surface split."""

    enabled: bool
    trace_enabled: bool
    total_events: int
    overall: Savings
    by_surface: tuple[SurfaceBreakdown, ...]


def build_summary(
    events: Iterable[UsageEvent], *, enabled: bool, trace_enabled: bool = False
) -> MetricsSummary:
    """Aggregate ``events`` into overall savings and a per-surface breakdown."""
    materialized = list(events)
    surfaces = sorted({event.surface for event in materialized})
    by_surface = tuple(
        SurfaceBreakdown(
            surface=surface,
            events=sum(1 for event in materialized if event.surface == surface),
            savings=compute_savings(event for event in materialized if event.surface == surface),
        )
        for surface in surfaces
    )
    return MetricsSummary(
        enabled=enabled,
        trace_enabled=trace_enabled,
        total_events=len(materialized),
        overall=compute_savings(materialized),
        by_surface=by_surface,
    )


def savings_to_dict(savings: Savings) -> dict[str, object]:
    """A JSON-serializable mapping of a :class:`Savings`, including its percentages."""
    return {
        "events": savings.events,
        "returned_tokens": savings.returned_tokens,
        "full_file_paste_avoided": savings.full_file_paste_avoided,
        "full_file_paste_pct": savings.full_file_paste_pct,
        "targeted_read_avoided": savings.targeted_read_avoided,
        "targeted_read_baseline": savings.targeted_read_baseline,
        "targeted_read_pct": savings.targeted_read_pct,
        "whole_corpus_tokens": savings.whole_corpus_tokens,
    }


def summary_to_dict(summary: MetricsSummary) -> dict[str, object]:
    """A JSON-serializable mapping of the whole summary, surface split included."""
    return {
        "enabled": summary.enabled,
        "trace_enabled": summary.trace_enabled,
        "total_events": summary.total_events,
        "targeted_read_multiple": TARGETED_READ_MULTIPLE,
        "overall": savings_to_dict(summary.overall),
        "by_surface": {
            breakdown.surface: {
                "events": breakdown.events,
                "savings": savings_to_dict(breakdown.savings),
            }
            for breakdown in summary.by_surface
        },
    }


def render_summary_text(summary: MetricsSummary) -> str:
    """Render the summary as labeled human-readable text."""
    state = "on" if summary.enabled else "off"
    lines = [
        f"Usage metrics: {state}",
        f"Trace: {'on' if summary.trace_enabled else 'off'}",
        f"Events recorded: {summary.total_events}",
        "",
        *_render_savings_block("Overall", summary.overall),
    ]
    if summary.by_surface:
        lines.append("")
        lines.append("By surface (CLI vs MCP):")
        for breakdown in summary.by_surface:
            saved = breakdown.savings.full_file_paste_avoided
            pct = breakdown.savings.full_file_paste_pct
            lines.append(
                f"  {breakdown.surface}: {breakdown.events} event(s), "
                f"{saved} tokens saved vs full-file paste ({pct}%)"
            )
    return "\n".join(lines)


def _render_savings_block(title: str, savings: Savings) -> list[str]:
    """The labeled savings lines: two headlines, then the demoted upper bound."""
    return [
        f"{title} savings (from {savings.events} query event(s)):",
        f"  Full-file paste:  {savings.full_file_paste_avoided} tokens saved "
        f"({savings.full_file_paste_pct}%) [headline]",
        f"  Targeted read:    {savings.targeted_read_avoided} tokens saved "
        f"({savings.targeted_read_pct}%) "
        f"[conservative; assumes a targeted read ~= {TARGETED_READ_MULTIPLE}x returned tokens]",
        f"  Whole-corpus:     {savings.whole_corpus_tokens} tokens "
        "[upper bound -- cost of dumping every queried corpus; not a realized saving]",
    ]
