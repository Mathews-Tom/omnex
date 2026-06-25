"""Anonymous usage recording for the query and index surfaces.

The recorder is the one place a surface calls to log a run. It is gated on
:func:`omnex.metrics.settings.metrics_enabled`, so it is a no-op unless usage
metrics are explicitly turned on -- a surface always calls it unconditionally and
the default-off posture lives here.

It records only anonymous counters taken from the public receipt and bundle: the
operation, the surface, a coarse content-free category, the receipt's
returned/baseline token counts, the file count, and a repo-local random id. It
never sees or stores the question, the corpus path, unit text, symbols, or the
rendered output, and it makes no network call.

Recording is a non-essential side channel, so its failures are isolated from the
retrieval path: a broken or full home directory, a locked or corrupt ledger, or a
malformed settings file must never discard an otherwise-successful retrieval (on
the MCP surface the result would be lost entirely). Such failures are surfaced
loudly on stderr -- never silently swallowed -- and only the expected I/O,
database, and settings-parse errors are caught, so a real programming bug still
propagates.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime

from omnex.kernel.bundle import ContextBundle
from omnex.kernel.receipt import Receipt
from omnex.metrics import settings, store


def _now() -> str:
    """The current instant as an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _unrecorded(error: Exception) -> None:
    """Surface a metrics-recording failure loudly, without breaking retrieval."""
    print(f"omnex: usage metrics not recorded: {error}", file=sys.stderr)


def record_query(
    *,
    surface: str,
    receipt: Receipt,
    bundle: ContextBundle,
    file_count: int,
) -> None:
    """Record one anonymous ``query`` event when metrics are enabled.

    The token counts are copied verbatim from the receipt -- never recomputed from
    files or a model -- and the category is the bundle's coarse render style. The
    question and the corpus path are never passed in and never stored. When the
    separate trace opt-in is also on, an anonymous diagnostic trace is appended
    alongside the event. A recording failure is isolated, never propagated.
    """
    try:
        if not settings.metrics_enabled():
            return
        occurred_at = _now()
        repo_id = settings.repo_id()
        ledger = settings.ledger_path()
        store.insert_event(
            ledger,
            store.UsageEvent(
                occurred_at=occurred_at,
                tool="query",
                surface=surface,
                category=bundle.dominant_style(),
                returned_tokens=receipt.returned_tokens,
                baseline_tokens=receipt.baseline_tokens,
                file_count=file_count,
                repo_id=repo_id,
            ),
        )
        if settings.trace_enabled():
            store.insert_trace(
                ledger,
                store.UsageTrace(
                    occurred_at=occurred_at,
                    tool="query",
                    surface=surface,
                    repo_id=repo_id,
                    tier=",".join(receipt.tiers_run),
                    determinism_class=receipt.determinism_class,
                    recall_basis=receipt.recall_basis,
                    reference_closure_complete=receipt.reference_closure_complete,
                ),
            )
    except (OSError, sqlite3.Error, ValueError) as error:
        _unrecorded(error)


def record_index(*, surface: str, file_count: int) -> None:
    """Record one anonymous ``index`` event when metrics are enabled.

    An index validates a corpus rather than retrieving from it, so it carries no
    token savings: the token counts are zero and the category is ``"index"``. Only
    the anonymous file count and the repo-local id are recorded. A recording
    failure is isolated, never propagated.
    """
    try:
        if not settings.metrics_enabled():
            return
        store.insert_event(
            settings.ledger_path(),
            store.UsageEvent(
                occurred_at=_now(),
                tool="index",
                surface=surface,
                category="index",
                returned_tokens=0,
                baseline_tokens=0,
                file_count=file_count,
                repo_id=settings.repo_id(),
            ),
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        _unrecorded(error)
