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
"""

from __future__ import annotations

from datetime import UTC, datetime

from omnex.kernel.bundle import ContextBundle
from omnex.kernel.receipt import Receipt
from omnex.metrics import settings, store


def _now() -> str:
    """The current instant as an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


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
    question and the corpus path are never passed in and never stored.
    """
    if not settings.metrics_enabled():
        return
    event = store.UsageEvent(
        occurred_at=_now(),
        tool="query",
        surface=surface,
        category=bundle.dominant_style(),
        returned_tokens=receipt.returned_tokens,
        baseline_tokens=receipt.baseline_tokens,
        file_count=file_count,
        repo_id=settings.repo_id(),
    )
    store.insert_event(settings.ledger_path(), event)


def record_index(*, surface: str, file_count: int) -> None:
    """Record one anonymous ``index`` event when metrics are enabled.

    An index validates a corpus rather than retrieving from it, so it carries no
    token savings: the token counts are zero and the category is ``"index"``. Only
    the anonymous file count and the repo-local id are recorded.
    """
    if not settings.metrics_enabled():
        return
    event = store.UsageEvent(
        occurred_at=_now(),
        tool="index",
        surface=surface,
        category="index",
        returned_tokens=0,
        baseline_tokens=0,
        file_count=file_count,
        repo_id=settings.repo_id(),
    )
    store.insert_event(settings.ledger_path(), event)
