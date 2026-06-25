"""Token-savings calculator over recorded usage events.

Every figure is derived from the receipt token counts already stored on each
event -- ``returned_tokens`` (what omnex emitted) and ``baseline_tokens`` (the
full-document dump). Nothing here re-reads files or calls a model.

Three figures, from the most conservative to the loosest:

* **Targeted read** (most conservative headline): the counterfactual where, rather
  than pasting a file whole, you open it and read around the answer. That pull is
  modeled as a small multiple of the tokens omnex finally returns, capped at the
  full document (:data:`TARGETED_READ_MULTIPLE`). This is a labeled modeling
  assumption, surfaced in the summary, never presented as measured.
* **Full-file paste** (headline): the realized saving versus pasting the queried
  file(s) whole -- ``baseline_tokens - returned_tokens``, straight from the
  receipt.
* **Whole-corpus tokens** (demoted upper bound): the gross cost of dumping every
  queried corpus in full (``sum(baseline_tokens)``). It is reported as a labeled
  upper bound and never as the headline saving, because no one re-pastes the whole
  corpus on every query.

Only token-bearing events (queries) contribute; ``index`` events carry no
retrieval and no savings.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from omnex.metrics.store import UsageEvent

# A targeted read pulls roughly this multiple of the tokens omnex finally returns
# -- the matched region plus its immediate surroundings, before structural
# trimming -- capped at the full document. It defines the most conservative
# (smallest) reported saving and is the hardest baseline for omnex to beat.
TARGETED_READ_MULTIPLE = 3


@dataclass(frozen=True, slots=True)
class Savings:
    """Aggregated token savings across a set of usage events.

    ``full_file_paste_avoided`` and ``targeted_read_avoided`` are the two labeled
    headline savings. ``whole_corpus_tokens`` is the gross full-dump total, the
    demoted upper bound and the denominator for the full-file-paste percentage.
    """

    events: int
    returned_tokens: int
    full_file_paste_avoided: int
    targeted_read_avoided: int
    targeted_read_baseline: int
    whole_corpus_tokens: int

    @property
    def full_file_paste_pct(self) -> float:
        """Percent of the full-corpus dump avoided by returning omnex's set instead."""
        return _percent(self.full_file_paste_avoided, self.whole_corpus_tokens)

    @property
    def targeted_read_pct(self) -> float:
        """Percent of a targeted read avoided -- the most conservative figure."""
        return _percent(self.targeted_read_avoided, self.targeted_read_baseline)


def _percent(part: int, whole: int) -> float:
    """``part`` as a one-decimal percent of ``whole``; 0.0 when ``whole`` is zero."""
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 1)


def targeted_read_baseline(returned_tokens: int, baseline_tokens: int) -> int:
    """The targeted-read counterfactual cost: a multiple of returned, capped at full."""
    return min(baseline_tokens, returned_tokens * TARGETED_READ_MULTIPLE)


def compute_savings(events: Iterable[UsageEvent]) -> Savings:
    """Aggregate savings over ``events``, counting only token-bearing queries."""
    counted = 0
    returned = 0
    full_avoided = 0
    targeted_avoided = 0
    targeted_baseline = 0
    whole_corpus = 0
    for event in events:
        baseline = event.baseline_tokens
        if baseline <= 0:
            # index events validate a corpus; they retrieve nothing and save nothing.
            continue
        returned_tokens = event.returned_tokens
        counted += 1
        returned += returned_tokens
        whole_corpus += baseline
        full_avoided += max(0, baseline - returned_tokens)
        read_baseline = targeted_read_baseline(returned_tokens, baseline)
        targeted_baseline += read_baseline
        targeted_avoided += max(0, read_baseline - returned_tokens)
    return Savings(
        events=counted,
        returned_tokens=returned,
        full_file_paste_avoided=full_avoided,
        targeted_read_avoided=targeted_avoided,
        targeted_read_baseline=targeted_baseline,
        whole_corpus_tokens=whole_corpus,
    )
