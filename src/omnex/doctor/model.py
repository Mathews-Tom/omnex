"""The doctor health-check data model: one check result and its status.

A :class:`Check` is the unit every diagnostic produces -- a name, a health
``status``, a one-line human summary, and a JSON-serializable ``details`` mapping
of structured fields. The report layer aggregates these into the overall verdict
and renders them as text or JSON. This module is pure data: no I/O, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# A check is healthy (``ok``), a soft advisory that something actionable is
# pending (``warn``), or broken (``error``). ``--strict`` treats anything other
# than ``ok`` as unhealthy.
CheckStatus = Literal["ok", "warn", "error"]


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnostic result: a named status with a summary and structured details."""

    name: str
    status: CheckStatus
    summary: str
    details: dict[str, object] = field(default_factory=dict)
