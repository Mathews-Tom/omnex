"""omnex health diagnostics: the ``doctor`` checks and their data model.

Each check probes one operational surface (MCP registration, the usage-metrics
ledger, installed extras, the modality adapters, and -- added downstack -- the
persistence mode) and returns a :class:`Check`. The report layer aggregates them
into an overall verdict rendered as text or JSON for the ``omnex doctor`` command.

Nothing here is pulled by ``import omnex``; the CLI reaches for it explicitly, so
the core library import stays cheap and side-effect free.
"""

from __future__ import annotations

from omnex.doctor.checks import (
    check_adapters,
    check_extras,
    check_metrics,
    check_registration,
)
from omnex.doctor.model import Check, CheckStatus

__all__ = [
    "Check",
    "CheckStatus",
    "check_adapters",
    "check_extras",
    "check_metrics",
    "check_registration",
]
