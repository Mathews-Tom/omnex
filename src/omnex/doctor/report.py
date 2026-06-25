"""Assemble the doctor report and render it as text or JSON.

:func:`run_doctor` runs every check in a stable order into a :class:`DoctorReport`
whose overall status is the worst single check status; a report is healthy only
when every check is ``ok``. :func:`report_to_dict` is the stable JSON schema the
``omnex doctor --format json`` command emits, and :func:`render_report_text` is
the human-readable rendering. Pure aggregation and string shaping over the checks
-- no I/O beyond what the checks themselves read.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnex.doctor.checks import (
    check_adapters,
    check_extras,
    check_metrics,
    check_persistence,
    check_registration,
)
from omnex.doctor.model import Check, CheckStatus

# Worst-to-best severity. The report's overall status is the worst check status,
# and a report is healthy only when every check is ``ok``.
_SEVERITY: dict[CheckStatus, int] = {"ok": 0, "warn": 1, "error": 2}


@dataclass(frozen=True, slots=True)
class DoctorReport:
    """The full diagnostic: every check plus the derived overall verdict."""

    checks: tuple[Check, ...]

    @property
    def status(self) -> CheckStatus:
        """The worst single check status (``ok`` < ``warn`` < ``error``)."""
        worst: CheckStatus = "ok"
        for check in self.checks:
            if _SEVERITY[check.status] > _SEVERITY[worst]:
                worst = check.status
        return worst

    @property
    def healthy(self) -> bool:
        """Whether every check is ``ok`` -- nothing is ``warn`` or ``error``."""
        return self.status == "ok"


def run_doctor() -> DoctorReport:
    """Run every health check in a stable order and assemble the report."""
    return DoctorReport(
        checks=(
            check_registration(),
            check_metrics(),
            check_extras(),
            check_adapters(),
            check_persistence(),
        )
    )


def report_to_dict(report: DoctorReport) -> dict[str, object]:
    """The stable JSON schema: the overall verdict plus an ordered list of checks."""
    return {
        "healthy": report.healthy,
        "status": report.status,
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "summary": check.summary,
                "details": check.details,
            }
            for check in report.checks
        ],
    }


def render_report_text(report: DoctorReport) -> str:
    """Render the report as labeled lines with an overall verdict footer."""
    lines = [f"[{check.status}] {check.name}: {check.summary}" for check in report.checks]
    verdict = "healthy" if report.healthy else "unhealthy"
    lines.extend(("", f"Overall: {verdict} ({report.status})"))
    return "\n".join(lines)
