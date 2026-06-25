"""The core omnex health checks: registration, metrics, extras, and adapters.

Each function probes one operational surface and returns a :class:`Check`. The
checks are read-only and side-effect free: they create no ledger, write no
config, and make no network call. The registration check reuses the M1 client
registry rather than re-deriving any config path, so registration health and the
``install-client`` writer share one source of config-path and config-shape truth.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import tempfile
from pathlib import Path

from omnex.adapters import ModalityAdapter, available_adapters, select_adapter
from omnex.client_setup import ALL_CLIENTS, is_registered
from omnex.doctor.model import Check
from omnex.metrics import settings, store

# The optional-dependency extras omnex ships, mapped to the import name that
# proves each is installed. ``bench`` is an alias of ``embed`` and is not probed
# separately.
_EXTRAS: dict[str, str] = {"mcp": "mcp", "embed": "fastembed"}

# The modality each probe file must route to. A mismatch is a real misroute and
# fails the adapter check.
_EXPECTED_ROUTES: dict[str, str] = {"prose": "ProseAdapter", "spec": "SpecAdapter"}

# Minimal probe sources: a prose document and an OpenAPI spec. The content is the
# smallest input each adapter's routing predicate accepts.
_PROBES: dict[str, tuple[str, str]] = {
    "prose": ("probe.md", "# Probe\n\nA prose probe paragraph.\n"),
    "spec": (
        "probe.json",
        '{"openapi": "3.0.0", "info": {"title": "p", "version": "1"}, "paths": {}}',
    ),
}


def check_registration() -> Check:
    """Report which clients have the omnex MCP server registered at user scope.

    Reuses the M1 registry's :func:`omnex.client_setup.is_registered`, so the
    config path and shape are never re-derived here. Reports ``warn`` when no
    client is registered (the server is unreachable by any agent until then).
    """
    registered = tuple(client for client in ALL_CLIENTS if is_registered(client))
    details: dict[str, object] = {
        "registered": list(registered),
        "clients": {client: client in registered for client in ALL_CLIENTS},
    }
    if registered:
        return Check(
            name="registration",
            status="ok",
            summary=f"omnex registered in {len(registered)} client(s): {', '.join(registered)}",
            details=details,
        )
    return Check(
        name="registration",
        status="warn",
        summary="no MCP client registered (run: omnex install-client <client>)",
        details=details,
    )


def check_metrics() -> Check:
    """Report the local usage-ledger state: enable, trace, and event count.

    Reads only anonymous state and never creates the ledger: a missing ledger is
    the default-off posture. A diagnostics command must report a broken subsystem,
    not crash, so a corrupt ledger or settings file is surfaced as an error rather
    than propagated.
    """
    try:
        enabled = settings.metrics_enabled()
        trace = settings.trace_enabled()
        ledger = settings.ledger_path()
        ledger_present = ledger.exists()
        event_count = len(store.read_events(ledger)) if ledger_present else 0
    except (OSError, ValueError, sqlite3.Error) as error:
        return Check(
            name="metrics",
            status="error",
            summary=f"usage metrics unreadable: {error}",
            details={},
        )
    details: dict[str, object] = {
        "enabled": enabled,
        "trace_enabled": trace,
        "ledger_present": ledger_present,
        "event_count": event_count,
    }
    state = "enabled" if enabled else "disabled (default)"
    return Check(
        name="metrics",
        status="ok",
        summary=f"usage metrics {state}; {event_count} event(s) recorded",
        details=details,
    )


def _extra_installed(module: str) -> bool:
    """Whether MODULE is importable, without importing it."""
    return importlib.util.find_spec(module) is not None


def check_extras() -> Check:
    """Report which optional extras are installed (informational)."""
    installed = {extra: _extra_installed(module) for extra, module in _EXTRAS.items()}
    present = [extra for extra, ok in installed.items() if ok]
    absent = [extra for extra, ok in installed.items() if not ok]
    summary = f"extras installed: {', '.join(present) if present else 'none'}"
    if absent:
        summary += f"; absent: {', '.join(absent)}"
    return Check(name="extras", status="ok", summary=summary, details={"installed": installed})


def _probe_routes() -> dict[str, str]:
    """Route each probe source, returning a modality -> adapter-class-name map."""
    routes: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for modality, (filename, content) in _PROBES.items():
            path = root / filename
            path.write_text(content, encoding="utf-8")
            routes[modality] = type(select_adapter(path)).__name__
    return routes


def check_adapters() -> Check:
    """Probe the modality adapter registry: contract conformance and routing.

    A diagnostics probe must report a broken subsystem, not crash, so the
    expected adapter failures (no claiming adapter, an unreadable probe) are
    caught and surfaced as ``error``; a genuine programming bug still propagates.
    """
    try:
        adapters = available_adapters()
        names = [type(adapter).__name__ for adapter in adapters]
        if not adapters:
            return Check(
                name="adapters",
                status="error",
                summary="no modality adapters registered",
                details={"adapters": names},
            )
        for adapter in adapters:
            if not isinstance(adapter, ModalityAdapter):
                return Check(
                    name="adapters",
                    status="error",
                    summary=f"{type(adapter).__name__} does not satisfy the adapter contract",
                    details={"adapters": names},
                )
            adapter.capabilities()
        routes = _probe_routes()
    except (OSError, ValueError) as error:
        return Check(
            name="adapters",
            status="error",
            summary=f"adapter probe failed: {error}",
            details={},
        )
    if routes != _EXPECTED_ROUTES:
        return Check(
            name="adapters",
            status="error",
            summary=f"adapter routing mismatch: {routes}",
            details={"adapters": names, "routes": routes, "expected": _EXPECTED_ROUTES},
        )
    return Check(
        name="adapters",
        status="ok",
        summary=f"{len(adapters)} adapter(s) healthy: {', '.join(names)}",
        details={"adapters": names, "routes": routes},
    )
