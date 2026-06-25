"""Tests for the omnex doctor diagnostics.

The registration check resolves each client's config under ``Path.home()``;
redirect that home to an isolated temp dir so no test reads a developer's real
client configs. The autouse ``omnex_home`` fixture (conftest) already isolates the
usage-metrics home and forces metrics off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnex.client_setup import build_client_install_plan, is_registered, write_client_install_plan
from omnex.doctor.checks import (
    _extra_installed,
    check_adapters,
    check_extras,
    check_metrics,
    check_persistence,
    check_registration,
)
from omnex.metrics import settings, store


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` to an isolated temporary directory."""
    target = tmp_path / "home"
    target.mkdir()
    monkeypatch.setenv("HOME", str(target))
    return target


def _event() -> store.UsageEvent:
    """A minimal anonymous usage event for ledger-count assertions."""
    return store.UsageEvent(
        occurred_at="2026-06-25T00:00:00+00:00",
        tool="query",
        surface="cli",
        category="prose",
        returned_tokens=100,
        baseline_tokens=400,
        file_count=2,
        repo_id="abc123",
    )


# --- registration detection (reusing the M1 registry) ---------------------------


def test_is_registered_false_when_no_config() -> None:
    assert is_registered("cursor") is False


def test_is_registered_true_after_write() -> None:
    write_client_install_plan(build_client_install_plan("cursor", scope="user"))
    assert is_registered("cursor") is True


def test_is_registered_codex_toml_after_write() -> None:
    write_client_install_plan(build_client_install_plan("codex", scope="user"))
    assert is_registered("codex") is True


def test_is_registered_ignores_unrelated_entry(home: Path) -> None:
    target = home / ".cursor" / "mcp.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"mcpServers": {"other": {"command": "x"}}}', encoding="utf-8")
    assert is_registered("cursor") is False


# --- registration check ---------------------------------------------------------


def test_check_registration_absent_warns() -> None:
    check = check_registration()
    assert check.name == "registration"
    assert check.status == "warn"
    assert check.details["registered"] == []
    clients = check.details["clients"]
    assert isinstance(clients, dict)
    assert all(value is False for value in clients.values())


def test_check_registration_present_is_ok() -> None:
    write_client_install_plan(build_client_install_plan("omp", scope="user"))
    check = check_registration()
    assert check.status == "ok"
    assert check.details["registered"] == ["omp"]
    assert "omp" in check.summary


# --- metrics state --------------------------------------------------------------


def test_check_metrics_default_disabled_creates_no_ledger() -> None:
    check = check_metrics()
    assert check.name == "metrics"
    assert check.status == "ok"
    assert check.details["enabled"] is False
    assert check.details["ledger_present"] is False
    assert check.details["event_count"] == 0
    assert not settings.ledger_path().exists()


def test_check_metrics_enabled_counts_events() -> None:
    settings.set_metrics_enabled(True)
    store.insert_event(settings.ledger_path(), _event())
    check = check_metrics()
    assert check.status == "ok"
    assert check.details["enabled"] is True
    assert check.details["ledger_present"] is True
    assert check.details["event_count"] == 1


def test_check_metrics_reports_corrupt_ledger() -> None:
    ledger = settings.ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("not a sqlite database", encoding="utf-8")
    check = check_metrics()
    assert check.status == "error"
    assert "unreadable" in check.summary


# --- installed extras -----------------------------------------------------------


def test_extra_installed_detects_presence_and_absence() -> None:
    assert _extra_installed("json") is True
    assert _extra_installed("no_such_module_xyz") is False


def test_check_extras_reports_known_extras() -> None:
    check = check_extras()
    assert check.name == "extras"
    assert check.status == "ok"
    installed = check.details["installed"]
    assert isinstance(installed, dict)
    assert set(installed) == {"mcp", "embed"}
    assert all(isinstance(value, bool) for value in installed.values())


# --- adapter sanity -------------------------------------------------------------


def test_check_adapters_routes_each_modality() -> None:
    check = check_adapters()
    assert check.name == "adapters"
    assert check.status == "ok"
    assert check.details["routes"] == {"prose": "ProseAdapter", "spec": "SpecAdapter"}
    assert check.details["adapters"] == ["SpecAdapter", "ProseAdapter"]


# --- persistence mode -----------------------------------------------------------


def test_check_persistence_reports_stateless() -> None:
    check = check_persistence()
    assert check.name == "persistence"
    assert check.status == "ok"
    assert check.details["mode"] == "stateless"
    assert "stateless" in check.summary
