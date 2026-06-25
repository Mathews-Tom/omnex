"""Tests for the local usage-metrics layer.

The layer is off by default, local-only, and free of network access. These tests
assert the default-off posture, the enable resolution (persisted setting and the
``OMNEX_USAGE_METRICS`` override), and the SQLite ledger round-trip. Every test
redirects the omnex home directory to a temporary path with ``OMNEX_HOME`` and
clears the enable override, so nothing touches the real ``~/.omnex``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnex.metrics import settings, store
from omnex.metrics.store import UsageEvent


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the omnex home to a temp dir and clear the enable override."""
    home = tmp_path / ".omnex"
    monkeypatch.setenv("OMNEX_HOME", str(home))
    monkeypatch.delenv("OMNEX_USAGE_METRICS", raising=False)
    return home


def _event(
    *,
    occurred_at: str = "2026-06-25T00:00:00+00:00",
    tool: str = "query",
    surface: str = "cli",
    category: str = "spec",
    returned_tokens: int = 120,
    baseline_tokens: int = 4000,
    file_count: int = 1,
    repo_id: str = "abc123",
) -> UsageEvent:
    """A representative anonymous event; override fields per test."""
    return UsageEvent(
        occurred_at=occurred_at,
        tool=tool,
        surface=surface,
        category=category,
        returned_tokens=returned_tokens,
        baseline_tokens=baseline_tokens,
        file_count=file_count,
        repo_id=repo_id,
    )


def test_metrics_off_by_default(_isolated_home: Path) -> None:
    assert settings.metrics_enabled() is False


def test_reading_state_creates_no_file(_isolated_home: Path) -> None:
    # Default-off must not materialize the home directory, settings, or ledger.
    settings.metrics_enabled()
    store.read_events(settings.ledger_path())
    assert not _isolated_home.exists()
    assert not settings.settings_path().exists()
    assert not settings.ledger_path().exists()


def test_persisted_setting_enables_metrics() -> None:
    settings.set_metrics_enabled(True)
    assert settings.metrics_enabled() is True
    assert settings.settings_path().exists()


def test_persisted_setting_round_trips_off() -> None:
    settings.set_metrics_enabled(True)
    settings.set_metrics_enabled(False)
    assert settings.metrics_enabled() is False


@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "ON", "Yes"])
def test_env_override_forces_on(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNEX_USAGE_METRICS", value)
    assert settings.metrics_enabled() is True


@pytest.mark.parametrize("value", ["0", "off", "false", "no", ""])
def test_env_override_forces_off_over_persisted(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.set_metrics_enabled(True)
    monkeypatch.setenv("OMNEX_USAGE_METRICS", value)
    assert settings.metrics_enabled() is False


def test_env_override_on_beats_persisted_off(monkeypatch: pytest.MonkeyPatch) -> None:
    settings.set_metrics_enabled(False)
    monkeypatch.setenv("OMNEX_USAGE_METRICS", "on")
    assert settings.metrics_enabled() is True


def test_omnex_home_honors_override(_isolated_home: Path) -> None:
    assert settings.omnex_home() == _isolated_home
    assert settings.ledger_path() == _isolated_home / "usage.sqlite"


def test_ledger_round_trips_event() -> None:
    path = settings.ledger_path()
    store.insert_event(path, _event())
    events = store.read_events(path)
    assert events == [_event()]


def test_ledger_preserves_insertion_order() -> None:
    path = settings.ledger_path()
    store.insert_event(path, _event(returned_tokens=10))
    store.insert_event(path, _event(returned_tokens=20))
    assert [event.returned_tokens for event in store.read_events(path)] == [10, 20]


def test_read_events_on_absent_ledger_is_empty() -> None:
    assert store.read_events(settings.ledger_path()) == []


def test_insert_creates_ledger_lazily() -> None:
    path = settings.ledger_path()
    assert not path.exists()
    store.insert_event(path, _event())
    assert path.exists()


def test_delete_ledger_removes_file_and_reports() -> None:
    path = settings.ledger_path()
    store.insert_event(path, _event())
    assert store.delete_ledger(path) is True
    assert not path.exists()
    assert store.delete_ledger(path) is False
