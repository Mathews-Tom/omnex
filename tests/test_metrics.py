"""Tests for the local usage-metrics layer.

The layer is off by default, local-only, and free of network access. These tests
assert the default-off posture, the enable resolution (persisted setting and the
``OMNEX_USAGE_METRICS`` override), and the SQLite ledger round-trip. Every test
redirects the omnex home directory to a temporary path with ``OMNEX_HOME`` and
clears the enable override, so nothing touches the real ``~/.omnex``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import omnex
from omnex._surface import default_config
from omnex.metrics import recorder, savings, settings, store
from omnex.metrics.store import UsageEvent

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_PAYMENTS = _FIXTURES / "payments_openapi.json"
# A distinctive marker so the redaction test can prove the question never reaches
# the ledger; "create a payment" is the same query the surface tests use.
_SECRET_QUESTION = "ZZTOPSECRETMARKER create a payment"


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


def test_metrics_off_by_default(omnex_home: Path) -> None:
    assert settings.metrics_enabled() is False


def test_reading_state_creates_no_file(omnex_home: Path) -> None:
    # Default-off must not materialize the home directory, settings, or ledger.
    settings.metrics_enabled()
    store.read_events(settings.ledger_path())
    assert not omnex_home.exists()
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


def test_omnex_home_honors_override(omnex_home: Path) -> None:
    assert settings.omnex_home() == omnex_home
    assert settings.ledger_path() == omnex_home / "usage.sqlite"


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


def _query() -> tuple[omnex.ContextBundle, omnex.Receipt]:
    """Run a real T0 query over the spec fixture, returning bundle and receipt."""
    return omnex.query_sources([_PAYMENTS], _SECRET_QUESTION, 2000, default_config())


def test_disabled_recording_writes_no_event() -> None:
    bundle, receipt = _query()
    recorder.record_query(surface="cli", receipt=receipt, bundle=bundle, file_count=1)
    assert store.read_events(settings.ledger_path()) == []
    assert not settings.ledger_path().exists()


def test_enabled_query_records_one_event() -> None:
    settings.set_metrics_enabled(True)
    bundle, receipt = _query()
    recorder.record_query(surface="cli", receipt=receipt, bundle=bundle, file_count=1)
    events = store.read_events(settings.ledger_path())
    assert len(events) == 1
    event = events[0]
    assert event.tool == "query"
    assert event.surface == "cli"
    assert event.file_count == 1
    assert event.repo_id != ""


def test_event_token_counts_come_from_receipt_verbatim() -> None:
    settings.set_metrics_enabled(True)
    bundle, receipt = _query()
    recorder.record_query(surface="cli", receipt=receipt, bundle=bundle, file_count=1)
    event = store.read_events(settings.ledger_path())[0]
    assert event.returned_tokens == receipt.returned_tokens
    assert event.baseline_tokens == receipt.baseline_tokens


def test_query_category_is_the_spec_render_style() -> None:
    settings.set_metrics_enabled(True)
    bundle, receipt = _query()
    recorder.record_query(surface="cli", receipt=receipt, bundle=bundle, file_count=1)
    assert store.read_events(settings.ledger_path())[0].category == "spec"


def test_event_row_redacts_question_and_paths() -> None:
    settings.set_metrics_enabled(True)
    bundle, receipt = _query()
    recorder.record_query(surface="cli", receipt=receipt, bundle=bundle, file_count=1)
    event = store.read_events(settings.ledger_path())[0]
    fields = "\x00".join(str(value) for value in dataclasses.astuple(event))
    assert "ZZTOPSECRETMARKER" not in fields
    assert str(_PAYMENTS) not in fields
    # The strongest guarantee: the raw ledger bytes carry no question, path, or
    # even the source filename -- only anonymous counters.
    raw = settings.ledger_path().read_bytes()
    assert b"ZZTOPSECRETMARKER" not in raw
    assert str(_PAYMENTS).encode() not in raw
    assert b"payments_openapi" not in raw


def test_index_records_zero_token_event() -> None:
    settings.set_metrics_enabled(True)
    recorder.record_index(surface="mcp", file_count=3)
    event = store.read_events(settings.ledger_path())[0]
    assert event.tool == "index"
    assert event.category == "index"
    assert event.surface == "mcp"
    assert event.file_count == 3
    assert event.returned_tokens == 0
    assert event.baseline_tokens == 0


def test_surface_split_is_recorded() -> None:
    settings.set_metrics_enabled(True)
    bundle, receipt = _query()
    recorder.record_query(surface="cli", receipt=receipt, bundle=bundle, file_count=1)
    recorder.record_query(surface="mcp", receipt=receipt, bundle=bundle, file_count=1)
    assert [event.surface for event in store.read_events(settings.ledger_path())] == ["cli", "mcp"]


def test_repo_id_is_stable_and_path_free(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    first = settings.repo_id(repo)
    second = settings.repo_id(repo)
    assert first == second
    content = settings.settings_path().read_text(encoding="utf-8")
    assert str(repo) not in content
    assert str(repo.resolve()) not in content


def test_distinct_repos_get_distinct_ids(tmp_path: Path) -> None:
    left = tmp_path / "left"
    left.mkdir()
    right = tmp_path / "right"
    right.mkdir()
    assert settings.repo_id(left) != settings.repo_id(right)


def _ev(returned: int, baseline: int, *, tool: str = "query") -> UsageEvent:
    """A usage event carrying given token counts; defaults to a query event."""
    return _event(returned_tokens=returned, baseline_tokens=baseline, tool=tool)


def test_savings_of_no_events_is_all_zero() -> None:
    result = savings.compute_savings([])
    assert result == savings.Savings(0, 0, 0, 0, 0, 0)
    assert result.full_file_paste_pct == 0.0
    assert result.targeted_read_pct == 0.0


def test_single_query_savings_math() -> None:
    multiple = savings.TARGETED_READ_MULTIPLE
    targeted_baseline = min(1000, 100 * multiple)
    result = savings.compute_savings([_ev(returned=100, baseline=1000)])
    assert result.events == 1
    assert result.returned_tokens == 100
    assert result.whole_corpus_tokens == 1000
    assert result.full_file_paste_avoided == 900
    assert result.targeted_read_baseline == targeted_baseline
    assert result.targeted_read_avoided == targeted_baseline - 100
    assert result.full_file_paste_pct == round(100.0 * 900 / 1000, 1)
    assert result.targeted_read_pct == round(
        100.0 * (targeted_baseline - 100) / targeted_baseline, 1
    )


def test_savings_aggregate_across_events() -> None:
    events = [_ev(returned=100, baseline=1000), _ev(returned=50, baseline=400)]
    result = savings.compute_savings(events)
    assert result.events == 2
    assert result.returned_tokens == 150
    assert result.whole_corpus_tokens == 1400
    assert result.full_file_paste_avoided == 900 + 350


def test_index_events_carry_no_savings() -> None:
    query_only = savings.compute_savings([_ev(returned=100, baseline=1000)])
    with_index = savings.compute_savings(
        [_ev(returned=100, baseline=1000), _ev(returned=0, baseline=0, tool="index")]
    )
    assert with_index == query_only


def test_targeted_read_is_capped_at_the_full_file() -> None:
    # returned * multiple exceeds the full document, so a targeted read can only
    # cost the whole file: the targeted figure collapses onto the full-file paste.
    result = savings.compute_savings([_ev(returned=400, baseline=1000)])
    assert result.targeted_read_baseline == result.whole_corpus_tokens
    assert result.targeted_read_avoided == result.full_file_paste_avoided


def test_savings_never_go_negative() -> None:
    # A degenerate event where returned exceeds baseline must clamp to zero, never
    # report a negative saving.
    result = savings.compute_savings([_ev(returned=1000, baseline=500)])
    assert result.full_file_paste_avoided == 0
    assert result.targeted_read_avoided == 0


def test_targeted_read_never_exceeds_full_file_paste() -> None:
    events = [_ev(returned=120, baseline=4000), _ev(returned=300, baseline=900)]
    result = savings.compute_savings(events)
    assert result.targeted_read_avoided <= result.full_file_paste_avoided
    assert result.full_file_paste_avoided <= result.whole_corpus_tokens
