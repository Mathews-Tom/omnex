"""Shared test fixtures.

The usage-metrics layer reads and writes under the omnex home directory, and the
CLI and MCP surfaces now record an anonymous event on every ``query``/``index``.
Redirect the home to a per-test temporary directory and force the metrics enable
override off, so no test ever touches the real ``~/.omnex`` or records into a
developer's ledger.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def omnex_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the omnex home at an isolated temp dir; clear the enable overrides."""
    home = tmp_path_factory.mktemp("omnex-home") / ".omnex"
    monkeypatch.setenv("OMNEX_HOME", str(home))
    monkeypatch.delenv("OMNEX_USAGE_METRICS", raising=False)
    monkeypatch.delenv("OMNEX_USAGE_TRACE", raising=False)
    return home
