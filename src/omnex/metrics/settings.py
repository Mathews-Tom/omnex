"""Usage-metrics settings: the omnex home directory and the enable resolution.

The usage ledger is off by default. It turns on only when the operator opts in,
either persistently (a setting written by ``omnex metrics enable``) or for one
session (the ``OMNEX_USAGE_METRICS`` environment variable). Resolution order:

* If ``OMNEX_USAGE_METRICS`` is set, it wins -- a truthy value forces metrics on,
  any other value forces them off, so a session can override the persisted state
  in either direction.
* Otherwise the persisted ``usage_metrics`` setting applies.
* With neither set, metrics are off and no file is created.

Everything lives under the user's omnex home directory (``~/.omnex`` by default,
or ``OMNEX_HOME`` when set). Reading state never creates a file or directory;
only an explicit write (enabling metrics, recording an event) does. No network
access, no background process, no upload.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Environment overrides. ``OMNEX_HOME`` relocates the state directory (useful in
# containers and required for isolated tests); ``OMNEX_USAGE_METRICS`` is the
# per-session enable override.
_ENV_HOME = "OMNEX_HOME"
_ENV_METRICS = "OMNEX_USAGE_METRICS"

# Values that read as "on" for the enable override (compared case-insensitively).
_TRUTHY = frozenset({"1", "on", "true", "yes"})

_SETTINGS_NAME = "settings.json"
_LEDGER_NAME = "usage.sqlite"

_METRICS_KEY = "usage_metrics"


def omnex_home() -> Path:
    """The omnex state directory: ``OMNEX_HOME`` when set, else ``~/.omnex``.

    Pure path resolution -- it neither reads nor creates anything on disk.
    """
    override = os.environ.get(_ENV_HOME)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".omnex"


def settings_path() -> Path:
    """Path to the JSON settings file under the omnex home directory."""
    return omnex_home() / _SETTINGS_NAME


def ledger_path() -> Path:
    """Path to the SQLite usage ledger under the omnex home directory."""
    return omnex_home() / _LEDGER_NAME


def read_settings() -> dict[str, object]:
    """Load the settings mapping, or an empty mapping when no file exists.

    A missing file is the default-off state, not an error: reading it returns an
    empty mapping and creates nothing.
    """
    path = settings_path()
    if not path.exists():
        return {}
    loaded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"settings file is not a JSON object: {path}")
    return loaded


def write_settings(settings: dict[str, object]) -> None:
    """Persist the settings mapping, creating the omnex home directory if needed."""
    home = omnex_home()
    home.mkdir(parents=True, exist_ok=True)
    settings_path().write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _env_override(name: str) -> bool | None:
    """The boolean an enable env var forces, or None when the var is unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in _TRUTHY


def metrics_enabled() -> bool:
    """Whether usage recording is on, resolving the env override then the setting."""
    override = _env_override(_ENV_METRICS)
    if override is not None:
        return override
    return bool(read_settings().get(_METRICS_KEY, False))


def set_metrics_enabled(enabled: bool) -> None:
    """Persist the usage-recording setting (independent of the env override)."""
    settings = read_settings()
    settings[_METRICS_KEY] = enabled
    write_settings(settings)
