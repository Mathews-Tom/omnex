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

import hashlib
import json
import os
import uuid
from pathlib import Path

# Environment overrides. ``OMNEX_HOME`` relocates the state directory (useful in
# containers and required for isolated tests); ``OMNEX_USAGE_METRICS`` is the
# per-session enable override.
_ENV_HOME = "OMNEX_HOME"
_ENV_METRICS = "OMNEX_USAGE_METRICS"
_ENV_TRACE = "OMNEX_USAGE_TRACE"

# Values that read as "on" for the enable override (compared case-insensitively).
_TRUTHY = frozenset({"1", "on", "true", "yes"})

_SETTINGS_NAME = "settings.json"
_LEDGER_NAME = "usage.sqlite"

_METRICS_KEY = "usage_metrics"
_TRACE_KEY = "usage_trace"
# Per-install random salt and the repo-id map. The salt makes the repo key a
# non-reversible hash of the repo root; the map holds only random ids keyed by
# that hash, so the repo path is never written to disk.
_SALT_KEY = "metrics_salt"
_REPOS_KEY = "repos"


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


def trace_enabled() -> bool:
    """Whether detailed tracing is on -- the second, separate opt-in.

    Resolved like the metrics flag (the ``OMNEX_USAGE_TRACE`` override, then the
    persisted ``usage_trace`` setting), and off by default. Tracing only takes
    effect when usage recording is also on; it never enables recording by itself.
    """
    override = _env_override(_ENV_TRACE)
    if override is not None:
        return override
    return bool(read_settings().get(_TRACE_KEY, False))


def set_trace_enabled(enabled: bool) -> None:
    """Persist the tracing setting (independent of the env override)."""
    settings = read_settings()
    settings[_TRACE_KEY] = enabled
    write_settings(settings)


def _repo_root(start: Path) -> Path:
    """The nearest enclosing git work tree, or START itself when there is none."""
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return resolved


def repo_id(start: Path | None = None) -> str:
    """A stable, anonymous per-repo id -- never the repo path.

    The id is keyed by a salted hash of the repo root, so the same repo reuses one
    random id across runs while the path itself is never written to disk. The salt
    is a per-install random value and the id is random; both live in the settings
    file, never in the ledger. START defaults to the current working directory.
    """
    settings = read_settings()
    salt = settings.get(_SALT_KEY)
    if not isinstance(salt, str):
        salt = uuid.uuid4().hex
    repos: dict[str, str] = {}
    raw_repos = settings.get(_REPOS_KEY)
    if isinstance(raw_repos, dict):
        repos = {k: v for k, v in raw_repos.items() if isinstance(k, str) and isinstance(v, str)}
    root = _repo_root(start if start is not None else Path.cwd())
    key = hashlib.sha256(f"{salt}:{root}".encode()).hexdigest()[:16]
    existing = repos.get(key)
    if existing is not None:
        return existing
    new_id = uuid.uuid4().hex[:12]
    repos[key] = new_id
    settings[_SALT_KEY] = salt
    settings[_REPOS_KEY] = repos
    write_settings(settings)
    return new_id
