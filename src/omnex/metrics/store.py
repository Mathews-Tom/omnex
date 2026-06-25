"""The SQLite usage ledger: a local, append-only store of anonymous events.

Each row is an anonymous usage counter -- the operation, the surface that drove
it (CLI vs MCP), a coarse category, the receipt's returned/baseline token counts,
the file count, and a repo-local random id. It never holds query text, paths,
symbols, handles, or rendered output.

Storage is the standard library ``sqlite3`` -- no new dependency, no network, no
background process. The ledger file is created lazily on the first write, so a
default-off install never materializes ``~/.omnex/usage.sqlite``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from pathlib import Path

# Bumped when the on-disk schema changes so a future reader can detect and reject
# a ledger it does not understand rather than silently misreading it.
_SCHEMA_VERSION = 1

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    tool TEXT NOT NULL,
    surface TEXT NOT NULL,
    category TEXT NOT NULL,
    returned_tokens INTEGER NOT NULL,
    baseline_tokens INTEGER NOT NULL,
    file_count INTEGER NOT NULL,
    repo_id TEXT NOT NULL
)
"""

_INSERT_EVENT = """
INSERT INTO events (
    occurred_at, tool, surface, category,
    returned_tokens, baseline_tokens, file_count, repo_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_EVENTS = """
SELECT occurred_at, tool, surface, category,
       returned_tokens, baseline_tokens, file_count, repo_id
FROM events
ORDER BY id ASC
"""


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One anonymous usage row.

    ``tool`` is the operation (``query`` or ``index``); ``surface`` is the entry
    point that drove it (``cli`` or ``mcp``); ``category`` is a coarse,
    content-free label (a render style, or ``index``). ``returned_tokens`` and
    ``baseline_tokens`` are copied verbatim from the receipt -- the savings layer
    derives every figure from them and never recomputes from files. ``repo_id`` is
    a repo-local random id, never a path.
    """

    occurred_at: str
    tool: str
    surface: str
    category: str
    returned_tokens: int
    baseline_tokens: int
    file_count: int
    repo_id: str


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the ledger, ensuring the schema and the home directory exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(_CREATE_EVENTS)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_ledger(path: Path) -> None:
    """Create the ledger file and its schema if they do not already exist."""
    with _connect(path):
        pass


def insert_event(path: Path, event: UsageEvent) -> None:
    """Append one anonymous event to the ledger, creating it on first write."""
    with _connect(path) as conn:
        conn.execute(_INSERT_EVENT, astuple(event))


def read_events(path: Path) -> list[UsageEvent]:
    """Read every event in insertion order, or ``[]`` when no ledger exists.

    A missing ledger is the default-off state, not an error.
    """
    if not path.exists():
        return []
    with _connect(path) as conn:
        rows = conn.execute(_SELECT_EVENTS).fetchall()
    return [
        UsageEvent(
            occurred_at=str(row["occurred_at"]),
            tool=str(row["tool"]),
            surface=str(row["surface"]),
            category=str(row["category"]),
            returned_tokens=int(row["returned_tokens"]),
            baseline_tokens=int(row["baseline_tokens"]),
            file_count=int(row["file_count"]),
            repo_id=str(row["repo_id"]),
        )
        for row in rows
    ]


def delete_ledger(path: Path) -> bool:
    """Delete the ledger file. Returns whether a file was actually removed."""
    if not path.exists():
        return False
    path.unlink()
    return True
