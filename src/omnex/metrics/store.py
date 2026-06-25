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

# Stamped into the ledger's PRAGMA user_version so a future reader can tell which
# on-disk schema it is looking at. Bump it whenever the schema changes.
_SCHEMA_VERSION = 2

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

# The trace table holds richer per-run diagnostics that the summary aggregates
# away -- but still only anonymous fields. It deliberately has no column for the
# question, the corpus path, unit text, or rendered output.
_CREATE_TRACES = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    tool TEXT NOT NULL,
    surface TEXT NOT NULL,
    repo_id TEXT NOT NULL,
    tier TEXT NOT NULL,
    determinism_class TEXT NOT NULL,
    recall_basis TEXT NOT NULL,
    reference_closure_complete INTEGER NOT NULL
)
"""

_INSERT_TRACE = """
INSERT INTO traces (
    occurred_at, tool, surface, repo_id,
    tier, determinism_class, recall_basis, reference_closure_complete
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_TRACES = """
SELECT occurred_at, tool, surface, repo_id,
       tier, determinism_class, recall_basis, reference_closure_complete
FROM traces
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


@dataclass(frozen=True, slots=True)
class UsageTrace:
    """One anonymous diagnostic trace for a retrieval.

    Carries only receipt-derived diagnostics the summary aggregates away -- the
    ``tier`` exercised, the ``determinism_class``, the ``recall_basis``, and
    whether the reference closure was complete -- plus the same anonymous
    identifiers an event holds. It never carries the question, a path, unit text,
    or rendered output.
    """

    occurred_at: str
    tool: str
    surface: str
    repo_id: str
    tier: str
    determinism_class: str
    recall_basis: str
    reference_closure_complete: bool


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open the ledger, ensuring the schema and the home directory exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(_CREATE_EVENTS)
        conn.execute(_CREATE_TRACES)
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


def insert_trace(path: Path, trace: UsageTrace) -> None:
    """Append one anonymous diagnostic trace, creating the ledger on first write."""
    with _connect(path) as conn:
        conn.execute(_INSERT_TRACE, astuple(trace))


def read_traces(path: Path) -> list[UsageTrace]:
    """Read every trace in insertion order, or ``[]`` when no ledger exists."""
    if not path.exists():
        return []
    with _connect(path) as conn:
        rows = conn.execute(_SELECT_TRACES).fetchall()
    return [
        UsageTrace(
            occurred_at=str(row["occurred_at"]),
            tool=str(row["tool"]),
            surface=str(row["surface"]),
            repo_id=str(row["repo_id"]),
            tier=str(row["tier"]),
            determinism_class=str(row["determinism_class"]),
            recall_basis=str(row["recall_basis"]),
            reference_closure_complete=bool(int(row["reference_closure_complete"])),
        )
        for row in rows
    ]


def delete_ledger(path: Path) -> bool:
    """Delete the ledger file. Returns whether a file was actually removed."""
    if not path.exists():
        return False
    path.unlink()
    return True
