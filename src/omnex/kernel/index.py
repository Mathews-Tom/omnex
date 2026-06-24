"""Generic SQLite FTS5 index over the IR.

The index is modality-blind: it stores only ``Unit`` text fields (``text``,
``title``, ``breadcrumb``, ``summary``). The kernel never hard-codes per-modality
behavior in its source, so the same index serves prose, code, and specs without
branching on modality.

This module performs no model load, no network, and no file-system access. The
SQLite connection is created in-memory on instantiation, not on import.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from omnex.ir.types import Unit

# Indexed columns, in the order the FTS5 table declares them. ``unit_id`` is
# stored UNINDEXED before these.
_WEIGHTED_COLUMNS: tuple[str, ...] = ("text", "title", "breadcrumb", "summary")


class FtsIndex:
    """An in-memory SQLite FTS5 index over IR units."""

    __slots__ = ("_conn",)

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        columns = ", ".join(_WEIGHTED_COLUMNS)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE units USING fts5("
            f"unit_id UNINDEXED, {columns}, tokenize='unicode61')"
        )

    def index_units(self, units: Iterable[Unit]) -> None:
        """Index ``units``, replacing any existing row with the same unit id.

        Re-indexing a unit id overwrites its prior row, so repeated calls remain
        idempotent and the index never holds duplicate rows for one unit.
        """
        with self._conn:
            for unit in units:
                self._conn.execute("DELETE FROM units WHERE unit_id = ?", (unit.id,))
                self._conn.execute(
                    "INSERT INTO units (unit_id, text, title, breadcrumb, summary) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        unit.id,
                        unit.text,
                        unit.title or "",
                        " ".join(unit.breadcrumb),
                        unit.summary or "",
                    ),
                )
