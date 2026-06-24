"""Generic SQLite FTS5 index with BM25F column weighting over the IR.

The index is modality-blind: it stores only ``Unit`` text fields (``text``,
``title``, ``breadcrumb``, ``summary``) and ranks with BM25F weights supplied by
a per-modality ``profile``. The kernel never hard-codes per-modality weights in
its source; they arrive as configuration, so the same index serves prose, code,
and specs without branching on modality.

Determinism is a contract: the same corpus and query always yield the same
ordering. BM25 ties are broken by unit id so the result is total and stable.
Queries are sanitized into quoted FTS terms, so arbitrary Unicode and FTS
special characters are matched as literal terms instead of being interpreted as
query operators.

This module performs no model load, no network, and no file-system access. The
SQLite connection is created in-memory on instantiation, not on import.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable, Mapping

from omnex.ir.types import Unit

# Indexed, BM25F-weighted columns, in the order the FTS5 table declares them.
# ``unit_id`` is stored UNINDEXED before these and carries weight 0.0.
_WEIGHTED_COLUMNS: tuple[str, ...] = ("text", "title", "breadcrumb", "summary")

# Default BM25F weight for a column a profile does not mention.
_DEFAULT_WEIGHT = 1.0

# Token pattern: Unicode word runs only. Everything else (FTS operators,
# punctuation, quotes) is dropped before it can reach the query parser.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _build_match(query: str) -> str | None:
    """Turn a free-text query into a safe FTS5 MATCH expression.

    Extracts Unicode word tokens and joins them as quoted ``OR`` terms. Quoting
    each token neutralizes FTS keywords (``AND``/``OR``/``NOT``/``NEAR``) and any
    residual special characters, so the query is always parsed as literal terms.
    Returns ``None`` when the query has no word tokens.
    """
    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return None
    return " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)


class FtsIndex:
    """An in-memory SQLite FTS5 index over IR units with BM25F ranking."""

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

    def search(
        self,
        query: str,
        profile: Mapping[str, float],
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Return ``(unit_id, score)`` pairs ranked best-first under BM25F.

        ``profile`` maps weighted column names to BM25F weights; a column absent
        from ``profile`` uses the default weight. Scores are the negated SQLite
        BM25 value (higher is better). Ties on score are broken by ascending unit
        id, making the order total and deterministic regardless of insertion
        order. A token-free query or a non-positive ``limit`` returns an empty
        list.
        """
        match = _build_match(query)
        if match is None or limit <= 0:
            return []
        # Column-order weights: unit_id (UNINDEXED) first at 0.0, then the
        # weighted columns in declared order. Only the placeholder count is
        # interpolated into the SQL; every weight and the query are bound.
        weights = [0.0] + [float(profile.get(col, _DEFAULT_WEIGHT)) for col in _WEIGHTED_COLUMNS]
        placeholders = ", ".join("?" for _ in weights)
        rows = self._conn.execute(
            f"SELECT unit_id, bm25(units, {placeholders}) FROM units WHERE units MATCH ?",
            (*weights, match),
        ).fetchall()
        scored = [(str(unit_id), -float(raw)) for unit_id, raw in rows]
        scored.sort(key=lambda row: (-row[1], row[0]))
        return scored[:limit]
