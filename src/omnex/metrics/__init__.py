"""Local, off-by-default usage-metrics layer for omnex.

This package records anonymous usage counters to a machine-local SQLite ledger so
an operator can see the token savings omnex's structure-aware retrieval delivers.
It is **off by default**, **local-only**, and makes **no network call**: it never
uploads, never runs a background process, and stores only anonymous counters --
never query text, paths, symbols, handles, or rendered output.

Nothing here is imported by ``import omnex``; the surfaces (the CLI and the MCP
server) reach for it explicitly, so the core library import stays cheap and
side-effect free. The ledger and settings live under the user's omnex home
directory (``~/.omnex`` by default, overridable with ``OMNEX_HOME``), never in
the repository.
"""

from __future__ import annotations
