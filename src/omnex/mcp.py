"""MCP server surface over the omnex public library API.

Exposes the same ``index`` and ``query`` operations the library and CLI provide
as MCP tools over stdio, so an MCP client gets identical retrieval, the same
returned set, and the same receipt the library produces. The tools are thin
wrappers over :mod:`omnex.api`; they change no retrieval ranking, no returned
set, and no receipt schema.

This module requires the optional ``mcp`` dependency (the ``[mcp]`` extra). It is
never imported by ``import omnex``, so the core install does not depend on it;
importing :mod:`omnex.mcp` without the extra fails loud with an ImportError.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from omnex import api
from omnex.cli import _collect_files

server = FastMCP(
    "omnex",
    instructions=(
        "Universal, structure-aware retrieval at a fraction of the tokens. "
        "Use `index` to validate and summarize a corpus, and `query` to retrieve "
        "a token-budgeted ContextBundle with an auditable Receipt."
    ),
)


@server.tool()
def index(paths: list[str]) -> dict[str, int]:
    """Ingest, parse, and link PATHS into IR and report the indexed corpus shape.

    Each path is routed through its claiming adapter -- failing loud when none
    claims it -- and built into the FTS index and StructureGraph to validate the
    full index path. A directory path is expanded to its files. No state is
    persisted; the tool returns the corpus shape it would index.
    """
    sources = _collect_files([Path(p) for p in paths])
    units, references, documents = api._route_sources(sources)
    # Build the index and graph so a corpus that routes but cannot be indexed
    # fails here rather than silently at query time.
    api.index(units, references)
    return {
        "documents": len(documents),
        "units": len(units),
        "references": len(references),
    }


def main() -> None:
    """Run the omnex MCP server over stdio."""
    server.run()
