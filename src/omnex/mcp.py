"""MCP server surface over the omnex public library API.

Exposes the same ``index`` and ``query`` operations the library and CLI provide
as MCP tools over stdio, so an MCP client gets identical retrieval, the same
returned set, and the same receipt the library produces. The tools are thin
wrappers over :mod:`omnex.api` (via the shared :mod:`omnex._surface` helpers);
they change no retrieval ranking, no returned set, and no receipt schema.

This module requires the optional ``mcp`` dependency (the ``[mcp]`` extra). It is
never imported by ``import omnex``, so the core install does not depend on it;
importing :mod:`omnex.mcp` without the extra fails loud with an ImportError.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from omnex import api
from omnex._surface import (
    _DEFAULT_BUDGET,
    collect_files,
    default_config,
    index_corpus,
    result_payload,
)

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
    """Ingest, parse, and link the given paths into IR and report the corpus shape.

    Each path is routed through its claiming adapter -- failing loud when none
    claims it -- and built into the FTS index and StructureGraph to validate the
    full index path. A directory path is expanded to its files. No state is
    persisted; the tool returns the corpus shape it would index.
    """
    if not paths:
        raise ValueError("index requires at least one path")
    documents, units, references = index_corpus(collect_files([Path(p) for p in paths]))
    return {"documents": documents, "units": units, "references": references}


@server.tool()
def query(corpus: str, question: str, budget: int = _DEFAULT_BUDGET) -> dict[str, object]:
    """Answer a question over a corpus under a token budget; return bundle and receipt.

    Routes the corpus (a file or directory) through its adapters and runs the same
    byte-exact, model-free T0 pipeline the library and CLI do, then returns the
    structured ContextBundle and Receipt. The retrieval, ranking, and returned set
    are exactly the library's; the tool only shapes them into the shared result
    payload the CLI also emits.
    """
    sources = collect_files([Path(corpus)])
    bundle, receipt = api.query_sources(sources, question, budget, default_config())
    return result_payload(bundle, receipt)


def main() -> None:
    """Run the omnex MCP server over stdio."""
    server.run()
