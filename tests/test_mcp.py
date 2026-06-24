"""Tests for the MCP server surface: tool registration, query parity, optional extra.

The MCP tools are thin wrappers over :mod:`omnex.api`, so these assert the query
tool returns the same structured ContextBundle and Receipt the library and CLI
produce (a JSON-normalized payload), that the byte-exact floor invokes no model,
and -- via a fresh interpreter -- that ``import omnex`` never pulls in the MCP
surface or its optional ``mcp`` dependency, so the core install is unaffected.

Tools are invoked in-process through the server's async ``call_tool`` /
``list_tools``; each test drives them with ``asyncio.run`` so no pytest-asyncio
plugin is required.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import omnex
from omnex.cli import _render_json, default_config
from omnex.mcp import server

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_PAYMENTS = _FIXTURES / "payments_openapi.json"
_QUESTION = "create a payment"
_BUDGET = 2000

_RECEIPT_KEYS = {
    "returned_tokens",
    "baseline_tokens",
    "tiers_run",
    "model_used",
    "model_version",
    "extraction_used",
    "determinism_class",
    "reference_closure_complete",
    "recall_basis",
    "recall_limitations",
}


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # call_tool returns (content_blocks, structured_output); return the structured dict.
    result = asyncio.run(server.call_tool(name, arguments))
    assert isinstance(result, tuple), result
    structured = result[1]
    assert isinstance(structured, dict)
    return structured


def test_tools_are_registered() -> None:
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {"index", "query"} <= names


def test_index_tool_reports_corpus_shape() -> None:
    units, references, documents = omnex.api._route_sources([_PAYMENTS])
    assert _call("index", {"paths": [str(_PAYMENTS)]}) == {
        "documents": len(documents),
        "units": len(units),
        "references": len(references),
    }


def test_query_tool_returns_bundle_and_receipt() -> None:
    result = _call("query", {"corpus": str(_PAYMENTS), "question": _QUESTION, "budget": _BUDGET})
    assert set(result) == {"bundle", "receipt"}
    assert set(result["bundle"]) == {"context", "total_tokens", "representations"}
    assert set(result["receipt"]) == _RECEIPT_KEYS
    # The byte-exact floor invokes no model and no extraction; the receipt says so.
    assert result["receipt"]["model_used"] is False
    assert result["receipt"]["extraction_used"] is False
    assert result["receipt"]["determinism_class"] == "byte_exact"


def test_query_tool_matches_library_and_cli() -> None:
    result = _call("query", {"corpus": str(_PAYMENTS), "question": _QUESTION, "budget": _BUDGET})
    bundle, receipt = omnex.query_sources([_PAYMENTS], _QUESTION, _BUDGET, default_config())
    # The MCP structured output is exactly the CLI's JSON payload (both are the
    # library's bundle and receipt, JSON-normalized).
    assert result == json.loads(_render_json(bundle, receipt))
    assert result["bundle"]["context"] == bundle.render()
    assert result["receipt"]["returned_tokens"] == receipt.returned_tokens
    assert result["receipt"]["baseline_tokens"] == receipt.baseline_tokens
    assert [rep["unit_id"] for rep in result["bundle"]["representations"]] == [
        rep.unit_id for rep in bundle.representations
    ]


def test_query_tool_returns_fewer_tokens_than_full_dump() -> None:
    receipt = _call("query", {"corpus": str(_PAYMENTS), "question": _QUESTION, "budget": _BUDGET})[
        "receipt"
    ]
    assert receipt["returned_tokens"] < receipt["baseline_tokens"]


def test_core_import_does_not_require_mcp() -> None:
    # A fresh interpreter importing the core package must not load the MCP surface
    # or the optional `mcp` dependency, so a core install without the [mcp] extra
    # works even though this dev environment has mcp installed.
    code = (
        "import sys, omnex; "
        "assert 'omnex.mcp' not in sys.modules, 'core import loaded omnex.mcp'; "
        "assert 'mcp' not in sys.modules, 'core import loaded the mcp dependency'; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
