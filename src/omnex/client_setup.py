"""MCP client registration registry and config writer for omnex.

This module knows where each supported MCP client keeps its configuration and
the exact server-entry shape that client expects, and it writes (or merges) an
``omnex`` entry that launches the ``omnex-mcp`` stdio server. It is pure data
plus file IO: it imports no surface framework (no ``click``, no ``mcp``) and no
``archex``. The supported client set mirrors archex's published compatibility
matrix, but the code is original and the registered command is ``omnex-mcp``.

The MCP server itself already exists (``src/omnex/mcp.py``, the ``omnex-mcp``
console script); this module only adds the registration ergonomics so adopters
do not hand-write client JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ClientName = Literal["claude-code", "codex", "cursor", "opencode", "pi", "omp"]
ClientScope = Literal["project", "user"]

# Every supported client, in stable presentation order. Used by surfaces to
# enumerate the choices and by tests to assert full coverage.
ALL_CLIENTS: tuple[ClientName, ...] = (
    "claude-code",
    "codex",
    "cursor",
    "opencode",
    "pi",
    "omp",
)

# The MCP server entry every client registers: the ``omnex-mcp`` console script
# over stdio with no extra arguments (it is a dedicated entry point, not a
# subcommand of ``omnex``).
_SERVER_NAME = "omnex"
_SERVER_COMMAND = "omnex-mcp"

# Clients whose configuration is only ever user/global, never repo-local.
_USER_ONLY_CLIENTS: frozenset[ClientName] = frozenset({"pi", "omp"})

# Client-specific config schemas (not omnex's): the JSON-schema URL each client
# advertises for its own config file. Injected as a top-level ``$schema`` only
# when the client documents one and the existing file does not already declare it.
_OPENCODE_SCHEMA = "https://opencode.ai/config.json"
_OMP_SCHEMA = (
    "https://raw.githubusercontent.com/can1357/oh-my-pi/main/"
    "packages/coding-agent/src/config/mcp-schema.json"
)
_CLIENT_SCHEMA: dict[ClientName, str] = {
    "opencode": _OPENCODE_SCHEMA,
    "omp": _OMP_SCHEMA,
}


@dataclass(frozen=True)
class ClientInstallPlan:
    """A resolved, ready-to-write registration for one client at one scope."""

    client: ClientName
    scope: ClientScope
    target_path: Path
    content: str


def build_client_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope,
) -> ClientInstallPlan:
    """Resolve the config target and rendered content for CLIENT at SCOPE.

    SOURCE is the repo root for a ``project`` scope (defaults to the current
    directory); it is unused for ``user`` scope. Raises ``ValueError`` for a
    user-only client (``pi``/``omp``) asked to install at ``project`` scope.
    """
    if client in _USER_ONLY_CLIENTS and scope != "user":
        raise ValueError(f"{client} client config supports only --scope user")
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    return ClientInstallPlan(
        client=client,
        scope=scope,
        target_path=_target_path(client, repo_root, scope),
        content=_render_content(client),
    )


def _target_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
    home = Path.home()
    if client == "claude-code":
        return repo_root / ".mcp.json" if scope == "project" else home / ".claude.json"
    if client == "cursor":
        return (
            repo_root / ".cursor" / "mcp.json"
            if scope == "project"
            else home / ".cursor" / "mcp.json"
        )
    if client == "opencode":
        return (
            repo_root / "opencode.json"
            if scope == "project"
            else home / ".config" / "opencode" / "opencode.json"
        )
    if client == "codex":
        return (
            repo_root / ".codex" / "config.toml"
            if scope == "project"
            else home / ".codex" / "config.toml"
        )
    if client == "pi":
        return home / ".pi" / "agent" / "mcp.json"
    if client == "omp":
        return home / ".omp" / "agent" / "mcp.json"
    raise ValueError(f"unsupported client: {client}")


def _stdio_entry() -> dict[str, object]:
    """The stdio launch entry for the omnex-mcp console script (no arguments)."""
    return {"command": _SERVER_COMMAND, "args": []}


def _json_document(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _render_content(client: ClientName) -> str:
    if client == "codex":
        return f'[mcp_servers.{_SERVER_NAME}]\ncommand = "{_SERVER_COMMAND}"\nargs = []\n'
    if client == "opencode":
        return _json_document(
            {
                "$schema": _OPENCODE_SCHEMA,
                "mcp": {
                    _SERVER_NAME: {
                        "type": "local",
                        "command": [_SERVER_COMMAND],
                        "enabled": True,
                    }
                },
            }
        )
    if client == "omp":
        return _json_document(
            {
                "$schema": _OMP_SCHEMA,
                "mcpServers": {_SERVER_NAME: _stdio_entry()},
            }
        )
    return _json_document({"mcpServers": {_SERVER_NAME: _stdio_entry()}})
