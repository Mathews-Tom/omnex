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
from typing import Literal, cast

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


def write_client_install_plan(plan: ClientInstallPlan) -> Path:
    """Write PLAN's config, merging the ``omnex`` entry into any existing file.

    JSON clients merge an ``omnex`` server entry into the existing server map
    without clobbering unrelated keys; Codex appends one ``[mcp_servers.omnex]``
    TOML section. A re-run with an identical entry is an idempotent no-op; a
    differing existing ``omnex`` entry is left untouched and the write fails
    rather than overwrite it. Returns the written target path.
    """
    target = plan.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_toml_plan(plan):
        return _write_toml_plan(target, plan.content)
    return _write_json_plan(target, plan.client, plan.content)


def _write_toml_plan(target: Path, content: str) -> Path:
    block = content.strip()
    marker = f"[mcp_servers.{_SERVER_NAME}]"
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if marker in existing:
            if block in existing:
                return target
            raise ValueError(f"{_SERVER_NAME} already configured in {target}")
        new_content = existing.rstrip() + "\n\n" + content if existing.strip() else content
    else:
        new_content = content
    target.write_text(new_content, encoding="utf-8")
    return target


def _write_json_plan(target: Path, client: ClientName, content: str) -> Path:
    key = _json_server_key(client)
    payload_obj: object = json.loads(content)
    if not isinstance(payload_obj, dict):
        raise ValueError(f"expected JSON object in generated content for {client}")
    payload = cast("dict[str, object]", payload_obj)

    existing_payload: dict[str, object]
    if target.exists():
        existing_obj: object = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(existing_obj, dict):
            raise ValueError(f"expected JSON object in {target}")
        existing_payload = cast("dict[str, object]", existing_obj)
    else:
        existing_payload = {}

    raw_container = existing_payload.get(key)
    if raw_container is None:
        container: dict[str, object] = {}
        existing_payload[key] = container
    elif isinstance(raw_container, dict):
        container = cast("dict[str, object]", raw_container)
    else:
        raise ValueError(f"expected object at {key!r} in {target}")

    payload_container_obj = payload.get(key)
    if not isinstance(payload_container_obj, dict):
        raise ValueError(f"expected object at {key!r} in generated content for {client}")
    payload_container = cast("dict[str, object]", payload_container_obj)

    entry_obj = payload_container.get(_SERVER_NAME)
    if not isinstance(entry_obj, dict):
        raise ValueError(f"expected {_SERVER_NAME} entry in generated content for {client}")
    entry = cast("dict[str, object]", entry_obj)

    existing_entry = container.get(_SERVER_NAME)
    if existing_entry is not None:
        if existing_entry == entry:
            return target
        raise ValueError(f"{_SERVER_NAME} already configured in {target}")
    container[_SERVER_NAME] = entry

    schema = _CLIENT_SCHEMA.get(client)
    if schema is not None and "$schema" not in existing_payload:
        existing_payload["$schema"] = schema

    target.write_text(json.dumps(existing_payload, indent=2) + "\n", encoding="utf-8")
    return target


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


def _json_server_key(client: ClientName) -> str:
    return "mcp" if client == "opencode" else "mcpServers"


def _is_toml_plan(plan: ClientInstallPlan) -> bool:
    return plan.client == "codex"
