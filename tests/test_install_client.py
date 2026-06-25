"""Tests for the cross-client MCP registration registry and config writer.

Each supported client gets its config written to a temporary HOME (user scope)
or repo root (project scope), then parsed back and asserted to register an
``omnex`` server entry that launches ``omnex-mcp``. Merge, idempotency, and
scope-rejection invariants are exercised per format (JSON vs Codex TOML).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnex.cli import main
from omnex.client_setup import (
    ALL_CLIENTS,
    ClientName,
    build_client_install_plan,
    render_client_install_preview,
    resolve_scope,
    write_client_install_plan,
)

# Clients that also support a repo-local (``project``) scope; ``pi``/``omp`` are
# user-only and excluded.
_PROJECT_CLIENTS: tuple[ClientName, ...] = ("claude-code", "codex", "cursor", "opencode")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` to an isolated temporary directory."""
    target = tmp_path / "home"
    target.mkdir()
    monkeypatch.setenv("HOME", str(target))
    return target


def _omnex_command(client: ClientName, target: Path) -> str:
    """Parse TARGET and return the command string the omnex entry registers."""
    if client == "codex":
        data = tomllib.loads(target.read_text(encoding="utf-8"))
        entry = data["mcp_servers"]["omnex"]
        assert entry["args"] == []
        command = entry["command"]
    elif client == "opencode":
        data = json.loads(target.read_text(encoding="utf-8"))
        entry = data["mcp"]["omnex"]
        assert entry["type"] == "local"
        assert entry["enabled"] is True
        command = entry["command"][0]
    else:
        data = json.loads(target.read_text(encoding="utf-8"))
        entry = data["mcpServers"]["omnex"]
        assert entry["args"] == []
        command = entry["command"]
    assert isinstance(command, str)
    return command


def test_registry_enumerates_six_clients() -> None:
    assert set(ALL_CLIENTS) == {"claude-code", "codex", "cursor", "opencode", "pi", "omp"}


@pytest.mark.parametrize("client", ALL_CLIENTS)
def test_each_client_registers_omnex_mcp(client: ClientName, home: Path) -> None:
    plan = build_client_install_plan(client, scope="user")
    target = write_client_install_plan(plan)
    assert target.exists()
    assert _omnex_command(client, target) == "omnex-mcp"


@pytest.mark.parametrize("client", _PROJECT_CLIENTS)
def test_project_scope_targets_repo_root(client: ClientName, tmp_path: Path) -> None:
    root = tmp_path.resolve()
    plan = build_client_install_plan(client, str(root), scope="project")
    assert plan.target_path.is_relative_to(root)
    target = write_client_install_plan(plan)
    assert _omnex_command(client, target) == "omnex-mcp"


@pytest.mark.parametrize("client", ["pi", "omp"])
def test_user_only_clients_reject_project_scope(client: ClientName) -> None:
    with pytest.raises(ValueError, match="only --scope user"):
        build_client_install_plan(client, scope="project")


def test_json_merge_preserves_unrelated_keys(home: Path) -> None:
    target = home / ".claude.json"
    target.write_text(
        json.dumps({"mcpServers": {"other": {"command": "x"}}, "unrelated": 1}),
        encoding="utf-8",
    )
    write_client_install_plan(build_client_install_plan("claude-code", scope="user"))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["mcpServers"]["other"] == {"command": "x"}
    assert data["unrelated"] == 1
    assert data["mcpServers"]["omnex"]["command"] == "omnex-mcp"


def test_rewrite_with_identical_entry_is_noop(home: Path) -> None:
    plan = build_client_install_plan("cursor", scope="user")
    first = write_client_install_plan(plan)
    content = first.read_text(encoding="utf-8")
    again = write_client_install_plan(plan)
    assert again == first
    assert again.read_text(encoding="utf-8") == content


def test_conflicting_existing_json_entry_raises(home: Path) -> None:
    target = home / ".cursor" / "mcp.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"mcpServers": {"omnex": {"command": "old"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="already configured"):
        write_client_install_plan(build_client_install_plan("cursor", scope="user"))
    assert json.loads(target.read_text(encoding="utf-8"))["mcpServers"]["omnex"]["command"] == "old"


def test_codex_appends_without_clobbering(home: Path) -> None:
    target = home / ".codex" / "config.toml"
    target.parent.mkdir(parents=True)
    target.write_text('[mcp_servers.other]\ncommand = "x"\nargs = []\n', encoding="utf-8")
    write_client_install_plan(build_client_install_plan("codex", scope="user"))
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["mcp_servers"]["other"]["command"] == "x"
    assert data["mcp_servers"]["omnex"]["command"] == "omnex-mcp"


def test_codex_rewrite_is_noop(home: Path) -> None:
    plan = build_client_install_plan("codex", scope="user")
    first = write_client_install_plan(plan)
    content = first.read_text(encoding="utf-8")
    write_client_install_plan(plan)
    assert first.read_text(encoding="utf-8") == content


def test_codex_conflicting_entry_raises(home: Path) -> None:
    target = home / ".codex" / "config.toml"
    target.parent.mkdir(parents=True)
    target.write_text('[mcp_servers.omnex]\ncommand = "old"\nargs = []\n', encoding="utf-8")
    with pytest.raises(ValueError, match="already configured"):
        write_client_install_plan(build_client_install_plan("codex", scope="user"))
    assert 'command = "old"' in target.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("client", "schema_marker"),
    [("opencode", "opencode.ai"), ("omp", "oh-my-pi")],
)
def test_client_schema_is_injected(client: ClientName, schema_marker: str, home: Path) -> None:
    target = write_client_install_plan(build_client_install_plan(client, scope="user"))
    data = json.loads(target.read_text(encoding="utf-8"))
    assert schema_marker in data["$schema"]


def test_resolve_scope_defaults_to_user_without_source() -> None:
    assert resolve_scope("claude-code", None, None) == "user"


def test_resolve_scope_selects_project_for_source() -> None:
    assert resolve_scope("claude-code", "/repo", None) == "project"


def test_resolve_scope_explicit_flag_wins_over_source() -> None:
    assert resolve_scope("claude-code", "/repo", "user") == "user"


@pytest.mark.parametrize("client", ["pi", "omp"])
def test_resolve_scope_user_only_clients_ignore_source(client: ClientName) -> None:
    assert resolve_scope(client, "/repo", None) == "user"


def test_command_writes_user_config(home: Path) -> None:
    result = CliRunner().invoke(main, ["install-client", "claude-code"])
    assert result.exit_code == 0, result.output
    target = home / ".claude.json"
    assert "Wrote claude-code config" in result.output
    assert str(target) in result.output
    assert _omnex_command("claude-code", target) == "omnex-mcp"


def test_command_default_scope_is_user(home: Path) -> None:
    result = CliRunner().invoke(main, ["install-client", "cursor"])
    assert result.exit_code == 0, result.output
    assert (home / ".cursor" / "mcp.json").exists()


def test_command_source_selects_project_scope(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    result = CliRunner().invoke(main, ["install-client", "cursor", str(root)])
    assert result.exit_code == 0, result.output
    assert _omnex_command("cursor", root / ".cursor" / "mcp.json") == "omnex-mcp"


def test_command_scope_flag_overrides_source(home: Path, tmp_path: Path) -> None:
    root = tmp_path.resolve()
    result = CliRunner().invoke(main, ["install-client", "cursor", str(root), "--scope", "user"])
    assert result.exit_code == 0, result.output
    assert (home / ".cursor" / "mcp.json").exists()
    assert not (root / ".cursor" / "mcp.json").exists()


def test_command_user_only_client_rejects_project_scope() -> None:
    result = CliRunner().invoke(main, ["install-client", "pi", "--scope", "project"])
    assert result.exit_code != 0
    assert "only --scope user" in result.output


def test_command_rejects_unknown_client() -> None:
    result = CliRunner().invoke(main, ["install-client", "nano"])
    assert result.exit_code != 0


def test_render_preview_includes_scope_target_and_content() -> None:
    plan = build_client_install_plan("claude-code", scope="user")
    preview = render_client_install_preview(plan)
    assert "Client: claude-code" in preview
    assert "Scope: user" in preview
    assert str(plan.target_path) in preview
    assert "omnex-mcp" in preview
    assert "Dry run." in preview


def test_dry_run_writes_no_file(home: Path) -> None:
    result = CliRunner().invoke(main, ["install-client", "claude-code", "--dry-run"])
    assert result.exit_code == 0, result.output
    target = home / ".claude.json"
    assert str(target) in result.output
    assert "omnex-mcp" in result.output
    assert "Dry run." in result.output
    assert not target.exists()


def test_dry_run_project_scope_writes_no_file(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    result = CliRunner().invoke(main, ["install-client", "cursor", str(root), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not (root / ".cursor" / "mcp.json").exists()


def test_dry_run_codex_previews_toml_without_writing(home: Path) -> None:
    result = CliRunner().invoke(main, ["install-client", "codex", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "[mcp_servers.omnex]" in result.output
    assert not (home / ".codex" / "config.toml").exists()
