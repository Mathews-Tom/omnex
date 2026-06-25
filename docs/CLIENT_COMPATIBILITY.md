# Client Compatibility

`omnex install-client <client> [SOURCE] [--scope user|project] [--dry-run] [--agent-file PATH]` writes or previews an MCP registration for one supported client. All six clients register the same stdio server entry: server name `omnex`, command `omnex-mcp`, and `args: []`.

`SOURCE` selects a repo-local target when the effective scope is `project`. Without `--dry-run`, `install-client` writes the config immediately. With `--dry-run`, it prints the resolved client, scope, target path, description, and exact content without writing anything.

## Compatibility matrix

| Client | Scopes | Target path(s) | Config shape | Notes |
| --- | --- | --- | --- | --- |
| `claude-code` | `user`, `project` | User: `~/.claude.json`<br>Project: `<repo>/.mcp.json` | JSON with top-level `mcpServers.omnex = { "command": "omnex-mcp", "args": [] }` | Shares the standard `mcpServers` shape. |
| `cursor` | `user`, `project` | User: `~/.cursor/mcp.json`<br>Project: `<repo>/.cursor/mcp.json` | JSON with top-level `mcpServers.omnex = { "command": "omnex-mcp", "args": [] }` | Shares the standard `mcpServers` shape. |
| `opencode` | `user`, `project` | User: `~/.config/opencode/opencode.json`<br>Project: `<repo>/opencode.json` | JSON with top-level `$schema`, then `mcp.omnex = { "type": "local", "command": ["omnex-mcp"], "enabled": true }` | Uses `mcp`, not `mcpServers`. |
| `codex` | `user`, `project` | User: `~/.codex/config.toml`<br>Project: `<repo>/.codex/config.toml` | TOML section `[mcp_servers.omnex]` with `command = "omnex-mcp"` and `args = []` | The only TOML target. |
| `pi` | `user` only | User: `~/.pi/agent/mcp.json`<br>Project: not supported | JSON with top-level `mcpServers.omnex = { "command": "omnex-mcp", "args": [] }` | User-only client. `install-client` rejects project scope. |
| `omp` | `user` only | User: `~/.omp/agent/mcp.json`<br>Project: not supported | JSON with top-level `$schema`, then `mcpServers.omnex = { "command": "omnex-mcp", "args": [] }` | User-only client. Adds the oh-my-pi MCP schema. |

## Exact config payloads

For `claude-code`, `cursor`, and `pi`, the written entry is the same; only the target path changes:

```json
{
  "mcpServers": {
    "omnex": {
      "command": "omnex-mcp",
      "args": []
    }
  }
}
```

For `omp`, the server entry is the same, with the oh-my-pi schema added at the top level:

```json
{
  "$schema": "https://raw.githubusercontent.com/can1357/oh-my-pi/main/packages/coding-agent/src/config/mcp-schema.json",
  "mcpServers": {
    "omnex": {
      "command": "omnex-mcp",
      "args": []
    }
  }
}
```

For `opencode`, the written JSON is:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "omnex": {
      "type": "local",
      "command": [
        "omnex-mcp"
      ],
      "enabled": true
    }
  }
}
```

For `codex`, the written TOML is:

```toml
[mcp_servers.omnex]
command = "omnex-mcp"
args = []
```

## Scope resolution

`install-client` resolves scope in this order:

1. An explicit `--scope` wins.
2. If the client is `pi` or `omp`, the effective scope is always `user`.
3. Otherwise, a provided `SOURCE` path implies `project` scope.
4. Otherwise, the default is `user`.

`pi` and `omp` are user-only clients. Asking for `--scope project` on either one fails instead of writing a repo-local file.

## Safe-write behavior

`install-client` writes by default. `--dry-run` previews the resolved target path and exact config content and writes nothing.

JSON clients merge the `omnex` entry into the existing server map without clobbering unrelated keys:

- `claude-code`, `cursor`, `pi`, and `omp` merge into `mcpServers`.
- `opencode` merges into `mcp`.
- For `opencode` and `omp`, the top-level `$schema` is added only when the existing file does not already declare one.

`codex` appends one `[mcp_servers.omnex]` section to `config.toml`.

Re-running `install-client` with an identical `omnex` entry is an idempotent no-op. If a different `omnex` entry is already present, omnex leaves that entry untouched and fails instead of overwriting it.

## `--agent-file` guidance block

`--agent-file PATH` appends a delimited guidance block to an agent file so a harness reaches for the omnex MCP tools first. The append is idempotent: if the start marker is already present, omnex does not add a second copy.

The exact delimiters are:

```text
<!-- omnex:mcp-guidance start -->
<!-- omnex:mcp-guidance end -->
```

Inside that block, omnex tells the agent to prefer the omnex MCP `index` and `query` tools for retrieval over docs, specs, and configs, to activate the omnex tools first in harnesses with on-demand tool discovery, and to verify with reads or tests before editing.

## Registration is not surfacing is not invocation

Writing a client config only registers the server. It does not guarantee that the client will surface the tools to the agent, and surfacing still does not guarantee that the agent will call them.

Those are three separate steps:

1. **Registration**: `install-client` writes the `omnex` MCP entry into the client config.
2. **Surfacing**: the client or harness exposes the registered tools to the running agent.
3. **Invocation**: the agent chooses the omnex MCP tools instead of reading files directly.

This distinction matters most in on-demand discovery harnesses such as Pi and oh-my-pi: the MCP server can be registered and discoverable, but still inactive until the session activates those tools. `--agent-file` helps nudge tool selection, but it does not force surfacing or invocation.

## Verification

After writing a client config:

1. Run `omnex doctor` to check MCP registration health.
2. Inspect the written file directly at the resolved target path for that client and scope.
3. If you want to preview before changing anything, use `--dry-run` first.

Examples:

```bash
omnex install-client claude-code --dry-run
omnex install-client cursor . --scope project
omnex doctor
```