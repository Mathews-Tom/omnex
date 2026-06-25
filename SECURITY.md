# Security Policy

## Security posture and threat model

omnex is designed to keep its default trust boundary small.

- The core product path is local-first, stateless, and deterministic.
- The core retrieval surfaces — the Python library, the `omnex` CLI, the `omnex-mcp` stdio server, and the locally built Docker slim image — do not require hosted inference, API keys, or any network call.
- The core install depends only on `networkx`, `tiktoken`, and `click`.
- `import omnex` is side-effect free on its import path: it loads no model, opens no socket, and reads no file.
- T0 and T1 runs are model-free and byte-exact for the same corpus, config, and query. The CLI and MCP surfaces use the fixed T0 floor by default.
- Usage metrics are off by default, local-only, and CLI-only. They do not upload data, do not start a background process, and do not expose metrics controls through MCP.
- omnex has no telemetry path in core operation and does not upload corpora, bundles, or receipts to a hosted service.

The only optional network path is the opt-in `[embed]` extra, which the `[bench]` extra also pulls in. That path enables omnex's local `fastembed`-backed T2 vector lane or the benchmark embedder. The model is loaded lazily on first use, and any third-party weight download or cache population happens on the local machine through that dependency stack rather than through a hosted omnex service. omnex does not upload corpus data or telemetry as part of that path.

## What omnex writes

omnex is stateless. `index` and `query` build their in-memory index and graph for each call and then discard them. omnex does not create a repo-local `.omnex/` index, an on-disk retrieval cache, or a lifecycle state directory.

The only persistent writes are:

| Path or target | When it is written | Notes |
| --- | --- | --- |
| `~/.omnex/usage.sqlite` | Only after the operator opts into usage metrics and a metrics write occurs | Local SQLite ledger; never created on a default-off install. `OMNEX_HOME` can relocate the home directory. |
| `~/.omnex/settings.json` | When the operator persists metrics or trace settings | Local settings only. `OMNEX_HOME` can relocate the home directory. |
| The selected MCP client config file | When `omnex install-client` writes or merges the `omnex` server entry | This is local client configuration, not omnex retrieval state. |
| An optional agent file passed with `--agent-file` | When `omnex install-client --agent-file ...` appends the guidance block | The append is delimited and idempotent. |

If you run `omnex install-client --dry-run`, omnex resolves and previews the target configuration but writes nothing.

## Supported versions

The current line receives security fixes.

| Version | Status |
| --- | --- |
| `0.1.0` (alpha) | Supported current line |

## Reporting vulnerabilities

Please report suspected vulnerabilities privately through GitHub Security Advisories for the `Mathews-Tom/omnex` repository:

- <https://github.com/Mathews-Tom/omnex/security/advisories/new>

Do not file public GitHub issues for unpatched security reports.

Include, at minimum:

- the affected omnex version or commit
- the surface involved (`omnex`, `omnex-mcp`, Python API, or Docker image)
- a minimal reproduction
- observed impact
- any known workaround

## Scope notes

The optional `[embed]` extra, and anything that depends on it such as `[bench]`, uses third-party model weights outside omnex's core trust boundary. Weight download and cache behavior for that path follow the standard `fastembed` and Hugging Face cache behavior on the local machine, not a special omnex-managed storage or upload service.
