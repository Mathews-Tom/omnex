# Installation

This document states what omnex needs to run, how to install it today, what it reads and writes, when it can use the network, and how to remove it. It covers the CLI, Python library, MCP server, and local Docker images.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)

omnex is not published to PyPI yet. PyPI publication is on the roadmap, so installs use the GitHub git URL today.

## Install methods

### CLI tool install

```bash
# Core install: CLI + Python library, with the byte-exact default path.
uv tool install "omnex @ git+https://github.com/Mathews-Tom/omnex"

# Optional T2 local embedding lane.
uv tool install "omnex[embed] @ git+https://github.com/Mathews-Tom/omnex"

# Optional stdio MCP server.
uv tool install "omnex[mcp] @ git+https://github.com/Mathews-Tom/omnex"

# Optional LangChain retriever integration.
uv tool install "omnex[langchain] @ git+https://github.com/Mathews-Tom/omnex"

# Optional LlamaIndex retriever integration.
uv tool install "omnex[llamaindex] @ git+https://github.com/Mathews-Tom/omnex"
```

### Project dependency install

If you want omnex in a project environment instead of a global tool, use `uv add` with the same git URL spec:

```bash
uv add "omnex @ git+https://github.com/Mathews-Tom/omnex"
uv add "omnex[mcp] @ git+https://github.com/Mathews-Tom/omnex"
```

The same `uv add` pattern works for the optional `embed`, `langchain`, and `llamaindex` extras.

## Extras matrix

| Install target | Pulls in | Enables |
| --- | --- | --- |
| Core install | `networkx`, `tiktoken`, `click` | The core CLI and Python library surfaces |
| `omnex[embed]` | `fastembed` | The opt-in T2 local embedding lane |
| `omnex[mcp]` | `mcp` | The `omnex-mcp` stdio server |
| `omnex[langchain]` | `langchain-core` | `omnex.integrations.langchain.OmnexRetriever` for LangChain-based RAG |
| `omnex[llamaindex]` | `llama-index-core` | `omnex.integrations.llamaindex.OmnexLlamaRetriever` for LlamaIndex-based RAG |
| `omnex[bench]` | `omnex[embed]` | The `omnex-bench` chunk-and-embed baseline for benchmarking; never a product path |

The core install is intentionally small: its only dependencies are `networkx`, `tiktoken`, and `click`. `import omnex` loads no model, opens no socket, and reads no file.

## Docker

omnex ships two local Docker build targets at the repo root. Neither image is published to a registry.

```bash
# Core-only image.
docker build -f Dockerfile.slim -t omnex:slim .

# Core + [embed] + [mcp].
docker build -f Dockerfile.full -t omnex:full .
```

Both images run as an unprivileged user, expect you to mount a corpus at runtime, and use `omnex` as the `ENTRYPOINT`.

```bash
# Query a mounted corpus with the slim image.
docker run --rm -v "$PWD:/work:ro" omnex:slim query /work/spec.json "create payment" --budget 400

# Run any omnex subcommand from the full image.
docker run --rm -v "$PWD:/work:ro" omnex:full doctor
```

For end-to-end usage examples, see [USAGE.md](USAGE.md). For the repo-level Docker overview, see the [README Docker section](../README.md#docker).

## MCP registration

If you want omnex available through MCP, install the `[mcp]` extra and register the `omnex-mcp` stdio server with `omnex install-client <client>`. omnex supports six client targets (`claude-code`, `codex`, `cursor`, `opencode`, `pi`, and `omp`), writes the client-specific config for you, supports `--dry-run`, and can append an idempotent agent-guidance block with `--agent-file`. The client-by-client target paths and config shapes live in [CLIENT_COMPATIBILITY.md](CLIENT_COMPATIBILITY.md).

## Verify the install

Use `omnex doctor` after installation:

```bash
omnex doctor
omnex doctor --strict
```

`omnex doctor` reports installation and operational health across MCP registration, usage-metrics state, installed extras, adapter sanity, and persistence mode. `--strict` turns it into a gate by exiting non-zero when any check is not `ok`.

## What omnex reads and writes

omnex is stateless for retrieval. `index` and `query` build their in-memory index and structure graph on each call and discard them afterward. There is no repo-local `.omnex/` index, no persisted retrieval cache, and no `init` or `status` lifecycle.

### Reads

| What | When |
| --- | --- |
| The file or directory paths you pass to `omnex index` or `omnex query` | To route sources through adapters and answer the request |
| Installed-client config files | When you run `omnex install-client` or `omnex doctor` |
| `~/.omnex/settings.json` and `~/.omnex/usage.sqlite` | Only for the opt-in local metrics and health reporting paths |

### Writes

| What | When |
| --- | --- |
| `~/.omnex/settings.json` | When you opt in or out of local usage metrics settings |
| `~/.omnex/usage.sqlite` | Only when local usage metrics recording is enabled |
| The chosen MCP client config | When you run `omnex install-client` without `--dry-run` |
| The file passed to `--agent-file` | When you ask `install-client` to append the idempotent omnex MCP guidance block |

Retrieval itself persists nothing.

## Network behavior by feature

| Feature | Network behavior |
| --- | --- |
| Core CLI | No network |
| Core Python library | No network |
| MCP server (`omnex-mcp`) | No network |
| Docker slim image | No network |
| Docker full image | No network for core commands; using the T2 embedding lane inside it follows the same first-use FastEmbed download behavior as `omnex[embed]` |
| `omnex[embed]` | May download the FastEmbed model into its local cache on first use |
| `omnex[bench]` | May download the FastEmbed model into its local cache on first use because it pulls `omnex[embed]` |
| `install-client` and `--agent-file` | No network |
| Telemetry | None |

There is no hosted inference path, no telemetry upload path, and no remote-code toggle in omnex. The only model-download path is the opt-in FastEmbed cache behind `[embed]` or `[bench]`.

## Uninstall

```bash
uv tool uninstall omnex
rm -rf ~/.omnex
```

Then remove the `omnex` entry from any client config you registered with `omnex install-client`.