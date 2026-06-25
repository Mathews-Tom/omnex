# Usage

This guide covers every shipped omnex surface: the CLI, the Python API, the stdio MCP server, and the LangChain and LlamaIndex retriever adapters. It assumes omnex is already installed; use the repo README for installation paths and extras.

## Quickstart

The CLI is the fastest way to validate a corpus and retrieve a token-budgeted bundle plus receipt. `index` accepts one or more file or directory paths. `query` accepts one file or directory as the corpus, runs the fixed T0 floor, and renders either Markdown or JSON.

| Task | Command | What you get |
| --- | --- | --- |
| Index one file | `omnex index tests/fixtures/payments_openapi.json` | Corpus counts: documents, units, references |
| Index one directory | `omnex index tests/fixtures/tls_docs` | The same validation and counts across every file in the directory |
| Query one file as Markdown | `omnex query tests/fixtures/payments_openapi.json "create payment" --budget 400 --format markdown` | `ContextBundle.render()` output followed by a human-readable receipt |
| Query one directory as JSON | `omnex query tests/fixtures/tls_docs "configure TLS for the ingress" --budget 2000 --format json` | A structured `{bundle, receipt}` payload |

A few useful defaults to remember:

- `--budget` defaults to `4000`.
- `--format` defaults to `markdown`.
- The CLI is stateless: every `index` and `query` call rebuilds the in-memory index and graph, then discards them.
- For a fixed corpus, question, budget, and output format, `query` is deterministic.

## CLI reference

### `omnex index`

Syntax:

```bash
omnex index PATHS...
```

What it does:

- Routes each path through its claiming adapter.
- Builds the FTS index and `StructureGraph` to validate the full indexing path.
- Prints `indexed N document(s), M unit(s), K reference(s)`.
- Persists nothing.

Use it when you want a quick sanity check that omnex can ingest the corpus before you start querying it.

### `omnex query`

Syntax:

```bash
omnex query CORPUS QUESTION --budget INT --format markdown
omnex query CORPUS QUESTION --budget INT --format json
```

What it does:

- Accepts a single file or directory as `CORPUS`.
- Expands that corpus to files, routes them through adapters, and runs the same fixed T0 `KernelConfig` the CLI and MCP server share.
- Returns the same retrieval, ranking, and returned set the library produced; the CLI only renders them.

Output formats:

| Format | Shape |
| --- | --- |
| `markdown` | `bundle.render()`, then `## Receipt`, then one bullet per receipt field; when lexical recall caveats apply, a `### Recall limitations` section is appended |
| `json` | `{"bundle": {"context", "total_tokens", "representations"}, "receipt": {...}}` |

For the receipt fields and how to interpret them, see [RECEIPTS.md](RECEIPTS.md).

### `omnex doctor`

Syntax:

```bash
omnex doctor
omnex doctor --format json
omnex doctor --strict
```

What it reports:

- MCP registration status
- Usage-metrics ledger state
- Installed extras
- Adapter sanity
- Persistence mode (`stateless`)

Notes:

- `--format json` emits a stable object with top-level `checks`, `healthy`, and `status` keys.
- `--strict` exits non-zero when any check is not `ok`.
- Text output uses `[ok]` and `[warn]` prefixes.

For the retrieval-side audit trail that `doctor` complements, see [RECEIPTS.md](RECEIPTS.md).

### `omnex metrics`

Syntax:

```bash
omnex metrics enable --on
omnex metrics enable --off
omnex metrics trace --on
omnex metrics summary --format text
omnex metrics summary --format json
omnex metrics export
omnex metrics delete --yes
```

This command group is CLI-only and off by default. It manages the local anonymous usage ledger; the MCP server does not expose metrics tools. For ledger paths, environment variables, the stored fields, and how savings are computed, see [LOCAL_METRICS.md](LOCAL_METRICS.md).

### `omnex install-client`

Syntax:

```bash
omnex install-client CLIENT [SOURCE] [--scope user|project] [--dry-run] [--agent-file PATH]
```

Supported clients:

- `claude-code`
- `codex`
- `cursor`
- `opencode`
- `pi`
- `omp`

What it does:

- Writes the MCP client configuration that registers the `omnex-mcp` stdio server.
- Merges the `omnex` entry into the client's existing config without clobbering unrelated sections.
- Treats `SOURCE` as the repo root for a project-scope install; without a source path, scope resolves to user unless `--scope project` is set.
- Prints the resolved target and config without writing when `--dry-run` is set.
- Appends an idempotent guidance block to an agent file when `--agent-file PATH` is set.
- `pi` and `omp` are user-scope only.

For per-client target paths, config shapes, scope rules, and registration caveats, see [CLIENT_COMPATIBILITY.md](CLIENT_COMPATIBILITY.md).

## Python API

The library is the source of truth for omnex behavior. The CLI and MCP server are thin surfaces over it.

Important differences from the CLI and MCP server:

- The library exposes no implicit default config. Every retrieval run must supply a `KernelConfig` explicitly.
- `query(...)` and `query_sources(...)` are one-shot helpers that build a kernel, retrieve once, and return `(ContextBundle, Receipt)`.
- `index(...)` and `index_sources(...)` build a reusable `RetrievalKernel`; call `kernel.retrieve(question, budget_tokens, config)` for repeated queries over the same corpus.
- `query_sources(...)` rewrites `receipt.baseline_tokens` to the true whole-document dump of the indexed source files.

### Build an explicit `KernelConfig`

If you want byte-exact parity with the CLI and MCP server, construct the same T0 floor they use internally:

```python
from omnex import KernelConfig

cfg_t0 = KernelConfig(
    tier="T0",
    bm25_profile={"title": 2.0, "breadcrumb": 1.5, "text": 1.0, "summary": 1.0},
    hop_budget_by_kind={
        "CONTAINS": 2,
        "SIBLING": 0,
        "CROSS_REF": 1,
        "CITES": 1,
        "LINKS_TO": 1,
        "REFERENCES": 1,
        "FOREIGN_KEY": 1,
        "IMPORTS": 1,
        "CALLS": 1,
    },
    confidence_decay=0.8,
    enable_vector_lane=False,
    enable_rerank=False,
)
```
A spec-focused T1 config can reuse the same BM25F weights and hop budgets as the surface floor, while switching the tier to `T1` so the kernel may compute deterministic closure over spec edges:

```python
from omnex import KernelConfig

cfg_t1_spec = KernelConfig(
    tier="T1",
    bm25_profile={"title": 2.0, "breadcrumb": 1.5, "text": 1.0, "summary": 1.0},
    hop_budget_by_kind={
        "CONTAINS": 2,
        "SIBLING": 0,
        "CROSS_REF": 1,
        "CITES": 1,
        "LINKS_TO": 1,
        "REFERENCES": 1,
        "FOREIGN_KEY": 1,
        "IMPORTS": 1,
        "CALLS": 1,
    },
    confidence_decay=0.8,
    enable_vector_lane=False,
    enable_rerank=False,
)
```

A prose-focused T2 config starts from the same explicit floor and opts into the local vector lane. This requires the `[embed]` extra.

```python
from omnex import KernelConfig

cfg_t2_prose = KernelConfig(
    tier="T2",
    bm25_profile={"title": 2.0, "breadcrumb": 1.5, "text": 1.0, "summary": 1.0},
    hop_budget_by_kind={
        "CONTAINS": 2,
        "SIBLING": 0,
        "CROSS_REF": 1,
        "CITES": 1,
        "LINKS_TO": 1,
        "REFERENCES": 1,
        "FOREIGN_KEY": 1,
        "IMPORTS": 1,
        "CALLS": 1,
    },
    confidence_decay=0.8,
    enable_vector_lane=True,
    enable_rerank=False,
)
```

### In-memory IR: `index(...)` and `query(...)`

Use `query(...)` when you already have `Unit` and `Reference` objects in memory and only need one retrieval. Use `index(...)` when you want to build the kernel once and reuse it.

```python
from omnex import Reference, Span, Unit, index, query

operation = Unit(
    id="unit:create-payment",
    document_id="doc:payments",
    span=Span(0, 52),
    text="Create payment operation accepts a PaymentRequest body.",
    token_count=7,
    title="payments",
    breadcrumb=("paths", "POST"),
    kind="OPERATION",
    summary="Create payment",
    protect=False,
)
schema = Unit(
    id="unit:payment-request",
    document_id="doc:payments",
    span=Span(53, 100),
    text="PaymentRequest requires amount and customer fields.",
    token_count=6,
    title="PaymentRequest",
    breadcrumb=("components", "schemas"),
    kind="SCHEMA",
    summary="Payment request schema",
    protect=False,
)
references = [
    Reference(
        source_id=operation.id,
        target_id=schema.id,
        kind="REFERENCES",
        confidence=1.0,
        evidence=("$ref -> #/components/schemas/PaymentRequest",),
    )
]
corpus = [operation, schema]

bundle, receipt = query(corpus, "create payment", 120, cfg_t0, references)
print(bundle.render())
print(receipt.tiers_run)

kernel = index(corpus, references)
follow_up_bundle, follow_up_receipt = kernel.retrieve("customer fields", 120, cfg_t0)
print(follow_up_bundle.total_tokens)
print(follow_up_receipt.determinism_class)
```

### Path-routed sources: `index_sources(...)` and `query_sources(...)`

Use `index_sources(...)` and `query_sources(...)` when you want omnex to route source files through their adapters first.

T1 over a shipped OpenAPI fixture:

```python
from pathlib import Path
from omnex import index_sources, query_sources

spec_sources = [Path("tests/fixtures/payments_openapi.json")]

spec_kernel = index_sources(spec_sources)
spec_bundle, spec_receipt = spec_kernel.retrieve("create payment", 400, cfg_t1_spec)
print(spec_receipt.tiers_run)
print(spec_receipt.reference_closure_complete)

spec_bundle_once, spec_receipt_once = query_sources(spec_sources, "create payment", 400, cfg_t1_spec)
print(spec_receipt_once.baseline_tokens)  # true whole-document dump, not the sum of unit texts
```

T2 over a prose corpus, with explicit file paths and the opt-in vector lane:

```python
from pathlib import Path
from omnex import index_sources

prose_sources = [
    Path("tests/fixtures/tls_docs/ingress.md"),
    Path("tests/fixtures/tls_docs/securing-traffic.md"),
    Path("tests/fixtures/tls_docs/service-discovery.md"),
]

prose_kernel = index_sources(prose_sources)
prose_bundle, prose_receipt = prose_kernel.retrieve(
    "configure TLS for the ingress",
    2000,
    cfg_t2_prose,
)
print(prose_receipt.recall_basis)
print(prose_receipt.embedding_provenance)
```

## MCP server

The `omnex-mcp` console script starts the FastMCP stdio server. Its core install stays separate from `import omnex`; install the `[mcp]` extra before you try to run or register it.

```bash
omnex-mcp
```

It exposes exactly two tools:

| Tool | Signature | Returns |
| --- | --- | --- |
| `index` | `index(paths: list[str])` | `{documents, units, references}` |
| `query` | `query(corpus: str, question: str, budget: int = 4000)` | The same `{bundle, receipt}` payload the CLI emits in JSON mode |

`query(...)` runs the same byte-exact T0 bundle-plus-receipt path as the CLI and the library surface configuration: identical bundle render, identical per-unit representations, and identical receipt content.

For client registration, repo-local versus user scope, and the generated config files, see [CLIENT_COMPATIBILITY.md](CLIENT_COMPATIBILITY.md).

## RAG framework retrievers

omnex ships thin retriever adapters behind extras. Both take a prebuilt `RetrievalKernel`, a `KernelConfig`, and a per-query `budget_tokens`. Both preserve omnex's packed order and attach provenance metadata under `omnex_receipt`.
These integrations are not imported by `import omnex`; importing an integration module without its extra fails loud with `ImportError`.

### LangChain: `OmnexRetriever`

Requires the `[langchain]` extra.

```python
from pathlib import Path
from omnex import index_sources
from omnex.integrations.langchain import OmnexRetriever

kernel = index_sources([Path("tests/fixtures/tls_docs/ingress.md")])
retriever = OmnexRetriever(kernel=kernel, config=cfg_t0, budget_tokens=300)

docs = retriever.invoke("configure TLS for the ingress")
print(docs[0].page_content)
print(docs[0].metadata["omnex_receipt"]["determinism_class"])
```

Each returned `langchain_core.documents.Document` carries these metadata keys: `unit_id`, `mode`, `token_count`, `title`, `breadcrumb`, `kind`, `document_id`, and `omnex_receipt`.

### LlamaIndex: `OmnexLlamaRetriever`

Requires the `[llamaindex]` extra.

```python
from pathlib import Path
from omnex import index_sources
from omnex.integrations.llamaindex import OmnexLlamaRetriever

kernel = index_sources([Path("tests/fixtures/payments_openapi.json")])
retriever = OmnexLlamaRetriever(kernel=kernel, config=cfg_t1_spec, budget_tokens=400)

nodes = retriever.retrieve("create payment")
print(nodes[0].node.text)
print(nodes[0].node.metadata["omnex_receipt"]["tiers_run"])
```

Each returned `NodeWithScore` wraps a `TextNode` whose `text` is the packed chunk text and whose metadata includes the same unit provenance plus `omnex_receipt`.

## Reading results

Every omnex query returns two things: a rendered bundle and a receipt.

- `ContextBundle.render()` is the human-readable bundle text. Prose units render as Markdown with breadcrumb context, spec units render as canonical path-qualified fragments, and skipped representations stay out of the render.
- The receipt is the audit trail: returned tokens, full-dump baseline, tiers run, determinism class, closure completeness, recall basis, and optional embedding provenance.
- CLI Markdown output is `bundle.render()` followed by `## Receipt`; lexical-only runs add `### Recall limitations`.
- CLI JSON output and the MCP `query(...)` tool return the structured payload with `bundle.context`, `bundle.total_tokens`, `bundle.representations`, and `receipt`.

For a field-by-field walkthrough of the receipt contract, see [RECEIPTS.md](RECEIPTS.md).
