# omnex — Roadmap

> This roadmap separates what omnex already ships in `0.1.0` from what is intentionally still ahead. Current surface details live in [README.md](../README.md); the architecture and invariants live in [system-design.md](system-design.md).

## How to read this roadmap

| Status | Meaning |
| --- | --- |
| **Shipped** | Present in `0.1.0` alpha today. |
| **Planned** | Explicitly on the roadmap, but not shipped yet. |

omnex is intentionally conservative about roadmap promises. This document makes no dated commitments; it records what exists, what is next, and which product decisions new work must preserve.

## Shipped in 0.1.0 (alpha)

### Core retrieval system

| Area | Shipped in `0.1.0` |
| --- | --- |
| **Modality-agnostic IR** | One intermediate representation for retrieval across structured corpora: `Document`, `Unit`, `Reference`, and `StructureGraph`. |
| **Shipped adapters** | Spec and prose adapters are the production adapters today. Specs map to structural units and hard edges that support deterministic closure; prose maps to section-oriented units and cross-reference-style edges. |
| **Modality-blind kernel** | The kernel runs over the IR rather than over source-format-specific code paths: SQLite FTS5 with BM25F scoring, reciprocal-rank and relative-score fusion, bounded graph expansion, deterministic T1 closure over hard edges, and the efficiency packer. |
| **Whole-unit packing** | Retrieval and packing operate on whole units, not arbitrary fragments, with deterministic `INCLUDE → COMPRESS → ELIDE → SKIP` decisions under budget pressure. |

### Tiered retrieval lanes

| Tier | Status in `0.1.0` | Notes |
| --- | --- | --- |
| **T0** | Shipped | The default floor: lexical retrieval plus the packer, byte-exact and model-free. |
| **T1** | Shipped | Adds deterministic graph closure for structured corpora while staying byte-exact and model-free. |
| **T2** | Shipped, opt-in | Adds the local vector lane for prose and natural-language recall. It is available behind the `[embed]` extra and is labeled as a different determinism class from T0/T1. |

### Provenance, benchmarks, and operator surfaces

| Area | Shipped in `0.1.0` |
| --- | --- |
| **Receipts** | Every query returns a `Receipt` alongside the packed bundle, so callers can inspect tier usage, determinism class, token counts, recall basis, and whether a reference closure was both computed and fully emitted. |
| **Benchmark families** | Labeled benchmark families for specs and prose are part of the shipped proof surface. |
| **Library surface** | The Python library is the source-of-truth surface for indexing and querying in memory. |
| **CLI surface** | The `omnex` CLI ships with `index`, `query`, `install-client`, `metrics`, and `doctor`. |
| **MCP surface** | The `omnex-mcp` server ships behind the `[mcp]` extra and exposes the same byte-exact T0 query behavior through MCP. |
| **Docker packaging** | Slim and full Docker images ship as local build targets. Slim carries the core CLI; full adds the optional extras used for embeddings and MCP. |
| **Framework integrations** | LangChain and LlamaIndex retrievers ship as optional integrations that preserve omnex ranking and receipt provenance. |
| **Adoption layer** | Cross-client `install-client` registration ships for claude-code, codex, cursor, opencode, pi, and omp. |
| **Local usage metrics** | Local usage metrics ship as an off-by-default, CLI-only capability. |
| **Diagnostics** | `doctor` ships as the installation and operational health check surface. |

### Persistence model

`0.1.0` also locks in a product decision: omnex is **stateless**. Indexing and querying build the FTS index and `StructureGraph` in memory per call and discard them when the call returns. That decision is already reflected in the shipped surfaces, receipt semantics, and diagnostics behavior.

## Planned (no dates)

The roadmap items below are explicit next steps, but none of them are part of `0.1.0` yet.

| Planned item | Scope |
| --- | --- |
| **T3 model-extraction lane** | Adapter-local OCR, caption, and transcription paths for sources that cannot enter the IR as usable text today. |
| **Code adapter** | A tree-sitter-backed adapter that emits `FUNCTION` and `CLASS` units plus `IMPORTS` and `CALLS` edges. This is the seam for an eventual archex-on-omnex migration, not a shipped capability yet. |
| **Mixed-corpus cross-modality linking** | Linking across prose and code so one corpus can recover relationships that span both modalities. |
| **PyPI publication** | Publishing omnex so `uv tool install omnex` works without a Git URL. |

## Principles guiding the roadmap

| Principle | What it means for future work |
| --- | --- |
| **Determinism is never regressed** | T0 and T1 stay byte-exact. Any new lane that weakens the determinism guarantee stays opt-in and is labeled clearly in the `Receipt`; the deterministic headline does not bleed across tiers. |
| **The core/extras boundary holds** | The core install stays narrow: `networkx`, `tiktoken`, and `click` remain the core dependency boundary, while embeddings, MCP, and framework adapters stay in extras. |
| **Statelessness is a decision, not an accident** | Roadmap work must fit the stateless persistence model already chosen for omnex rather than quietly reintroducing a persisted indexing lifecycle. |

## What this means in practice

The shipped alpha is already the full first proof: a modality-blind retrieval engine with deterministic structured retrieval, an opt-in vector assist for prose, auditable receipts, labeled benchmark families, and multiple ways to adopt the system. The planned work expands modality coverage and distribution, but it does so under the same three constraints: keep deterministic tiers honest, keep optional capabilities optional, and keep the product stateless.