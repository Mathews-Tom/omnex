# Contributing to omnex

Thank you for contributing to omnex. This guide covers the local workflow, the required gates, the coding standards enforced in this repository, and the high-level adapter contract.

For end-user setup and optional extras, see [docs/INSTALLATION.md](docs/INSTALLATION.md). For examples of the library, CLI, and MCP surfaces, see [docs/USAGE.md](docs/USAGE.md).

## Getting started

Clone the repository and install the development environment with `uv`:

```bash
git clone https://github.com/Mathews-Tom/omnex
cd omnex
uv sync
```

The repository targets Python 3.12+, and the repo standards assume Python 3.12 syntax and typing throughout.

## The gates

Run these before pushing. CI enforces the same four gates on every push and PR.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/omnex tests
uv run pytest
```

What each gate covers:

| Gate | Purpose |
| --- | --- |
| `uv run ruff check .` | Linting and import hygiene |
| `uv run ruff format --check .` | Formatting compliance |
| `uv run mypy --strict src/omnex tests` | Strict type checking for the package and test suite |
| `uv run pytest` | Full test suite |

CI also runs a separate Docker workflow that builds both local images (`Dockerfile.slim` and `Dockerfile.full`) and smoke-runs the slim image. You do not need a separate local command for that here, but changes that affect packaging or entry points should be made with that workflow in mind.

## Coding standards

The repository standards are intentionally narrow:

- **Python version:** Python 3.12+
- **Imports:** every first-party Python module should include `from __future__ import annotations`
- **Typing style:** use built-in generics (`list[str]`, `dict[str, int]`) and `|` unions
- **Type checking:** code should satisfy `mypy --strict`
- **Core dependency boundary:** the core package stays on `networkx`, `tiktoken`, and `click`
- **Optional features:** anything beyond the core boundary belongs behind an extra; see [docs/INSTALLATION.md](docs/INSTALLATION.md) for the supported extras
- **Project boundary:** do not introduce an `archex` import or dependency; omnex is independent

A few practical implications follow from that boundary:

- `import omnex` must stay lightweight and must not pull optional frameworks into the core import path.
- Framework integrations such as LangChain and LlamaIndex stay behind extras and are tested that way.


## Commit and pull request conventions

Keep contributions easy to review and easy to revert.

- Use **Conventional Commits** for commit messages and PR titles (for example, `feat: add prose adapter edge case coverage`).
- Keep commits **atomic**: one logical change per commit.
- Do not add attribution of any kind in commits, PRs, source files, or docs.
- Before opening or updating a PR, make sure all four gates are green locally.
- PRs should stay green on the repository gates before merge.

## Tests

The test suite lives under `tests/` and mostly mirrors the package structure where that helps clarity.

| Location | What lives there |
| --- | --- |
| `tests/kernel/` | Kernel behavior, packing, fusion, closure, receipts, and vector-lane coverage |
| `tests/adapters/` | Adapter contract and adapter-specific parse/link coverage |
| `tests/ir/` | IR type and graph coverage |
| `tests/bench/` | Benchmark runner and baseline coverage |
| `tests/fixtures/` | Test inputs used by adapter and end-to-end tests |
| top-level `tests/test_*.py` | Public API, CLI, MCP, metrics, doctor, end-to-end, and integration-facing tests |

Run the full suite with:

```bash
uv run pytest
```

Run a focused file with:

```bash
uv run pytest tests/test_api.py
```

Optional-feature tests are extras-gated and auto-skip when the extra is not installed. For example, the framework integration tests in `tests/test_integrations.py` skip unless the matching extra is present. To run them explicitly, install the extra for that invocation:

```bash
uv run --extra langchain pytest tests/test_integrations.py
```

The same pattern applies to other optional surfaces when you are changing them.

## Adding a modality adapter

Adapters are the boundary between source files and omnex's modality-agnostic IR. The contract lives in [`src/omnex/adapters/base.py`](src/omnex/adapters/base.py).

At a high level, a new adapter should:

1. Implement the `ModalityAdapter` Protocol.
2. Route with `claims(source)`.
3. Establish document identity with `ingest(source)`.
4. Emit retrievable IR units with `parse(document)`.
5. Recover typed edges with `link(document, units)`.
6. Report capabilities with `capabilities()`.

The important constraint is architectural: adapters are modality-specific, but the kernel is not. An adapter must emit the shared IR (`Document`, `Unit`, and `Reference`) and keep modality-specific parsing logic inside the adapter boundary.

After implementing the adapter, register it in the adapter routing registry in `src/omnex/adapters/__init__.py` so `select_adapter(...)` can dispatch to it in the right priority order.

When you add or change an adapter:

- extend adapter-focused coverage under `tests/adapters/`
- add or update fixtures under `tests/fixtures/` when needed
- add end-to-end coverage when the change affects routed source handling

If you are touching optional integration behavior or installation surfaces around that adapter work, cross-check the relevant user-facing expectations in [docs/USAGE.md](docs/USAGE.md) and [docs/INSTALLATION.md](docs/INSTALLATION.md).
