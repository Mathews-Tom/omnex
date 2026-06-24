"""Shared internals for the omnex surfaces (the CLI and the MCP server).

Both thin surfaces need the same retrieval config, corpus collection, index
core, and result serialization. They live here -- depended on by both surfaces,
so neither surface depends on the other and there is one source of truth for the
structured query result. This module is pure: it imports no surface framework
(no ``click``, no ``mcp``) and changes no retrieval ranking, returned set, or
receipt schema; it only routes, configures, and shapes results from
:mod:`omnex.api`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path

from omnex import ContextBundle, KernelConfig, Receipt, api

# Default token budget for a query when a surface caller gives none. A query is a
# budgeted retrieval, so a surface always passes a concrete budget through to the
# kernel rather than leaving it implicit. 4000 is a generous single-query context
# window; callers override it per query.
_DEFAULT_BUDGET = 4000


def default_config() -> KernelConfig:
    """The surfaces' fixed retrieval config: the byte-exact, model-free T0 floor.

    The config is modality-agnostic: the BM25F profile names every indexed FTS
    column and the hop budget names every reference kind an adapter can emit, so
    one config serves prose and spec corpora identically. A surface never picks a
    tier or tunes ranking per query -- the T0 floor is the documented default.
    The library intentionally exposes no global default config (every run must
    state one), so owning this surface default is the surfaces' job, not a
    ranking-policy change. It is exposed so callers, and the parity tests, can
    drive the library with the exact config the surfaces use.
    """
    return KernelConfig(
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


def collect_files(paths: Iterable[Path]) -> list[Path]:
    """Expand each path to files: a file is itself, a directory its sorted files.

    A directory is walked recursively and its files are sorted, so the routed
    order -- and therefore the packed output -- is stable for a fixed corpus, as
    the deterministic-output contract requires. Hidden files and anything under a
    hidden directory (a leading-dot path part, e.g. ``.git`` or ``.DS_Store``)
    are skipped so directory indexing is practical; a non-hidden source that no
    adapter claims still fails loud. Explicit file arguments are kept in the
    caller's given order and never filtered. A path that does not exist fails
    loud here, so the surfaces report a missing path uniformly rather than
    letting it reach the adapters as a misleading "unclaimable" error.
    """
    collected: list[Path] = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"path does not exist: {path}")
        if path.is_dir():
            collected.extend(
                sorted(
                    file
                    for file in path.rglob("*")
                    if file.is_file()
                    and not any(part.startswith(".") for part in file.relative_to(path).parts)
                )
            )
        else:
            collected.append(path)
    return collected


def index_corpus(sources: Sequence[Path]) -> tuple[int, int, int]:
    """Route SOURCES into IR, build the index/graph to validate, return counts.

    The shared core of both surfaces' index operation: routes each source
    (failing loud when no adapter claims it), then builds the FTS index and
    StructureGraph so a corpus that routes but cannot be indexed (e.g. a malformed
    edge) fails here rather than silently at query time. Returns the
    ``(documents, units, references)`` counts. No state is persisted. An empty
    corpus (no claimable files, e.g. a directory of only hidden files) fails loud
    rather than reporting a meaningless zero shape.
    """
    if not sources:
        raise ValueError("corpus is empty: no indexable files found")
    units, references, documents = api._route_sources(sources)
    api.index(units, references)
    return len(documents), len(units), len(references)


def receipt_dict(receipt: Receipt) -> dict[str, object]:
    """The receipt as a JSON-serializable mapping, with its recall caveats.

    The receipt schema is unchanged: this serializes its fields verbatim and
    appends the derived ``recall_limitations`` so the rendered audit trail carries
    the same honesty caveats the library exposes.
    """
    data: dict[str, object] = dataclasses.asdict(receipt)
    data["recall_limitations"] = list(receipt.recall_limitations)
    return data


def result_payload(bundle: ContextBundle, receipt: Receipt) -> dict[str, object]:
    """The bundle and receipt as a JSON-serializable mapping.

    The single source of truth for the structured query result shared across
    surfaces: the CLI serializes it to JSON and the MCP query tool returns it
    directly, so both emit the identical bundle render, token totals, per-unit
    representations, and receipt the library produced.
    """
    return {
        "bundle": {
            "context": bundle.render(),
            "total_tokens": bundle.total_tokens,
            "representations": [
                {"unit_id": rep.unit_id, "mode": rep.mode, "token_count": rep.token_count}
                for rep in bundle.representations
            ],
        },
        "receipt": receipt_dict(receipt),
    }
