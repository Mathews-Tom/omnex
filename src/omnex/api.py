"""Public library API: index a corpus and query it under a token budget.

These are the thin entry points most callers use. ``index`` and ``query`` operate
purely on the IR (``Unit`` and ``Reference``), are modality-blind, and do no
model, network, or file-system access. ``index_sources`` and ``query_sources``
are the source-level entry points: they route each source through the adapter
that claims it (the only file-system reads happen there) and then run the same IR
pipeline, so the kernel stays modality-blind while callers can hand omnex raw
spec files. ``query`` returns the rendered ``ContextBundle`` and its auditable
``Receipt``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

from omnex.adapters import select_adapter
from omnex.ir.types import Document, Reference, Unit, compute_content_hash, normalize_content
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.config import KernelConfig
from omnex.kernel.kernel import RetrievalKernel
from omnex.kernel.packer import count_tokens
from omnex.kernel.receipt import Receipt


def index(corpus: Sequence[Unit], references: Sequence[Reference] = ()) -> RetrievalKernel:
    """Build a reusable kernel indexed over ``corpus`` and its ``references``."""
    kernel = RetrievalKernel()
    kernel.index(corpus, references)
    return kernel


def _route_sources(
    sources: Sequence[Path],
) -> tuple[list[Unit], list[Reference], list[Document]]:
    """Route each source through its claiming adapter into shared IR.

    For each source the claiming adapter ingests, parses, and links, accumulating
    the units, reference edges, and ingested documents. Fails loud when no
    adapter claims a source.
    """
    units: list[Unit] = []
    references: list[Reference] = []
    documents: list[Document] = []
    for source in sources:
        adapter = select_adapter(source)
        document = adapter.ingest(source)
        documents.append(document)
        document_units = adapter.parse(document)
        units.extend(document_units)
        references.extend(adapter.link(document, document_units))
    return units, references, documents


def _full_dump_tokens(documents: Sequence[Document]) -> int:
    """Token count of the whole source(s), the naive paste-everything baseline.

    Counted in the same whitespace ``count_tokens`` ledger as the packed output.
    The re-read is hash-verified against each document's ``content_hash``, like
    the rest of the pipeline, so a source that changed since ingest fails loud
    rather than yielding a baseline that does not match the parsed units.
    """
    total = 0
    for document in documents:
        text = Path(document.uri).read_text(encoding="utf-8")
        if compute_content_hash(text) != document.content_hash:
            raise ValueError(f"source changed since ingest: {document.uri}")
        total += count_tokens(normalize_content(text))
    return total


def index_sources(sources: Sequence[Path]) -> RetrievalKernel:
    """Build a reusable kernel by routing ``sources`` through their adapters."""
    units, references, _ = _route_sources(sources)
    return index(units, references)


def query(
    corpus: Sequence[Unit],
    question: str,
    budget_tokens: int,
    config: KernelConfig,
    references: Sequence[Reference] = (),
) -> tuple[ContextBundle, Receipt]:
    """Index ``corpus`` and answer ``question`` under ``budget_tokens``.

    Returns the packed ``ContextBundle`` and its ``Receipt``. For repeated queries
    over the same corpus, call :func:`index` once and reuse the kernel instead.
    """
    kernel = index(corpus, references)
    return kernel.retrieve(question, budget_tokens, config)


def query_sources(
    sources: Sequence[Path],
    question: str,
    budget_tokens: int,
    config: KernelConfig,
) -> tuple[ContextBundle, Receipt]:
    """Route ``sources`` through their adapters and answer ``question``.

    Ingests, parses, and links each source into shared IR, then runs the same
    kernel pipeline. At tier T1 the bundle is the complete deterministic
    reference closure rendered as canonical spec fragments, packed under
    ``budget_tokens``. The receipt's ``baseline_tokens`` is set to the honest
    full-document dump (the naive paste-everything upper bound) so returned vs
    baseline compares against the real document rather than the kernel's
    sum-of-units (which double-counts schema text already inside its fields);
    ``reference_closure_complete`` is set by the kernel.
    """
    units, references, documents = _route_sources(sources)
    bundle, receipt = query(units, question, budget_tokens, config, references)
    full_dump = _full_dump_tokens(documents)
    return bundle, dataclasses.replace(receipt, baseline_tokens=full_dump)
