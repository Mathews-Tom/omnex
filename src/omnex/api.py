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
from omnex.ir.types import Reference, Unit, normalize_content
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


def _route_sources(sources: Sequence[Path]) -> tuple[list[Unit], list[Reference]]:
    """Route each source through its claiming adapter into shared IR.

    For each source the claiming adapter ingests, parses, and links, accumulating
    the units and reference edges. Failing loud when no adapter claims a source.
    """
    units: list[Unit] = []
    references: list[Reference] = []
    for source in sources:
        adapter = select_adapter(source)
        document = adapter.ingest(source)
        document_units = adapter.parse(document)
        units.extend(document_units)
        references.extend(adapter.link(document, document_units))
    return units, references


def index_sources(sources: Sequence[Path]) -> RetrievalKernel:
    """Build a reusable kernel by routing ``sources`` through their adapters."""
    units, references = _route_sources(sources)
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
    units, references = _route_sources(sources)
    bundle, receipt = query(units, question, budget_tokens, config, references)
    full_dump = sum(
        count_tokens(normalize_content(source.read_text(encoding="utf-8"))) for source in sources
    )
    return bundle, dataclasses.replace(receipt, baseline_tokens=full_dump)
