"""Public library API: index a corpus and query it under a token budget.

These are the thin entry points most callers use. ``index`` builds a reusable
indexed kernel; ``query`` is the one-shot convenience that indexes a corpus and
answers a single question, returning the rendered ``ContextBundle`` and its
auditable ``Receipt``.

Both operate purely on the IR (``Unit`` and ``Reference``), so the public surface
stays modality-blind. No model load, network, or file-system access.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnex.ir.types import Reference, Unit
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.config import KernelConfig
from omnex.kernel.kernel import RetrievalKernel
from omnex.kernel.receipt import Receipt


def index(corpus: Sequence[Unit], references: Sequence[Reference] = ()) -> RetrievalKernel:
    """Build a reusable kernel indexed over ``corpus`` and its ``references``."""
    kernel = RetrievalKernel()
    kernel.index(corpus, references)
    return kernel


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
