"""omnex: universal, structure-aware retrieval at a fraction of the tokens.

The top-level package exposes the public library API (``index`` and ``query``),
the result types (``ContextBundle`` and ``Receipt``), the kernel configuration
(``KernelConfig`` plus the ``Tier`` and ``DeterminismClass`` literals), and the
core IR types every modality adapter emits (``Document``, ``Span``, ``Unit``,
``Reference``).

Importing ``omnex`` stays cheap and side-effect free: no model load, no network
access, and no file-system read on this import path. The FTS index opens its
SQLite connection only when a kernel is instantiated, never at import time.
"""

from __future__ import annotations

from omnex.api import index, index_sources, query, query_sources
from omnex.ir.types import Document, Reference, Span, Unit
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.config import DeterminismClass, KernelConfig, Tier
from omnex.kernel.receipt import Receipt

__version__ = "0.1.0"

__all__ = [
    "ContextBundle",
    "DeterminismClass",
    "Document",
    "KernelConfig",
    "Receipt",
    "Reference",
    "Span",
    "Tier",
    "Unit",
    "__version__",
    "index",
    "index_sources",
    "query",
    "query_sources",
]
