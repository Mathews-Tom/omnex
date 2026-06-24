"""The retrieval kernel: wire index, fuse, expand, and pack into a bundle.

``RetrievalKernel`` is modality-blind and LLM-free. It indexes a corpus of IR
``Unit`` values and their ``Reference`` edges once, then answers queries by
running the same pipeline regardless of modality: FTS retrieval, rank fusion,
bounded graph expansion, and budget-aware packing, emitting a ``ContextBundle``
and an auditable ``Receipt``.

Only the T0 floor is wired here. T1 reference closure, the T2 vector lane, T3
model extraction, and the rerank lane are gated off and fail loud, so a run never
silently claims a guarantee it cannot keep. The T0 path is byte-exact: the same
corpus, config, and query produce a byte-identical bundle and receipt.

No model load, network, or file-system access.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnex.ir.graph import Hop, StructureGraph, build_graph
from omnex.ir.types import Reference, Unit
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.config import DeterminismClass, KernelConfig, Tier
from omnex.kernel.expand import graph_expand
from omnex.kernel.fusion import combine
from omnex.kernel.index import FtsIndex
from omnex.kernel.packer import Candidate, count_tokens, pack_efficiently, score_candidate
from omnex.kernel.receipt import Receipt

__all__ = [
    "DeterminismClass",
    "KernelConfig",
    "RetrievalKernel",
    "Tier",
]

# The only tier this stack wires, and the determinism class it may claim. T1
# (reference closure) lands in a later stack and is gated off until then.
_DETERMINISM_BY_TIER: dict[str, DeterminismClass] = {"T0": "byte_exact"}

# Upper bound on lexical candidates pulled before fusion and expansion.
_FTS_CANDIDATE_LIMIT = 200


class RetrievalKernel:
    """Indexes an IR corpus once and answers budgeted queries against it."""

    __slots__ = ("_graph", "_index", "_units")

    def __init__(self) -> None:
        self._index = FtsIndex()
        self._graph: StructureGraph | None = None
        self._units: dict[str, Unit] = {}

    def index(self, corpus: Sequence[Unit], references: Sequence[Reference] = ()) -> None:
        """Index a corpus of units and build the StructureGraph from its edges."""
        self._index.index_units(corpus)
        self._graph = build_graph(corpus, references)
        self._units = {unit.id: unit for unit in corpus}

    def retrieve(
        self, query: str, budget_tokens: int, config: KernelConfig
    ) -> tuple[ContextBundle, Receipt]:
        """Retrieve a budget-packed bundle and its receipt for ``query``.

        Runs the T0 pipeline: lexical FTS retrieval, single-lane fusion, bounded
        graph expansion, relevance-per-token scoring, and budget packing. T1
        closure, the T2 vector lane, T3 extraction, and the rerank lane are gated
        off and fail loud.
        """
        self._reject_unsupported(config)
        if self._graph is None:
            raise RuntimeError("index() must be called before retrieve()")

        lexical = self._index.search(query, config.bm25_profile, _FTS_CANDIDATE_LIMIT)
        fused = combine([[unit_id for unit_id, _ in lexical]])
        hops = graph_expand(fused, self._graph, config.hop_budget_by_kind, config.confidence_decay)

        signals = self._relevance_signals(lexical, hops)
        candidates = [
            Candidate(
                self._units[hop.unit_id],
                score_candidate(self._units[hop.unit_id], signals, hop.depth),
                hop.depth,
            )
            for hop in hops
        ]
        representations = pack_efficiently(candidates, budget_tokens, config)

        bundle = ContextBundle(tuple(representations), self._units)
        receipt = Receipt(
            returned_tokens=bundle.total_tokens,
            baseline_tokens=sum(count_tokens(unit.text) for unit in self._units.values()),
            tiers_run=(config.tier,),
            model_used=False,
            model_version=None,
            extraction_used=False,
            determinism_class=_DETERMINISM_BY_TIER[config.tier],
        )
        return bundle, receipt

    @staticmethod
    def _relevance_signals(
        lexical: Sequence[tuple[str, float]],
        hops: Sequence[Hop],
    ) -> dict[str, float]:
        """Build per-unit relevance signals for scoring.

        Lexical matches use their BM25F score normalized to ``(0, 1]`` (so a
        seed's lexical strength is comparable to a neighbor's expansion weight);
        expanded neighbors that did not match lexically fall back to their decayed
        expansion confidence.
        """
        max_score = max((score for _, score in lexical), default=0.0)
        seed_relevance = {
            unit_id: (score / max_score if max_score > 0.0 else 1.0) for unit_id, score in lexical
        }
        signals: dict[str, float] = {}
        for hop in hops:
            signals[hop.unit_id] = seed_relevance.get(hop.unit_id, hop.confidence)
        return signals

    @staticmethod
    def _reject_unsupported(config: KernelConfig) -> None:
        if config.enable_vector_lane or config.tier == "T2":
            raise NotImplementedError(
                "the T2 vector lane is not implemented in this kernel; "
                "set enable_vector_lane=False and tier='T0'"
            )
        if config.tier == "T1":
            raise NotImplementedError(
                "T1 reference closure is not implemented in this kernel; it lands in a "
                "later stack. Use tier='T0' for the byte-exact floor."
            )
        if config.tier == "T3":
            raise NotImplementedError("T3 model extraction is not implemented in this kernel")
        if config.enable_rerank:
            raise NotImplementedError(
                "the rerank lane is not implemented in this kernel; set enable_rerank=False"
            )
