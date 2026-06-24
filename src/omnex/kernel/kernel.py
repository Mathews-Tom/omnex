"""The retrieval kernel: wire index, fuse, expand, and pack into a bundle.

``RetrievalKernel`` is modality-blind and LLM-free. It indexes a corpus of IR
``Unit`` values and their ``Reference`` edges once, then answers queries by
running the same pipeline regardless of modality: FTS retrieval, rank fusion,
bounded graph expansion, and budget-aware packing, emitting a ``ContextBundle``
and an auditable ``Receipt``.

The byte-exact tiers are wired here: T0 (bounded floor) and T1, which adds the
deterministic transitive reference closure. T2 adds the opt-in vector lane, fused
with the lexical lane; T3 model extraction and the rerank lane are gated off and
fail loud, so a run never silently claims a guarantee it cannot keep. T0 and T1
are byte-exact (same corpus, config, and query -> byte-identical bundle and
receipt) and load no model and touch no network or file system. T2 is the weaker
pinned-reproducible class: enabling it loads the pinned embedding model on first
use, which the receipt records.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnex.ir.graph import Hop, StructureGraph, build_graph
from omnex.ir.types import Reference, Unit
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.config import DeterminismClass, KernelConfig, RecallBasis, Tier
from omnex.kernel.expand import closure_expand, graph_expand
from omnex.kernel.fusion import combine
from omnex.kernel.index import FtsIndex
from omnex.kernel.packer import Candidate, count_tokens, pack_efficiently, score_candidate
from omnex.kernel.receipt import Receipt
from omnex.kernel.vector import MISSING_EMBED_EXTRA, VectorIndex, vector_lane_available

__all__ = [
    "DeterminismClass",
    "KernelConfig",
    "RetrievalKernel",
    "Tier",
]

# Tiers this kernel wires and the determinism class each may claim. T0/T1 are the
# byte-exact, model-free floor (T1 adds the deterministic reference closure); T2
# adds the vector lane and is only pinned-reproducible -- a strictly weaker class --
# so byte_exact never covers a vector-assisted run.
_DETERMINISM_BY_TIER: dict[str, DeterminismClass] = {
    "T0": "byte_exact",
    "T1": "byte_exact",
    "T2": "pinned_reproducible",
}

# Hard reference edge kinds the T1 closure follows transitively to a fixpoint.
_T1_REF_KINDS: tuple[str, ...] = ("REFERENCES", "FOREIGN_KEY", "IMPORTS", "CALLS")

# Upper bound on lexical candidates pulled before fusion and expansion.
_FTS_CANDIDATE_LIMIT = 200

# Upper bound on vector candidates pulled before fusion, matching the FTS lane so
# neither lane dominates the fuse set purely by candidate count.
_VECTOR_CANDIDATE_LIMIT = 200


def _tiers_run(tier: Tier) -> tuple[str, ...]:
    """Tiers exercised by a run: the T0 floor always runs; T1 adds the closure."""
    return ("T0", "T1") if tier == "T1" else (tier,)


def _max_normalized(scored: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Scale lane scores to ``(0, 1]`` by the lane maximum (an empty lane -> {}).

    This is the byte-exact lexical normalization the kernel has always used: a
    seed's strength is its share of the strongest match, so it stays comparable to
    a neighbor's decayed expansion confidence.
    """
    max_score = max((score for _, score in scored), default=0.0)
    return {unit_id: (score / max_score if max_score > 0.0 else 1.0) for unit_id, score in scored}


def _minmax_normalized(scored: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Scale lane scores to ``[0, 1]`` by min-max spread (an empty lane -> {}).

    Used for the vector lane, whose cosine scores cluster in a narrow high band:
    dividing by the maximum alone barely separates a relevant unit from the noise
    floor, so the irrelevant tail would survive per-token packing. Spread
    normalization pushes that tail toward 0, matching the relative-score fusion the
    fusion module already uses. A lane whose scores are all equal maps to ``1.0``.
    """
    if not scored:
        return {}
    values = [score for _, score in scored]
    low, high = min(values), max(values)
    spread = high - low
    return {
        unit_id: (1.0 if spread == 0.0 else (score - low) / spread) for unit_id, score in scored
    }


class RetrievalKernel:
    """Indexes an IR corpus once and answers budgeted queries against it."""

    __slots__ = ("_graph", "_index", "_units", "_vector")

    def __init__(self) -> None:
        self._index = FtsIndex()
        self._graph: StructureGraph | None = None
        self._units: dict[str, Unit] = {}
        # The vector lane is built lazily on the first T2 retrieve and reused
        # across queries; it stays None on the core install and the T0/T1 paths.
        self._vector: VectorIndex | None = None

    def index(self, corpus: Sequence[Unit], references: Sequence[Reference] = ()) -> None:
        """Index a corpus of units and build the StructureGraph from its edges.

        Rebuilds every corpus-derived structure wholesale, including invalidating
        the lazily built vector lane, so re-indexing a reused kernel never leaves a
        stale embedding cache that would return wrong relevance or reference units
        absent from the rebuilt graph.
        """
        self._index.index_units(corpus)
        self._graph = build_graph(corpus, references)
        self._units = {unit.id: unit for unit in corpus}
        self._vector = None

    def retrieve(
        self, query: str, budget_tokens: int, config: KernelConfig
    ) -> tuple[ContextBundle, Receipt]:
        """Retrieve a budget-packed bundle and its receipt for ``query``.

        Runs the pipeline: lexical FTS retrieval, optional vector retrieval when
        the T2 lane is enabled, rank fusion across the active lanes, bounded graph
        expansion (plus the deterministic reference closure at T1), then
        relevance-per-token scoring and budget packing. The vector lane is fused
        through the same RRF as the lexical lane; T3 extraction and the rerank lane
        are gated off and fail loud.
        """
        self._reject_unsupported(config)
        if self._graph is None:
            raise RuntimeError("index() must be called before retrieve()")

        lexical = self._index.search(query, config.bm25_profile, _FTS_CANDIDATE_LIMIT)
        lanes: list[list[str]] = [[unit_id for unit_id, _ in lexical]]
        vector: list[tuple[str, float]] = []
        recall_basis: RecallBasis = "lexical"
        model_version: str | None = None
        if config.enable_vector_lane:
            lane = self._vector_lane()
            vector = lane.search(query, _VECTOR_CANDIDATE_LIMIT)
            lanes.append([unit_id for unit_id, _ in vector])
            recall_basis = "lexical_plus_vector"
            model_version = lane.model_name
        fused = combine(lanes)
        hops, closure_ids = self._expand(fused, config)

        signals = self._relevance_signals(lexical, vector, hops)
        candidates = [
            Candidate(
                self._units[hop.unit_id],
                score_candidate(self._units[hop.unit_id], signals, hop.depth),
                hop.depth,
            )
            for hop in hops
        ]
        representations = pack_efficiently(candidates, budget_tokens, config)

        included = frozenset(rep.unit_id for rep in representations if rep.mode == "INCLUDE")
        bundle = ContextBundle(tuple(representations), self._units)
        receipt = Receipt(
            returned_tokens=bundle.total_tokens,
            baseline_tokens=sum(count_tokens(unit.text) for unit in self._units.values()),
            tiers_run=_tiers_run(config.tier),
            model_used=config.enable_vector_lane,
            model_version=model_version,
            extraction_used=False,
            determinism_class=_DETERMINISM_BY_TIER[config.tier],
            reference_closure_complete=bool(closure_ids) and closure_ids <= included,
            recall_basis=recall_basis,
        )
        return bundle, receipt

    @staticmethod
    def _relevance_signals(
        lexical: Sequence[tuple[str, float]],
        vector: Sequence[tuple[str, float]],
        hops: Sequence[Hop],
    ) -> dict[str, float]:
        """Build per-unit relevance signals for scoring.

        Lexical matches use their BM25F score max-normalized to ``(0, 1]`` (so a
        seed's lexical strength is comparable to a neighbor's expansion weight).
        Vector matches add their cosine score min-max normalized to ``[0, 1]``,
        combined with the lexical signal by taking the stronger of the two, so a
        unit found only by the vector lane still carries a real relevance signal.
        Expanded neighbors that neither lane scored fall back to their decayed
        expansion confidence. With no vector lane the vector term is empty and the
        signals are byte-identical to the lexical-only floor.
        """
        seed_relevance = _max_normalized(lexical)
        for unit_id, relevance in _minmax_normalized(vector).items():
            seed_relevance[unit_id] = max(seed_relevance.get(unit_id, 0.0), relevance)
        signals: dict[str, float] = {}
        for hop in hops:
            signals[hop.unit_id] = seed_relevance.get(hop.unit_id, hop.confidence)
        return signals

    def _vector_lane(self) -> VectorIndex:
        """Return the lazily built vector lane, embedding the corpus on first use.

        Built once per kernel and reused across queries; never touched on the core
        install or the T0/T1 paths, so the optional ``fastembed`` dependency loads
        only when a caller opts into the lane.
        """
        if self._vector is None:
            self._vector = VectorIndex()
            self._vector.index_units(self._units.values())
        return self._vector

    def _expand(
        self, fused: Sequence[str], config: KernelConfig
    ) -> tuple[list[Hop], frozenset[str]]:
        """Return packing candidates and the T1 reference-closure id set.

        T0 pulls a bounded structural neighborhood. T1 also closure-expands the
        hard reference edges in full, so the candidate set is the union of the
        two (the nearest depth wins per unit) and the closure id set is what
        receipt completeness is checked against. T0 returns an empty closure set.
        """
        if self._graph is None:  # pragma: no cover - guarded by retrieve()
            raise RuntimeError("index() must be called before retrieve()")
        bounded = graph_expand(
            fused, self._graph, config.hop_budget_by_kind, config.confidence_decay
        )
        if config.tier != "T1":
            return bounded, frozenset()
        closure = closure_expand(fused, self._graph, _T1_REF_KINDS, config.confidence_decay)
        merged: dict[str, Hop] = {}
        for hop in (*bounded, *closure):
            current = merged.get(hop.unit_id)
            if current is None or hop.depth < current.depth:
                merged[hop.unit_id] = hop
        hops = [merged[unit_id] for unit_id in sorted(merged)]
        return hops, frozenset(hop.unit_id for hop in closure)

    @staticmethod
    def _reject_unsupported(config: KernelConfig) -> None:
        """Reject incoherent or unimplemented configs before any work runs.

        The T2 tier and the vector lane are two names for one capability, so they
        must agree: a T2 tier without the lane would silently run as the lexical
        floor, and the lane without the T2 tier would let the byte-exact label
        cover a vector-assisted run. T3 extraction and the rerank lane are not
        implemented and fail loud rather than degrade.
        """
        if config.tier == "T2" and not config.enable_vector_lane:
            raise ValueError(
                "tier 'T2' is the vector lane; set enable_vector_lane=True for a T2 run"
            )
        if config.enable_vector_lane and config.tier != "T2":
            raise ValueError(
                "the vector lane is the T2 tier; set tier='T2' to enable it so the "
                "receipt never claims byte_exact for a vector-assisted run"
            )
        if config.enable_vector_lane and not vector_lane_available():
            # Fail fast at the gate, before embedding the corpus, when the opt-in
            # dependency is absent rather than deep inside the first embed call.
            raise ModuleNotFoundError(MISSING_EMBED_EXTRA)
        if config.tier == "T3":
            raise NotImplementedError("T3 model extraction is not implemented in this kernel")
        if config.enable_rerank:
            raise NotImplementedError(
                "the rerank lane is not implemented in this kernel; set enable_rerank=False"
            )
