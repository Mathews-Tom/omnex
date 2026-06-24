"""Optional tier-T2 dense embedding lane over the IR.

This is omnex's *own* opt-in vector lane: a ``fastembed``-backed dense retriever
that runs alongside the FTS5/BM25F lane and is fused with it through the kernel's
existing rank fusion. It exists to recover semantically distant prose units that
share no vocabulary with the query -- the recall the byte-exact lexical floor
cannot reach at any budget.

It is strictly separate from the benchmark's chunk-and-embed baseline: nothing
here imports ``omnex.bench`` and the kernel never imports this module's optional
dependency unless the vector lane is enabled. The lane is reproducible only with
a pinned model, tokenizer, runtime, and architecture (a weaker determinism class
than T0/T1), so the receipt labels a vector-assisted run accordingly.

``fastembed`` lives behind the ``[embed]`` extra. Importing this module never
requires it: the model is loaded lazily on first use, so the core install still
imports omnex and runs the byte-exact T0/T1 paths unchanged. Asking for the lane
without the extra installed fails loud with an actionable message.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import platform
from collections.abc import Iterable, Sequence
from typing import Any

from omnex.ir.types import Unit
from omnex.kernel.config import EmbeddingProvenance

# The pinned default embedding model for the T2 lane. Named here -- not imported
# from the benchmark baseline -- so the product retrieval path stays independent
# of benchmark code. Its identity is recorded in the receipt's determinism class.
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Actionable hint raised whenever the lane is requested without the optional
# dependency installed, shared by the kernel's fail-fast gate and this module's
# lazy load so the message stays consistent.
MISSING_EMBED_EXTRA = (
    "the T2 vector lane requires the optional 'embed' extra; "
    "install it with `pip install omnex[embed]`"
)


def vector_lane_available() -> bool:
    """Whether the optional ``fastembed`` dependency can be imported.

    A pure capability probe: it inspects the import system without importing
    ``fastembed`` or loading any model, so it is safe to call on the core install.
    """
    return importlib.util.find_spec("fastembed") is not None


def embedding_cache_key(text: str, model_version: str) -> str:
    """Content-addressed cache key for one embedding: content hash + model version.

    Both components are load-bearing. The content hash lets identical text reuse a
    single embedding (across units and across repeated queries); the model version
    scopes the cache to one pinned model, since the same text embeds differently
    under a different model. The text is hashed, never stored, so the key is bounded.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_version}\n{content_hash}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two equal-length dense vectors; 0.0 if either is zero."""
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class VectorIndex:
    """A dense-embedding lane over IR unit text, backed by ``fastembed``.

    ``index_units`` embeds each unit's text once; ``search`` ranks the indexed
    units by cosine similarity to the query embedding. Ranking is total and
    order-stable -- ties on score break by ascending unit id -- so the same model
    and corpus always produce the same ordering. The cosine scores themselves are
    reproducible only with the pinned model, tokenizer, runtime, and architecture;
    that weaker determinism class is the caller's to record, not this lane's to
    claim.

    The model is loaded lazily on first embed, so constructing a ``VectorIndex``
    and importing this module never require ``fastembed``.
    """

    __slots__ = ("_cache", "_model", "_model_name", "_unit_ids", "_vectors")

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._unit_ids: list[str] = []
        self._vectors: list[list[float]] = []
        # Content-addressed embedding cache keyed by content hash + model version,
        # reused for both unit and query embeddings within this index's lifetime.
        self._cache: dict[str, list[float]] = {}

    @property
    def model_name(self) -> str:
        """The pinned embedding model identity backing this lane."""
        return self._model_name

    def provenance(self) -> EmbeddingProvenance:
        """Describe what this lane's embeddings depend on for replay.

        Records the pinned model and its bundled tokenizer (a fastembed model ships
        the tokenizer with the weights, so they share one pinned identity), the
        inference runtime (fastembed and its ONNX runtime), and the CPU
        architecture -- the inputs a pinned-reproducible T2 run reproduces only when
        matched. Imports the optional dependencies here because provenance is read
        only on the T2 path, where they are installed.
        """
        import fastembed
        import onnxruntime

        return EmbeddingProvenance(
            model=self._model_name,
            tokenizer=self._model_name,
            runtime=f"fastembed=={fastembed.__version__}; onnxruntime=={onnxruntime.__version__}",
            architecture=f"{platform.system()}/{platform.machine()}",
        )

    def _embedder(self) -> Any:
        """Load the pinned ``fastembed`` model once, failing loud when absent."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(MISSING_EMBED_EXTRA) from exc
            self._model = TextEmbedding(self._model_name)
        return self._model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` into dense float vectors with the pinned model."""
        model = self._embedder()
        return [[float(value) for value in vector] for vector in model.embed(list(texts))]

    def _embed_cached(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` through the content-addressed cache, in input order.

        Only texts not already cached are embedded, in a single batch, so identical
        text (a duplicate unit, or a repeated query) is embedded at most once and
        the model is loaded only when there is something new to embed.
        """
        keys = [embedding_cache_key(text, self._model_name) for text in texts]
        pending = {
            key: text for key, text in zip(keys, texts, strict=True) if key not in self._cache
        }
        if pending:
            for key, vector in zip(pending, self._embed(list(pending.values())), strict=True):
                self._cache[key] = vector
        return [self._cache[key] for key in keys]

    def index_units(self, units: Iterable[Unit]) -> None:
        """Embed and store ``units``, replacing any previously indexed content.

        Unit embeddings go through the content-addressed cache, so duplicate text
        across units is embedded once. A re-index fully replaces which units the
        lane ranks, so the lane never holds stale or duplicated rows.
        """
        materialized = list(units)
        self._unit_ids = [unit.id for unit in materialized]
        self._vectors = self._embed_cached([unit.text for unit in materialized])

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return ``(unit_id, cosine)`` pairs ranked best-first for ``query``.

        Scores are cosine similarity to the query embedding; the order sorts by
        descending score with ties broken by ascending unit id, so the result is
        total and stable. An empty index, an empty query, or a non-positive
        ``limit`` returns an empty list.
        """
        if not self._unit_ids or not query.strip() or limit <= 0:
            return []
        query_vector = self._embed_cached([query])[0]
        scored = [
            (unit_id, _cosine(query_vector, vector))
            for unit_id, vector in zip(self._unit_ids, self._vectors, strict=True)
        ]
        scored.sort(key=lambda row: (-row[1], row[0]))
        return scored[:limit]
