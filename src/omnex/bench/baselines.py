"""Benchmark baselines: the demoted full-dump upper bound and the chunk-and-embed
headline.

A baseline turns a corpus and a query into a *ranked list of passage texts*, the
shape the runner grades uniformly (count its tokens, check which gold labels it
covers) regardless of how the passages were produced. This module defines two:
the full-document dump (the naive paste-everything upper bound) and the realistic
chunk-and-embed retrieval (the headline omnex must beat).

The full dump is the *upper bound*, not the headline. It reaches full recall
trivially -- it returns the entire corpus -- so it only bounds how wasteful token
spend can be; it never demonstrates a competitive win. The chunk-and-embed
baseline is the headline: fixed-size token-window chunks ranked by embedding
similarity, the standard strong RAG retrieval. The two-number honesty framing in
:mod:`omnex.bench.report` keeps the upper bound demoted and the chunk-and-embed
number as the headline.

Chunking is denominated in ``tiktoken`` ``cl100k_base`` tokens (the realistic
"256-token chunk"), a deterministic offline encoder loaded once and reused. The
token *comparison* ledger is the product's whitespace ``count_tokens``, so omnex
and the baselines are measured identically; the runner applies it.

Benchmark-only. Nothing under ``omnex.kernel`` or ``omnex.adapters`` imports this
package; the embedding lane here is a benchmark dependency, never omnex's
retrieval path.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence, Set
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from omnex.bench.metrics import covered_labels, recall

if TYPE_CHECKING:
    import tiktoken

# Deterministic offline encoder defining a chunk's token boundaries. This is the
# chunk-size ledger only; token comparisons use the product's whitespace measure.
_CHUNK_ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    """Load the ``cl100k_base`` encoder once, deferred to first use."""
    import tiktoken

    return tiktoken.get_encoding(_CHUNK_ENCODING)


def chunk_text(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping fixed-size token windows.

    Windows are ``chunk_tokens`` ``cl100k_base`` tokens wide and advance by
    ``chunk_tokens - overlap`` tokens, so consecutive chunks share ``overlap``
    tokens. The overlap is what keeps a gold marker that straddles a window
    boundary present whole in at least one chunk, so coverage is not lost to an
    arbitrary split. Deterministic and offline. An empty string yields no chunks.
    """
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if not 0 <= overlap < chunk_tokens:
        raise ValueError(f"overlap must be in [0, chunk_tokens), got {overlap}")
    encoder = _encoder()
    tokens = encoder.encode(text)
    step = chunk_tokens - overlap
    return [encoder.decode(tokens[i : i + chunk_tokens]) for i in range(0, len(tokens), step)]


def full_dump_baseline(documents: Sequence[str]) -> list[str]:
    """Return the whole corpus as a single ranked passage: paste everything.

    Joining every document into one passage models the naive prompt that dumps
    the full corpus into the context window. Graded against any gold set it
    reaches recall ``1.0`` (the corpus contains everything), which is exactly why
    it is the upper bound and not a competitor: it bounds token waste, it does not
    win. Returned as a one-element list so the runner grades it through the same
    path as a chunked retrieval.
    """
    return ["\n".join(documents)]


# Pinned strong embedding model for the headline baseline: bge-small via fastembed.
# Named here, recorded in the artifact, and used as FastEmbedEmbedder's default.
_PINNED_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_WORD = re.compile(r"[A-Za-z]+")


class Embedder(Protocol):
    """Maps texts to dense vectors comparable by cosine similarity.

    ``name`` records the embedder identity in the artifact. ``embed`` must return
    one vector per input text, all in the same vector space, so the query and the
    chunks passed in one call are directly comparable.
    """

    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _terms(text: str) -> list[str]:
    return [match.lower() for match in _WORD.findall(text)]


class TfidfEmbedder:
    """Deterministic, offline TF-IDF cosine embedder.

    Fits its vocabulary and inverse document frequencies on the batch passed to
    :meth:`embed` and returns dense TF-IDF vectors over that vocabulary, so the
    query and chunks embedded together share one space. This is a genuine
    lexical retrieval baseline -- not a stand-in for the embedding model -- used
    where byte-exact, network-free reproducibility is required (CI and the
    deterministic artifact). It needs no model download and no extra dependency.
    """

    name = "tfidf-cosine"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        tokenized = [_terms(text) for text in texts]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        total = len(tokenized)
        vocabulary = sorted(document_frequency)
        position = {term: index for index, term in enumerate(vocabulary)}
        idf = {
            term: math.log((1.0 + total) / (1.0 + document_frequency[term])) + 1.0
            for term in vocabulary
        }
        vectors: list[list[float]] = []
        for tokens in tokenized:
            term_frequency = Counter(tokens)
            length = max(1, len(tokens))
            vector = [0.0] * len(vocabulary)
            for term, count in term_frequency.items():
                vector[position[term]] = (count / length) * idf[term]
            vectors.append(vector)
        return vectors


class FastEmbedEmbedder:
    """The pinned strong embedding lane: ``fastembed`` over a named model.

    Defaults to ``BAAI/bge-small-en-v1.5``, the pinned headline model. The model
    is loaded lazily on first :meth:`embed`, so importing this module never
    requires ``fastembed`` -- the dependency lives behind the ``bench``/``embed``
    extra and is a benchmark-only cost, never omnex's retrieval path. Its outputs
    are reproducible only with the pinned model, tokenizer, runtime, and
    architecture (a weaker determinism class than the byte-exact tiers), which the
    artifact records.
    """

    def __init__(self, model_name: str = _PINNED_EMBED_MODEL) -> None:
        self.name = model_name
        self._model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        return [[float(value) for value in vector] for vector in model.embed(list(texts))]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def chunk_and_embed_baseline(
    documents: Sequence[str],
    query: str,
    embedder: Embedder,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
) -> list[str]:
    """Rank fixed-size chunks of the corpus by embedding similarity to ``query``.

    Chunks every document into ``chunk_tokens``-wide token windows, embeds the
    query and all chunks together (one shared vector space), and returns the chunk
    texts ordered by descending cosine similarity, ties broken by original chunk
    position so the ranking is total and deterministic for a fixed embedder. This
    is the realistic strong RAG retrieval the headline comparison runs against.
    """
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_text(document, chunk_tokens, chunk_overlap)
    ]
    if not chunks:
        return []
    vectors = embedder.embed([query, *chunks])
    query_vector, chunk_vectors = vectors[0], vectors[1:]
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-_cosine(query_vector, chunk_vectors[index]), index),
    )
    return [chunks[index] for index in order]


@dataclass(frozen=True, slots=True)
class ChunkEmbedConfig:
    """The pinned chunk-and-embed configuration, recorded verbatim in the artifact.

    ``chunk_tokens``/``chunk_overlap`` size the token windows; ``embedder`` names
    the pinned model; ``rerank`` names an optional cross-encoder reranker and is
    ``None`` in v0 (the headline does not depend on reranking, so it stays off and
    out of the determinism surface). Pinning these and recording them is what
    makes the headline number defensible rather than self-graded.
    """

    chunk_tokens: int = 256
    chunk_overlap: int = 32
    embedder: str = _PINNED_EMBED_MODEL
    rerank: str | None = None


# The single pinned strong configuration the headline artifact is generated with.
PINNED_CHUNK_EMBED = ChunkEmbedConfig()


def chunks_for_recall(
    ranked: Sequence[str],
    labels: Set[str],
    target_recall: float,
) -> list[str]:
    """Shortest ranked prefix whose cumulative text reaches ``target_recall``.

    Walks ``ranked`` accumulating text and the gold ``labels`` covered so far, and
    returns the shortest prefix whose recall reaches ``target_recall``; returns the
    whole ranking when the target is never reached (the caller checks the achieved
    prefix, so the baseline's recall is tunable to a target. It exists to
    demonstrate that tunability (it is exercised by the baseline tests); the runner
    grades on the token axis via ``tokens_at_recall`` and does not call this.
    """
    if not 0.0 <= target_recall <= 1.0:
        raise ValueError(f"target_recall must be in [0.0, 1.0], got {target_recall}")
    if recall(frozenset(), labels) >= target_recall:
        return []
    cumulative = ""
    prefix: list[str] = []
    for text in ranked:
        prefix.append(text)
        cumulative += "\n" + text
        if recall(covered_labels(cumulative, labels), labels) >= target_recall:
            return prefix
    return prefix
