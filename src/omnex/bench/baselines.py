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

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

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
