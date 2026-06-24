"""Benchmark baselines: chunk splitter, embedders, the pinned chunk-and-embed
retrieval, recall tuning, and the product-import-isolation contract.
"""

from __future__ import annotations

import pytest

from omnex.bench.baselines import (
    FastEmbedEmbedder,
    TfidfEmbedder,
    _encoder,
    chunk_and_embed_baseline,
    chunk_text,
)


def _token_count(text: str) -> int:
    return len(_encoder().encode(text))


def test_chunk_text_keeps_short_text_in_one_chunk() -> None:
    assert chunk_text("hello world", 256, 32) == ["hello world"]


def test_chunk_text_splits_long_text_into_multiple_windows() -> None:
    text = " ".join(str(i) for i in range(300))
    chunks = chunk_text(text, 50, 10)
    assert len(chunks) > 1
    assert all(chunks)


def test_chunk_overlap_duplicates_boundary_tokens() -> None:
    text = " ".join(str(i) for i in range(300))
    total = _token_count(text)
    overlapped = sum(_token_count(chunk) for chunk in chunk_text(text, 50, 10))
    disjoint = sum(_token_count(chunk) for chunk in chunk_text(text, 50, 0))
    # No overlap covers each token once; overlap re-emits the shared boundary.
    assert disjoint == total
    assert overlapped > total


def test_chunk_text_is_deterministic_on_repeat() -> None:
    text = " ".join(str(i) for i in range(120))
    assert chunk_text(text, 40, 8) == chunk_text(text, 40, 8)


def test_chunk_text_rejects_bad_sizes() -> None:
    for tokens, overlap in ((0, 0), (-1, 0)):
        try:
            chunk_text("x", tokens, overlap)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for chunk_tokens={tokens}")
    for overlap in (-1, 50, 60):
        try:
            chunk_text("x", 50, overlap)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for overlap={overlap}")


def test_chunk_text_handles_empty_string() -> None:
    assert chunk_text("", 256, 32) == []


def test_tfidf_embedder_is_deterministic_and_shares_one_space() -> None:
    embedder = TfidfEmbedder()
    first = embedder.embed(["payment charge", "weather forecast"])
    second = embedder.embed(["payment charge", "weather forecast"])
    assert first == second
    assert len(first) == 2
    assert len(first[0]) == len(first[1])  # dense vectors over one shared vocabulary


def test_chunk_and_embed_ranks_the_query_relevant_chunk_first() -> None:
    documents = ["payment charge record", "weather forecast sunny", "monetary currency amount"]
    ranked = chunk_and_embed_baseline(
        documents,
        "payment charge",
        TfidfEmbedder(),
        chunk_tokens=256,
        chunk_overlap=32,
    )
    assert ranked[0] == "payment charge record"
    assert set(ranked) == set(documents)  # every chunk returned, just reordered


def test_chunk_and_embed_is_deterministic_on_repeat() -> None:
    documents = [" ".join(str(i) for i in range(80)), "payment charge record"]
    first = chunk_and_embed_baseline(
        documents, "payment", TfidfEmbedder(), chunk_tokens=40, chunk_overlap=8
    )
    second = chunk_and_embed_baseline(
        documents, "payment", TfidfEmbedder(), chunk_tokens=40, chunk_overlap=8
    )
    assert first == second


def test_chunk_and_embed_handles_empty_corpus() -> None:
    ranked = chunk_and_embed_baseline([], "q", TfidfEmbedder(), chunk_tokens=256, chunk_overlap=32)
    assert ranked == []


def test_fastembed_embedder_embeds_with_the_pinned_model_when_installed() -> None:
    pytest.importorskip("fastembed")
    embedder = FastEmbedEmbedder()
    assert embedder.name == "BAAI/bge-small-en-v1.5"
    vectors = embedder.embed(["create a payment", "a monetary value"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0
