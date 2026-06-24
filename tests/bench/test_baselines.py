"""Benchmark baselines: chunk splitter, embedders, the pinned chunk-and-embed
retrieval, recall tuning, and the product-import-isolation contract.
"""

from __future__ import annotations

from omnex.bench.baselines import _encoder, chunk_text


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
