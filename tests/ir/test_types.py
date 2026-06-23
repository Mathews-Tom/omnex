"""Tests for the IR core types and content-address helpers."""

from __future__ import annotations

import dataclasses

import pytest

from omnex.ir.types import Document, Span


def test_document_construction() -> None:
    doc = Document(
        id="doc:abc",
        uri="file:///a.md",
        modality="prose",
        content_hash="sha256:00",
        raw_token_count=42,
    )
    assert doc.modality == "prose"
    assert doc.raw_token_count == 42


def test_document_is_frozen() -> None:
    doc = Document(
        id="doc:abc",
        uri="file:///a.md",
        modality="prose",
        content_hash="sha256:00",
        raw_token_count=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.raw_token_count = 2  # type: ignore[misc]


def test_document_rejects_negative_token_count() -> None:
    with pytest.raises(ValueError, match="raw_token_count"):
        Document(
            id="doc:abc",
            uri="file:///a.md",
            modality="prose",
            content_hash="sha256:00",
            raw_token_count=-1,
        )


def test_span_construction_allows_empty_range() -> None:
    assert Span(5, 5).start == 5
    assert Span(0, 10).end == 10


def test_span_is_frozen() -> None:
    span = Span(0, 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.start = 2  # type: ignore[misc]


def test_span_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="must be <= end"):
        Span(10, 3)
