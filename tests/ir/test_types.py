"""Tests for the IR core types and content-address helpers."""

from __future__ import annotations

import dataclasses

import pytest

from omnex.ir.types import Document, Reference, Span, Unit


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


def _unit(**overrides: object) -> Unit:
    fields: dict[str, object] = {
        "id": "unit:1",
        "document_id": "doc:1",
        "span": Span(0, 5),
        "text": "hello",
        "token_count": 1,
        "title": None,
        "breadcrumb": (),
        "kind": "PARAGRAPH",
        "summary": None,
        "protect": False,
    }
    fields.update(overrides)
    return Unit(**fields)  # type: ignore[arg-type]


def test_unit_construction() -> None:
    unit = _unit(breadcrumb=("Guide", "Setup"), kind="SECTION", protect=True)
    assert unit.breadcrumb == ("Guide", "Setup")
    assert unit.kind == "SECTION"
    assert unit.protect is True


def test_unit_is_frozen() -> None:
    unit = _unit()
    with pytest.raises(dataclasses.FrozenInstanceError):
        unit.token_count = 9  # type: ignore[misc]


def test_unit_rejects_negative_token_count() -> None:
    with pytest.raises(ValueError, match="token_count"):
        _unit(token_count=-1)


def test_reference_construction() -> None:
    ref = Reference(
        source_id="unit:1",
        target_id="unit:2",
        kind="REFERENCES",
        confidence=0.5,
        evidence=("$ref",),
    )
    assert ref.kind == "REFERENCES"
    assert ref.evidence == ("$ref",)


def test_reference_is_frozen() -> None:
    ref = Reference("a", "b", "CONTAINS", 1.0, ())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.confidence = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_reference_rejects_out_of_range_confidence(bad: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        Reference("a", "b", "CONTAINS", bad, ())


@pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
def test_reference_accepts_boundary_confidence(ok: float) -> None:
    assert Reference("a", "b", "CONTAINS", ok, ()).confidence == ok
