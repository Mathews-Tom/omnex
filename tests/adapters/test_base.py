"""Tests for the modality adapter contract."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pytest

from omnex.adapters.base import AdapterCapabilities, ModalityAdapter
from omnex.ir.types import Document, Reference, Span, Unit


def test_adapter_capabilities_construction() -> None:
    caps = AdapterCapabilities(
        unit_kinds=frozenset({"PARAGRAPH", "SECTION"}),
        reference_kinds=frozenset({"CONTAINS"}),
        deterministic_parse=True,
        model_extraction_opt_in=False,
    )
    assert "SECTION" in caps.unit_kinds
    assert caps.reference_kinds == frozenset({"CONTAINS"})
    assert caps.deterministic_parse is True
    assert caps.model_extraction_opt_in is False


def test_adapter_capabilities_is_frozen() -> None:
    caps = AdapterCapabilities(
        unit_kinds=frozenset(),
        reference_kinds=frozenset(),
        deterministic_parse=True,
        model_extraction_opt_in=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.deterministic_parse = False  # type: ignore[misc]


class _FakeAdapter:
    """A trivial in-test adapter that structurally satisfies ModalityAdapter."""

    def claims(self, source: Path) -> bool:
        return source.suffix == ".txt"

    def ingest(self, source: Path) -> Document:
        return Document(
            id="doc:fake",
            uri=str(source),
            modality="prose",
            content_hash="sha256:00",
            raw_token_count=0,
        )

    def parse(self, document: Document) -> list[Unit]:
        return [
            Unit(
                id="unit:fake",
                document_id=document.id,
                span=Span(0, 0),
                text="",
                token_count=0,
                title=None,
                breadcrumb=(),
                kind="PARAGRAPH",
                summary=None,
                protect=False,
            )
        ]

    def link(self, document: Document, units: Sequence[Unit]) -> list[Reference]:
        return []

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            unit_kinds=frozenset({"PARAGRAPH"}),
            reference_kinds=frozenset(),
            deterministic_parse=True,
            model_extraction_opt_in=False,
        )


def test_fake_adapter_satisfies_protocol() -> None:
    # Static structural conformance (checked by mypy) and runtime structural
    # conformance (runtime_checkable Protocol).
    adapter: ModalityAdapter = _FakeAdapter()
    assert isinstance(adapter, ModalityAdapter)


def test_fake_adapter_round_trips_ir() -> None:
    adapter: ModalityAdapter = _FakeAdapter()
    assert adapter.claims(Path("notes.txt")) is True
    assert adapter.claims(Path("scan.png")) is False
    document = adapter.ingest(Path("notes.txt"))
    units = adapter.parse(document)
    assert units[0].document_id == document.id
    assert adapter.link(document, units) == []


def test_fake_adapter_reports_capabilities() -> None:
    caps = _FakeAdapter().capabilities()
    assert caps.unit_kinds == frozenset({"PARAGRAPH"})
    assert caps.reference_kinds == frozenset()
    assert caps.deterministic_parse is True
    assert caps.model_extraction_opt_in is False
