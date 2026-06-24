"""Tests for the Receipt value type: fields, immutability, equality."""

from __future__ import annotations

import dataclasses

import pytest

from omnex.kernel.receipt import Receipt


def _receipt() -> Receipt:
    return Receipt(
        returned_tokens=26,
        baseline_tokens=120,
        tiers_run=("T0",),
        model_used=False,
        model_version=None,
        extraction_used=False,
        determinism_class="byte_exact",
        reference_closure_complete=False,
        recall_basis="lexical",
    )


def test_receipt_records_its_fields() -> None:
    receipt = _receipt()
    assert receipt.returned_tokens == 26
    assert receipt.baseline_tokens == 120
    assert receipt.tiers_run == ("T0",)
    assert receipt.model_used is False
    assert receipt.model_version is None
    assert receipt.extraction_used is False
    assert receipt.determinism_class == "byte_exact"
    assert receipt.reference_closure_complete is False
    assert receipt.recall_basis == "lexical"


def test_receipt_is_frozen() -> None:
    receipt = _receipt()
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.returned_tokens = 0  # type: ignore[misc]


def test_equal_receipts_are_byte_identical() -> None:
    assert _receipt() == _receipt()
    assert repr(_receipt()) == repr(_receipt())


def test_distinct_receipts_differ() -> None:
    other = dataclasses.replace(_receipt(), determinism_class="pinned_reproducible")
    assert other != _receipt()


def test_lexical_recall_limitations_are_surfaced() -> None:
    limitations = _receipt().recall_limitations
    assert limitations, "a lexical-only run must surface its recall limitation"
    joined = " ".join(limitations).lower()
    assert "lexical" in joined
    assert "cross_ref" in joined
    assert "embedding" in joined


def test_vector_recall_has_no_lexical_limitation() -> None:
    receipt = dataclasses.replace(_receipt(), recall_basis="lexical_plus_vector")
    assert receipt.recall_limitations == ()
