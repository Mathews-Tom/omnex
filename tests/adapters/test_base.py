"""Tests for the modality adapter contract."""

from __future__ import annotations

import dataclasses

import pytest

from omnex.adapters.base import AdapterCapabilities


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
