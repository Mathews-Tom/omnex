"""Tests for the public library API and top-level package exports."""

from __future__ import annotations

import omnex
from omnex import KernelConfig, index, query
from omnex.ir.types import Reference, Span, Unit, UnitKind
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.receipt import Receipt


def _unit(uid: str, text: str, *, title: str | None, kind: UnitKind = "SECTION") -> Unit:
    return Unit(
        id=uid,
        document_id="doc:1",
        span=Span(0, max(len(text), 1)),
        text=text,
        token_count=len(text.split()),
        title=title,
        breadcrumb=("Doc",),
        kind=kind,
        summary=None,
        protect=False,
    )


def _corpus() -> tuple[list[Unit], list[Reference]]:
    units = [
        _unit("u_a", "Alpha section about retrieval and indexing.", title="Alpha"),
        _unit("u_b", "Beta section about packing and budgets.", title="Beta"),
    ]
    references = [Reference("u_a", "u_b", "REFERENCES", 1.0, ())]
    return units, references


def _config() -> KernelConfig:
    return KernelConfig(
        tier="T0",
        bm25_profile={"text": 1.0, "title": 2.0},
        hop_budget_by_kind={"REFERENCES": 1},
        confidence_decay=0.8,
        enable_vector_lane=False,
        enable_rerank=False,
    )


def test_query_returns_bundle_and_receipt() -> None:
    units, references = _corpus()
    bundle, receipt = query(units, "retrieval indexing", 50, _config(), references=references)
    assert isinstance(bundle, ContextBundle)
    assert isinstance(receipt, Receipt)
    assert "retrieval" in bundle.render().lower()
    assert receipt.determinism_class == "byte_exact"


def test_query_is_byte_identical_on_repeat() -> None:
    units, references = _corpus()
    config = _config()
    bundle_a, receipt_a = query(units, "retrieval indexing", 50, config, references=references)
    bundle_b, receipt_b = query(units, "retrieval indexing", 50, config, references=references)
    assert bundle_a.render() == bundle_b.render()
    assert receipt_a == receipt_b


def test_index_returns_reusable_kernel_matching_one_shot_query() -> None:
    units, references = _corpus()
    config = _config()
    kernel = index(units, references)
    reused_bundle, reused_receipt = kernel.retrieve("retrieval indexing", 50, config)
    one_shot_bundle, one_shot_receipt = query(
        units, "retrieval indexing", 50, config, references=references
    )
    assert reused_bundle.render() == one_shot_bundle.render()
    assert reused_receipt == one_shot_receipt


def test_query_works_without_references() -> None:
    units, _ = _corpus()
    bundle, receipt = query(units, "packing budgets", 50, _config())
    assert isinstance(bundle, ContextBundle)
    assert receipt.tiers_run == ("T0",)


def test_top_level_exports_are_present() -> None:
    for name in (
        "index",
        "query",
        "ContextBundle",
        "Receipt",
        "KernelConfig",
        "Tier",
        "DeterminismClass",
        "EmbeddingProvenance",
        "Document",
        "Span",
        "Unit",
        "Reference",
        "__version__",
    ):
        assert hasattr(omnex, name), name
    assert set(omnex.__all__) >= {"index", "query", "ContextBundle", "Receipt", "Unit"}
