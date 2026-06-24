"""Tests for the RetrievalKernel: T0 end-to-end, determinism, gating, blindness."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnex.ir.types import Reference, Span, Unit, UnitKind
from omnex.kernel.config import KernelConfig
from omnex.kernel.kernel import RetrievalKernel
from omnex.kernel.packer import count_tokens


def _unit(
    uid: str,
    text: str,
    *,
    title: str | None = None,
    breadcrumb: tuple[str, ...] = (),
    kind: UnitKind = "SECTION",
    protect: bool = False,
) -> Unit:
    return Unit(
        id=uid,
        document_id="doc:1",
        span=Span(0, max(len(text), 1)),
        text=text,
        token_count=len(text.split()),
        title=title,
        breadcrumb=breadcrumb,
        kind=kind,
        summary=None,
        protect=protect,
    )


def _corpus() -> tuple[list[Unit], list[Reference]]:
    units = [
        _unit(
            "u_op",
            "POST payments creates a payment from a request.",
            title="POST /payments",
            breadcrumb=("API", "POST /payments"),
            kind="OPERATION",
        ),
        _unit(
            "u_req",
            "PaymentRequest carries the amount and the customer.",
            title="PaymentRequest",
            breadcrumb=("API", "PaymentRequest"),
            kind="SCHEMA",
        ),
        _unit(
            "u_money",
            "Money has a currency and a value.",
            title="Money",
            breadcrumb=("API", "Money"),
            kind="SCHEMA",
            protect=True,
        ),
        _unit(
            "u_other",
            "Unrelated page about logging dashboards and metrics.",
            title="Observability",
            breadcrumb=("Ops",),
            kind="SECTION",
        ),
    ]
    references = [
        Reference("u_op", "u_req", "REFERENCES", 1.0, ()),
        Reference("u_req", "u_money", "REFERENCES", 1.0, ()),
    ]
    return units, references


def _config(tier: str = "T0", **overrides: object) -> KernelConfig:
    base: dict[str, object] = {
        "tier": tier,
        "bm25_profile": {"text": 1.0, "title": 3.0, "breadcrumb": 1.0, "summary": 1.0},
        "hop_budget_by_kind": {"REFERENCES": 2},
        "confidence_decay": 0.8,
        "enable_vector_lane": False,
        "enable_rerank": False,
    }
    base.update(overrides)
    return KernelConfig(**base)  # type: ignore[arg-type]


def _indexed() -> RetrievalKernel:
    units, references = _corpus()
    kernel = RetrievalKernel()
    kernel.index(units, references)
    return kernel


def test_t0_retrieve_returns_relevant_units() -> None:
    kernel = _indexed()
    bundle, _ = kernel.retrieve("payment request", 100, _config())
    included = {rep.unit_id for rep in bundle.representations if rep.mode != "SKIP"}
    # Lexical matches plus their reference closure within budget; the unrelated
    # page is neither matched nor reachable.
    assert {"u_op", "u_req"} <= included
    assert "u_other" not in included


def test_t0_bundle_and_receipt_are_byte_identical_on_repeat() -> None:
    kernel = _indexed()
    config = _config()
    first_bundle, first_receipt = kernel.retrieve("payment request", 100, config)
    second_bundle, second_receipt = kernel.retrieve("payment request", 100, config)
    assert first_bundle.render() == second_bundle.render()
    assert first_receipt == second_receipt
    assert repr(first_receipt) == repr(second_receipt)


def test_receipt_records_byte_exact_and_zero_model_use() -> None:
    kernel = _indexed()
    _, receipt = kernel.retrieve("payment request", 100, _config())
    assert receipt.determinism_class == "byte_exact"
    assert receipt.model_used is False
    assert receipt.model_version is None
    assert receipt.extraction_used is False
    assert receipt.tiers_run == ("T0",)
    # baseline is the full-dump upper bound in the same count_tokens ledger the
    # packer budgets against.
    assert receipt.baseline_tokens == sum(count_tokens(u.text) for u in _corpus()[0])


def test_budget_is_respected() -> None:
    kernel = _indexed()
    budget = 8
    bundle, receipt = kernel.retrieve("payment request", budget, _config())
    assert bundle.total_tokens <= budget
    assert receipt.returned_tokens <= budget
    assert receipt.returned_tokens == bundle.total_tokens


def test_protected_unit_is_never_compressed_or_elided_end_to_end() -> None:
    kernel = _indexed()
    # A budget large enough to reach the protected Money schema but too small to
    # INCLUDE it whole would force COMPRESS/ELIDE on an unprotected unit; the
    # protected unit must only ever INCLUDE or SKIP.
    for budget in range(0, 40):
        bundle, _ = kernel.retrieve("payment request money", budget, _config())
        money = next((rep for rep in bundle.representations if rep.unit_id == "u_money"), None)
        if money is not None:
            assert money.mode in ("INCLUDE", "SKIP")


def test_t2_vector_request_raises() -> None:
    kernel = _indexed()
    with pytest.raises(NotImplementedError, match="T2 vector lane"):
        kernel.retrieve("payment", 100, _config(tier="T2", enable_vector_lane=True))


def test_enable_vector_lane_raises_even_at_t0() -> None:
    kernel = _indexed()
    with pytest.raises(NotImplementedError, match="vector lane"):
        kernel.retrieve("payment", 100, _config(enable_vector_lane=True))


def test_t3_extraction_request_raises() -> None:
    kernel = _indexed()
    with pytest.raises(NotImplementedError, match="T3 model extraction"):
        kernel.retrieve("payment", 100, _config(tier="T3"))


def test_enable_rerank_request_raises() -> None:
    kernel = _indexed()
    with pytest.raises(NotImplementedError, match="rerank lane"):
        kernel.retrieve("payment", 100, _config(enable_rerank=True))


def test_retrieve_before_index_raises() -> None:
    with pytest.raises(RuntimeError, match="index"):
        RetrievalKernel().retrieve("payment", 100, _config())


def test_kernel_package_does_not_import_adapters() -> None:
    # Modality-blindness: no kernel module may depend on any adapter.
    kernel_dir = Path(__file__).resolve().parents[2] / "src" / "omnex" / "kernel"
    offenders = [
        path.name
        for path in kernel_dir.glob("*.py")
        if "omnex.adapters" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
