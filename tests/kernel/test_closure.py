"""Tests for the T1 transitive reference closure and its kernel wiring."""

from __future__ import annotations

import pytest

from omnex.ir.graph import StructureGraph, build_graph
from omnex.ir.types import Reference, ReferenceKind, Span, Unit, UnitKind
from omnex.kernel.config import KernelConfig
from omnex.kernel.expand import closure_expand
from omnex.kernel.kernel import RetrievalKernel

_REF_KINDS: tuple[ReferenceKind, ...] = ("REFERENCES", "FOREIGN_KEY", "IMPORTS", "CALLS")


def _unit(uid: str, *, kind: UnitKind = "SCHEMA", text: str | None = None) -> Unit:
    body = text if text is not None else uid
    return Unit(
        id=uid,
        document_id="doc:1",
        span=Span(0, max(len(body), 1)),
        text=body,
        token_count=len(body.split()),
        title=uid,
        breadcrumb=(),
        kind=kind,
        summary=None,
        protect=True,
    )


def _ref(source: str, target: str, kind: ReferenceKind = "REFERENCES") -> Reference:
    return Reference(source, target, kind, 1.0, ())


def _payments_graph() -> StructureGraph:
    # The worked Example A dependency graph.
    units = [
        _unit("POST /payments", kind="OPERATION"),
        _unit("PaymentRequest"),
        _unit("Payment"),
        _unit("Money"),
        _unit("Customer"),
        _unit("Address"),
    ]
    references = [
        _ref("POST /payments", "PaymentRequest"),
        _ref("POST /payments", "Payment"),
        _ref("PaymentRequest", "Money"),
        _ref("PaymentRequest", "Customer"),
        _ref("Customer", "Address"),
        _ref("Payment", "Money"),
    ]
    return build_graph(units, references)


def test_closure_of_operation_equals_full_request_response_closure() -> None:
    graph = _payments_graph()
    ids = [hop.unit_id for hop in closure_expand(["POST /payments"], graph, _REF_KINDS)]
    assert set(ids) - {"POST /payments"} == {
        "PaymentRequest",
        "Payment",
        "Money",
        "Customer",
        "Address",
    }


def test_shared_money_is_deduplicated() -> None:
    graph = _payments_graph()
    ids = [hop.unit_id for hop in closure_expand(["POST /payments"], graph, _REF_KINDS)]
    # Money is reachable via both PaymentRequest and Payment but appears once.
    assert ids.count("Money") == 1


def test_closure_is_byte_exact_on_repeat() -> None:
    graph = _payments_graph()
    first = closure_expand(["POST /payments"], graph, _REF_KINDS)
    second = closure_expand(["POST /payments"], graph, _REF_KINDS)
    assert first == second


def test_closure_includes_seeds() -> None:
    graph = _payments_graph()
    ids = {hop.unit_id for hop in closure_expand(["POST /payments"], graph, _REF_KINDS)}
    assert "POST /payments" in ids


def test_closure_terminates_on_cycles() -> None:
    graph = build_graph([_unit("A"), _unit("B")], [_ref("A", "B"), _ref("B", "A")])
    ids = {hop.unit_id for hop in closure_expand(["A"], graph, _REF_KINDS)}
    assert ids == {"A", "B"}


def test_closure_terminates_on_self_reference() -> None:
    graph = build_graph([_unit("A")], [_ref("A", "A")])
    ids = [hop.unit_id for hop in closure_expand(["A"], graph, _REF_KINDS)]
    assert ids == ["A"]


def test_no_reference_kinds_yields_only_seeds() -> None:
    graph = _payments_graph()
    ids = {hop.unit_id for hop in closure_expand(["POST /payments"], graph, ())}
    assert ids == {"POST /payments"}


def test_missing_seed_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        closure_expand(["nope"], _payments_graph(), _REF_KINDS)


def test_bare_str_seed_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="sequence of ids"):
        closure_expand("AB", _payments_graph(), _REF_KINDS)


def test_closure_spans_multiple_reference_kinds() -> None:
    # A path mixing all four hard reference kinds, including one multi-kind edge,
    # is fully reachable: the closure follows the union of the kinds.
    units = [_unit(name) for name in ("Root", "A", "B", "C", "D")]
    references = [
        _ref("Root", "A", "REFERENCES"),
        _ref("Root", "A", "IMPORTS"),  # multi-kind edge on the same pair
        _ref("A", "B", "FOREIGN_KEY"),
        _ref("B", "C", "IMPORTS"),
        _ref("C", "D", "CALLS"),
    ]
    graph = build_graph(units, references)
    ids = {hop.unit_id for hop in closure_expand(["Root"], graph, _REF_KINDS)}
    assert ids == {"Root", "A", "B", "C", "D"}


def test_closure_is_linear_on_deep_multi_kind_chain() -> None:
    # A long chain whose every edge carries all four kinds must stay cheap: a
    # per-kind multi-objective traversal would blow up here, a reachability BFS
    # does not. Completing at all and returning every node is the guard.
    length = 60
    units = [_unit(f"n{i}") for i in range(length)]
    references = [
        _ref(f"n{i}", f"n{i + 1}", kind) for i in range(length - 1) for kind in _REF_KINDS
    ]
    graph = build_graph(units, references)
    ids = {hop.unit_id for hop in closure_expand(["n0"], graph, _REF_KINDS)}
    assert ids == {f"n{i}" for i in range(length)}


def _kernel_corpus() -> tuple[list[Unit], list[Reference]]:
    units = [
        _unit("POST /payments", kind="OPERATION", text="create a payment"),
        _unit("PaymentRequest", text="request body fields"),
        _unit("Payment", text="stored record id"),
        _unit("Money", text="amount and currency"),
        _unit("Customer", text="buyer name reference"),
        _unit("Address", text="street and city"),
    ]
    references = [
        _ref("POST /payments", "PaymentRequest"),
        _ref("POST /payments", "Payment"),
        _ref("PaymentRequest", "Money"),
        _ref("PaymentRequest", "Customer"),
        _ref("Customer", "Address"),
        _ref("Payment", "Money"),
    ]
    return units, references


def _config(tier: str) -> KernelConfig:
    return KernelConfig(
        tier=tier,  # type: ignore[arg-type]
        bm25_profile={"text": 1.0, "title": 2.0, "breadcrumb": 1.0, "summary": 1.0},
        # A deliberately tight bounded budget: the closure must reach far units
        # (Address) that bounded expansion at one hop cannot.
        hop_budget_by_kind={"REFERENCES": 1},
        confidence_decay=0.9,
        enable_vector_lane=False,
        enable_rerank=False,
    )


def _indexed() -> RetrievalKernel:
    units, references = _kernel_corpus()
    kernel = RetrievalKernel()
    kernel.index(units, references)
    return kernel


def test_t1_receipt_reports_complete_closure() -> None:
    kernel = _indexed()
    bundle, receipt = kernel.retrieve("payment", 1000, _config("T1"))
    included = {rep.unit_id for rep in bundle.representations if rep.mode == "INCLUDE"}
    # The full closure (including the two-hop Address) is emitted under budget.
    assert {"PaymentRequest", "Payment", "Money", "Customer", "Address"} <= included
    assert receipt.reference_closure_complete is True
    assert receipt.determinism_class == "byte_exact"


def test_t0_does_not_reach_or_claim_the_full_closure() -> None:
    kernel = _indexed()
    bundle, receipt = kernel.retrieve("payment", 1000, _config("T0"))
    included = {rep.unit_id for rep in bundle.representations if rep.mode == "INCLUDE"}
    # T0's one-hop bounded budget never reaches the two-hop Address...
    assert "Address" not in included
    # ...and T0 computes no closure, so it never claims completeness.
    assert receipt.reference_closure_complete is False


def test_t1_retrieval_is_byte_identical_on_repeat() -> None:
    kernel = _indexed()
    first_bundle, first_receipt = kernel.retrieve("payment", 1000, _config("T1"))
    second_bundle, second_receipt = kernel.retrieve("payment", 1000, _config("T1"))
    assert first_bundle.render() == second_bundle.render()
    assert first_receipt == second_receipt
