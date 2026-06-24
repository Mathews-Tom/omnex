"""End-to-end spec query: closure completeness, tokens, determinism, no model."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

import omnex
from omnex import KernelConfig
from omnex.adapters.spec import SpecAdapter

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_PAYMENTS = _FIXTURES / "payments_openapi.json"
_QUESTION = "what is the request/response shape for creating a payment"
_CLOSURE = {"PaymentRequest", "Payment", "Money", "Customer", "Address"}


def _t1_config() -> KernelConfig:
    return KernelConfig(
        tier="T1",
        bm25_profile={"text": 1.0, "title": 3.0, "breadcrumb": 1.0, "summary": 1.0},
        hop_budget_by_kind={"REFERENCES": 1},
        confidence_decay=0.9,
        enable_vector_lane=False,
        enable_rerank=False,
    )


def _included_titles(bundle: Any) -> list[str]:
    return [
        bundle.units[rep.unit_id].title for rep in bundle.representations if rep.mode == "INCLUDE"
    ]


def test_t1_query_returns_the_complete_request_response_closure() -> None:
    bundle, _ = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, _t1_config())
    titles = _included_titles(bundle)
    # The full request and response closure is present and emitted in full.
    assert set(titles) >= _CLOSURE
    assert "POST /payments" in titles


def test_shared_money_schema_is_emitted_once() -> None:
    bundle, _ = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, _t1_config())
    titles = _included_titles(bundle)
    # Money is referenced by both PaymentRequest and Payment but packed once.
    assert titles.count("Money") == 1


def test_returned_tokens_are_below_the_full_dump() -> None:
    _, receipt = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, _t1_config())
    assert receipt.returned_tokens < receipt.baseline_tokens


def test_receipt_proves_closure_complete_byte_exact_and_zero_model() -> None:
    _, receipt = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, _t1_config())
    assert receipt.reference_closure_complete is True
    assert receipt.determinism_class == "byte_exact"
    assert receipt.model_used is False
    assert receipt.model_version is None
    assert receipt.extraction_used is False


def test_query_renders_canonical_spec_fragments() -> None:
    bundle, _ = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, _t1_config())
    rendered = bundle.render()
    # Spec units render as path-qualified canonical fragments.
    assert "components / schemas / Money:" in rendered
    assert "paths / POST /payments:" in rendered


def test_query_is_byte_identical_on_repeat() -> None:
    first_bundle, first_receipt = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, _t1_config())
    second_bundle, second_receipt = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, _t1_config())
    assert first_bundle.render() == second_bundle.render()
    assert first_receipt == second_receipt


def test_spec_path_makes_no_model_or_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # The no-model guarantee is the receipt assertion below (model_used /
    # extraction_used). This socket block is a regression guard against
    # accidental network access; warm the deterministic tiktoken tokenizer from
    # its local cache first, since the retrieval path itself must touch no socket.
    SpecAdapter().ingest(_PAYMENTS)

    def _blocked(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("the spec path must make no network call")

    monkeypatch.setattr(socket, "socket", _blocked)
    bundle, receipt = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, _t1_config())
    assert set(_included_titles(bundle)) >= _CLOSURE
    assert receipt.model_used is False
    assert receipt.extraction_used is False
    assert receipt.reference_closure_complete is True


def test_index_sources_builds_a_queryable_kernel() -> None:
    # The reusable source-level kernel answers identically to the one-shot query
    # (the receipts differ only in query_sources' full-dump baseline override).
    config = _t1_config()
    kernel = omnex.index_sources([_PAYMENTS])
    reused_bundle, reused_receipt = kernel.retrieve(_QUESTION, 5000, config)
    one_shot_bundle, _ = omnex.query_sources([_PAYMENTS], _QUESTION, 5000, config)
    assert reused_bundle.render() == one_shot_bundle.render()
    assert set(_included_titles(reused_bundle)) >= _CLOSURE
    assert reused_receipt.reference_closure_complete is True
