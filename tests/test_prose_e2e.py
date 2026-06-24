"""End-to-end T0 prose query: budget win, protected manifest, honest receipt."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

import omnex
from omnex import KernelConfig
from omnex.adapters.prose import ProseAdapter

_DOCS = (Path(__file__).resolve().parent / "fixtures" / "tls_docs").resolve()
_INGRESS = (_DOCS / "ingress.md").resolve()
_SECURING = (_DOCS / "securing-traffic.md").resolve()
_DISTRACTOR = (_DOCS / "service-discovery.md").resolve()
_SOURCES = [_INGRESS, _SECURING, _DISTRACTOR]
_QUESTION = "How do I configure TLS for the ingress controller?"
_BUDGET = 200
# Text that appears only in the body of the cross-linked page, never in the link
# anchor text or any other document, so its presence proves the page itself was
# retrieved (not merely the link that names it).
_SECURING_BODY = "handshake"


def _t0_config(cross_ref_hops: int = 1) -> KernelConfig:
    return KernelConfig(
        tier="T0",
        bm25_profile={"title": 2.0, "breadcrumb": 1.5, "text": 1.0, "summary": 1.0},
        # CONTAINS pulls a matched section's content; CROSS_REF reaches the
        # cross-linked page; SIBLING is present in the graph but not traversed here.
        hop_budget_by_kind={
            "CONTAINS": 2,
            "CROSS_REF": cross_ref_hops,
            "SIBLING": 0,
            "CITES": 1,
        },
        confidence_decay=0.8,
        enable_vector_lane=False,
        enable_rerank=False,
    )


def test_query_returns_lexical_matches_and_the_manifest() -> None:
    bundle, _ = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())
    rendered = bundle.render()
    # The lexical TLS sections matched...
    assert "TLS secrets" in rendered
    assert "Ingress" in rendered
    # ...the protected YAML manifest is present in full...
    assert "BASE64_ENCODED_CERT" in rendered
    assert "BASE64_ENCODED_KEY" in rendered
    # ...rendered as markdown with its section-path breadcrumb.
    assert "> Ingress / TLS secrets" in rendered


def test_crossref_reaches_the_semantically_distant_page() -> None:
    # The cross-linked page shares no query vocabulary (no TLS/configure/ingress/
    # controller), so it is not a lexical match; its body is retrieved only because
    # it is a CROSS_REF neighbor of the matched Ingress preamble. Asserting on body
    # text unique to that page (not the link anchor that names it) proves this.
    reached, _ = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config(cross_ref_hops=1))
    assert _SECURING_BODY in reached.render()
    # With no CROSS_REF budget the page is lexically unreachable and stays out,
    # so the edge -- not a lexical hit -- is what pulled it in.
    without, _ = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config(cross_ref_hops=0))
    assert _SECURING_BODY not in without.render()


def test_returned_tokens_are_far_below_the_full_dump() -> None:
    _, receipt = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())
    assert receipt.returned_tokens <= _BUDGET
    assert receipt.returned_tokens < receipt.baseline_tokens
    # The whole-corpus dump is several times the budget, so the returned context
    # is far below paste-everything.
    assert receipt.baseline_tokens >= 3 * _BUDGET


def test_protected_manifest_is_included_never_compressed_or_elided() -> None:
    bundle, _ = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())
    protected = [rep for rep in bundle.representations if bundle.units[rep.unit_id].protect]
    manifest = next(rep for rep in protected if "apiVersion" in bundle.units[rep.unit_id].text)
    assert manifest.mode == "INCLUDE"
    assert all(rep.mode in ("INCLUDE", "SKIP") for rep in protected)


def test_receipt_is_byte_exact_lexical_and_zero_model() -> None:
    _, receipt = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())
    assert receipt.determinism_class == "byte_exact"
    assert receipt.tiers_run == ("T0",)
    assert receipt.model_used is False
    assert receipt.model_version is None
    assert receipt.extraction_used is False
    # T0 computes no reference closure, so it claims none.
    assert receipt.reference_closure_complete is False


def test_receipt_states_the_lexical_only_limitation() -> None:
    _, receipt = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())
    assert receipt.recall_basis == "lexical"
    limitation = " ".join(receipt.recall_limitations).lower()
    assert "lexical" in limitation
    assert "cross_ref" in limitation
    # The receipt never claims to beat embeddings at T0.
    assert "no claim to beat embeddings" in limitation


def test_query_is_byte_identical_on_repeat() -> None:
    first_bundle, first_receipt = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())
    second_bundle, second_receipt = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())
    assert first_bundle.render() == second_bundle.render()
    assert first_receipt == second_receipt


def test_prose_path_makes_no_model_or_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # The no-model guarantee is the receipt assertion (model_used/extraction_used).
    # This socket block is a regression guard against accidental network access;
    # warm the deterministic tiktoken tokenizer from its local cache first, since
    # the retrieval path itself must touch no socket.
    ProseAdapter().ingest(_INGRESS)

    def _blocked(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("the prose path must make no network call")

    monkeypatch.setattr(socket, "socket", _blocked)
    _, receipt = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())
    assert receipt.model_used is False
    assert receipt.extraction_used is False


def test_query_drops_cross_link_to_an_unindexed_document() -> None:
    # Indexing only the Ingress page leaves its cross-link to securing-traffic.md
    # pointing outside the corpus. Routing scopes that edge away instead of letting
    # it dangle into the fail-loud graph, so the query still answers and the
    # unindexed page's body is absent.
    bundle, receipt = omnex.query_sources([_INGRESS], _QUESTION, _BUDGET, _t0_config())
    assert receipt.returned_tokens > 0
    assert _SECURING_BODY not in bundle.render()


def test_inter_document_link_resolves_through_path_normalization() -> None:
    # The neighbor is passed by a non-normalized path; routing resolves every
    # source to a canonical form before ingest, so the inter-document CROSS_REF
    # still lands on the neighbor's own units and its body is retrieved.
    non_normalized = _DOCS / ".." / "tls_docs" / "securing-traffic.md"
    assert ".." in str(non_normalized)
    sources = [_INGRESS, non_normalized, _DISTRACTOR]
    bundle, _ = omnex.query_sources(sources, _QUESTION, _BUDGET, _t0_config())
    assert _SECURING_BODY in bundle.render()


def test_duplicate_sources_do_not_inflate_the_baseline() -> None:
    # The same physical file passed twice is indexed once, so the full-dump
    # baseline counts it once and the token comparison stays honest.
    once = omnex.query_sources(_SOURCES, _QUESTION, _BUDGET, _t0_config())[1]
    twice = omnex.query_sources([*_SOURCES, _INGRESS], _QUESTION, _BUDGET, _t0_config())[1]
    assert twice.baseline_tokens == once.baseline_tokens
    assert twice.returned_tokens == once.returned_tokens
