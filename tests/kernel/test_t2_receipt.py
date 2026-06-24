"""Tests for the T2 determinism class, embedding provenance, and the cache key.

The cache-key composition and the byte-exact T0/T1 assertions run everywhere,
including CI without the ``[embed]`` extra. The T2 receipt assertions require a
real embedding run and so ``importorskip`` fastembed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import omnex
from omnex import KernelConfig
from omnex.kernel.vector import DEFAULT_EMBED_MODEL, embedding_cache_key

_DOCS = (Path(__file__).resolve().parents[1] / "fixtures" / "tls_docs").resolve()
_SOURCES = [
    _DOCS / "ingress.md",
    _DOCS / "securing-traffic.md",
    _DOCS / "service-discovery.md",
]
_QUESTION = "How do I configure TLS for the ingress controller?"


def _config(tier: str, *, vector: bool) -> KernelConfig:
    return KernelConfig(
        tier=tier,  # type: ignore[arg-type]
        bm25_profile={"title": 2.0, "breadcrumb": 1.5, "text": 1.0, "summary": 1.0},
        hop_budget_by_kind={"CONTAINS": 2, "CROSS_REF": 1, "REFERENCES": 4, "SIBLING": 0},
        confidence_decay=0.8,
        enable_vector_lane=vector,
        enable_rerank=False,
    )


# --- cache key composition (no extra required) ---


def test_cache_key_includes_content_hash_and_model_version() -> None:
    text = "Terminate TLS at the ingress with a certificate secret."
    key = embedding_cache_key(text, "model-a")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() in key
    assert "model-a" in key


def test_cache_key_changes_with_model_version() -> None:
    text = "Terminate TLS at the ingress with a certificate secret."
    assert embedding_cache_key(text, "model-a") != embedding_cache_key(text, "model-b")


def test_cache_key_changes_with_content() -> None:
    assert embedding_cache_key("alpha", "model-a") != embedding_cache_key("beta", "model-a")


# --- byte-exact tiers stay byte_exact (no extra required) ---


def test_t0_and_t1_runs_remain_byte_exact() -> None:
    for tier in ("T0", "T1"):
        _, receipt = omnex.query_sources(_SOURCES, _QUESTION, 200, _config(tier, vector=False))
        assert receipt.determinism_class == "byte_exact"
        assert receipt.model_used is False
        assert receipt.model_version is None
        assert receipt.embedding_provenance is None


# --- T2 determinism class and provenance (requires the embed extra) ---


def test_t2_run_is_pinned_reproducible_with_model_version() -> None:
    pytest.importorskip("fastembed")
    _, receipt = omnex.query_sources(_SOURCES, _QUESTION, 300, _config("T2", vector=True))
    # pinned_reproducible is a distinct, weaker class than the byte-exact floor,
    # so a vector-assisted run is never labeled byte_exact.
    assert receipt.determinism_class == "pinned_reproducible"
    assert receipt.model_used is True
    assert receipt.model_version == DEFAULT_EMBED_MODEL


def test_t2_receipt_records_embedding_provenance() -> None:
    pytest.importorskip("fastembed")
    _, receipt = omnex.query_sources(_SOURCES, _QUESTION, 300, _config("T2", vector=True))
    provenance = receipt.embedding_provenance
    assert provenance is not None
    assert provenance.model == DEFAULT_EMBED_MODEL
    assert provenance.tokenizer  # the bundled tokenizer identity
    assert "fastembed" in provenance.runtime
    assert provenance.architecture  # system/machine, non-empty
