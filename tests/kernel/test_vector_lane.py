"""Tests for the opt-in T2 vector lane: fusion, default-off, core independence.

The fusion tests run only where ``fastembed`` is installed (the ``[embed]`` extra);
the default-off and core-install tests run everywhere, including CI without the
extra, because the byte-exact floor must never depend on it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import omnex
from omnex import KernelConfig
from omnex.ir.types import Span, Unit
from omnex.kernel.kernel import RetrievalKernel
from omnex.kernel.vector import VectorIndex

_DOCS = (Path(__file__).resolve().parents[1] / "fixtures" / "tls_docs").resolve()
_SOURCES = [
    _DOCS / "ingress.md",
    _DOCS / "securing-traffic.md",
    _DOCS / "service-discovery.md",
]
_QUESTION = "How do I configure TLS for the ingress controller?"
# Body text unique to the semantically distant page: it shares no query vocabulary,
# and with CROSS_REF off no structural edge reaches it, so its presence proves the
# vector lane surfaced the page -- not lexical matching, not graph expansion.
_DISTANT_BODY = "handshake"


def _config(tier: str, *, vector: bool, cross_ref: int = 0) -> KernelConfig:
    return KernelConfig(
        tier=tier,  # type: ignore[arg-type]
        bm25_profile={"title": 2.0, "breadcrumb": 1.5, "text": 1.0, "summary": 1.0},
        hop_budget_by_kind={"CONTAINS": 2, "CROSS_REF": cross_ref, "SIBLING": 0, "CITES": 1},
        confidence_decay=0.8,
        enable_vector_lane=vector,
        enable_rerank=False,
    )


def _unit(unit_id: str, text: str) -> Unit:
    return Unit(
        id=unit_id,
        document_id="doc",
        span=Span(0, len(text)),
        text=text,
        token_count=len(text.split()),
        title=None,
        breadcrumb=(),
        kind="PARAGRAPH",
        summary=None,
        protect=False,
    )


# --- lane off by default (no fastembed required) ---


def test_lane_is_off_by_default_on_the_lexical_floor() -> None:
    # A plain T0 run takes the lexical floor: no model, lexical-only recall basis,
    # byte-exact determinism, and the distant page (no cross-ref edge) stays missing.
    bundle, receipt = omnex.query_sources(_SOURCES, _QUESTION, 200, _config("T0", vector=False))
    assert receipt.model_used is False
    assert receipt.model_version is None
    assert receipt.recall_basis == "lexical"
    assert receipt.determinism_class == "byte_exact"
    assert _DISTANT_BODY not in bundle.render()


# --- fusion of the lexical and vector lanes (requires the embed extra) ---


def test_t2_fuses_lexical_and_vector_candidates() -> None:
    pytest.importorskip("fastembed")
    # CROSS_REF is off, so graph expansion cannot rescue the distant page; only the
    # fused vector lane can surface it. The lexical TLS sections must still appear,
    # proving both lanes contribute rather than the vector lane replacing the other.
    t0_bundle, _ = omnex.query_sources(_SOURCES, _QUESTION, 200, _config("T0", vector=False))
    t2_bundle, receipt = omnex.query_sources(_SOURCES, _QUESTION, 200, _config("T2", vector=True))
    rendered = t2_bundle.render()
    assert _DISTANT_BODY not in t0_bundle.render()  # lexical floor misses it
    assert _DISTANT_BODY in rendered  # the vector lane recovers it
    assert "TLS secrets" in rendered  # the lexical lane still contributes
    assert receipt.recall_basis == "lexical_plus_vector"
    assert receipt.model_used is True
    # The lexical T0 floor always runs and is fused with the vector lane, so the
    # receipt reports both tiers, not T2 alone.
    assert receipt.tiers_run == ("T0", "T2")


def test_vector_index_ranks_by_similarity() -> None:
    pytest.importorskip("fastembed")
    units = [
        _unit("u_tls", "Terminate TLS at the ingress with a certificate stored in a secret."),
        _unit("u_dns", "Service discovery resolves stable names to the healthy backend pods."),
    ]
    index = VectorIndex()
    index.index_units(units)
    ranked = index.search("how do I configure TLS certificates", limit=2)
    assert [unit_id for unit_id, _ in ranked] == ["u_tls", "u_dns"]
    assert ranked[0][1] >= ranked[1][1]  # ordered by descending similarity


def test_t2_run_is_reproducible_on_repeat() -> None:
    pytest.importorskip("fastembed")
    # Pinned-reproducible, not byte-exact: on one machine/runtime two identical T2
    # runs must still produce an identical bundle and receipt.
    first_bundle, first_receipt = omnex.query_sources(
        _SOURCES, _QUESTION, 200, _config("T2", vector=True)
    )
    second_bundle, second_receipt = omnex.query_sources(
        _SOURCES, _QUESTION, 200, _config("T2", vector=True)
    )
    assert first_bundle.render() == second_bundle.render()
    assert repr(first_receipt) == repr(second_receipt)


def test_reindex_invalidates_the_vector_cache() -> None:
    pytest.importorskip("fastembed")
    # Reusing one kernel: index a corpus, run a T2 query (building the vector
    # cache), then re-index a disjoint corpus. The next T2 query must reflect the
    # new corpus -- never stale embeddings or a KeyError from ids absent in the
    # rebuilt graph.
    config = _config("T2", vector=True)
    kernel = RetrievalKernel()
    kernel.index([_unit("u_old", "An unrelated note about lunch menus and cafeteria hours.")])
    kernel.retrieve("configure TLS", 200, config)
    kernel.index([_unit("u_new", "Terminate TLS at the ingress using a certificate secret.")])
    bundle, receipt = kernel.retrieve("configure TLS", 200, config)
    rendered = bundle.render()
    assert "certificate secret" in rendered
    assert "cafeteria" not in rendered
    assert receipt.recall_basis == "lexical_plus_vector"


# --- core install without the embed extra (runs everywhere) ---


def test_core_install_without_embed_runs_t0_and_t1_unchanged() -> None:
    # Force fastembed unimportable in a fresh interpreter and prove the core install
    # contract: omnex and the vector module import, the lane reports unavailable, the
    # byte-exact T0/T1 paths still run, and asking for the lane fails loud with an
    # actionable message rather than silently degrading.
    code = (
        "import sys\n"
        "sys.modules['fastembed'] = None\n"  # makes `import fastembed` fail
        "import omnex\n"
        "from omnex import KernelConfig\n"
        "from omnex.kernel.vector import vector_lane_available\n"
        "from pathlib import Path\n"
        "assert vector_lane_available() is False\n"
        f"docs = Path({str(_DOCS)!r})\n"
        "sources = [docs / 'ingress.md', docs / 'securing-traffic.md']\n"
        "profile = {'text': 1.0, 'title': 2.0, 'breadcrumb': 1.5, 'summary': 1.0}\n"
        "def cfg(tier):\n"
        "    return KernelConfig(tier=tier, bm25_profile=profile,\n"
        "        hop_budget_by_kind={'CONTAINS': 2, 'CROSS_REF': 1, 'REFERENCES': 4,\n"
        "                            'SIBLING': 0},\n"
        "        confidence_decay=0.8, enable_vector_lane=False, enable_rerank=False)\n"
        "for tier in ('T0', 'T1'):\n"
        "    _, receipt = omnex.query_sources(sources, 'configure TLS', 200, cfg(tier))\n"
        "    assert receipt.determinism_class == 'byte_exact', receipt.determinism_class\n"
        "    assert receipt.model_used is False\n"
        "t2 = KernelConfig(tier='T2', bm25_profile=profile,\n"
        "    hop_budget_by_kind={'CONTAINS': 1}, confidence_decay=0.9,\n"
        "    enable_vector_lane=True, enable_rerank=False)\n"
        "try:\n"
        "    omnex.query_sources(sources, 'configure TLS', 200, t2)\n"
        "except ModuleNotFoundError as exc:\n"
        "    assert 'omnex[embed]' in str(exc), exc\n"
        "    print('ok')\n"
        "else:\n"
        "    raise AssertionError('expected ModuleNotFoundError without fastembed')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
