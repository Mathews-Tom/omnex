"""Tests for the optional framework retriever adapters and the core boundary.

The framework tests auto-skip unless the matching extra is installed, so the
default core test run stays green without LangChain or LlamaIndex. The core
boundary tests always run: they assert that ``import omnex`` pulls neither
framework and that omnex's declared core dependencies stay
``networkx``/``tiktoken``/``click`` -- every integration dependency is gated
behind an extra.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from importlib.metadata import requires

import pytest

from omnex import KernelConfig, api, index
from omnex.ir.types import Reference, Span, Unit, UnitKind
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.receipt import Receipt

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
_HAS_LLAMA_INDEX = importlib.util.find_spec("llama_index") is not None

_CORE_DEPENDENCIES = {"networkx", "tiktoken", "click"}


def _unit(uid: str, text: str, *, title: str, kind: UnitKind = "SECTION") -> Unit:
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


def _expected(query: str, budget: int) -> tuple[ContextBundle, Receipt]:
    """The bundle and receipt omnex itself returns, for fidelity comparison."""
    units, references = _corpus()
    return api.query(units, query, budget, _config(), references)


def _requirement_name(requirement: str) -> str:
    """The distribution name at the head of a PEP 508 requirement string."""
    return re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].strip().lower()


def _split_requirements() -> tuple[set[str], set[str]]:
    """Return (core, extra) distribution names from omnex's installed metadata."""
    declared = requires("omnex") or []
    core = {_requirement_name(req) for req in declared if "extra ==" not in req}
    extra = {_requirement_name(req) for req in declared if "extra ==" in req}
    return core, extra


def test_core_dependencies_exclude_integration_extras() -> None:
    """omnex's core deps stay the three; every framework dep is behind an extra."""
    core, extra = _split_requirements()
    assert core == _CORE_DEPENDENCIES
    assert "langchain-core" in extra
    assert "langchain-core" not in core


def test_import_omnex_does_not_load_integration_frameworks() -> None:
    """A fresh ``import omnex`` loads no integration framework module."""
    code = (
        "import sys, omnex; "
        "loaded = [m for m in sys.modules if m == 'langchain_core' "
        "or m.startswith('langchain_core.') or m == 'llama_index' "
        "or m.startswith('llama_index.')]; "
        "assert not loaded, loaded; "
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "clean"


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="requires the [langchain] extra")
def test_langchain_retriever_emits_documents_with_provenance() -> None:
    from langchain_core.documents import Document

    from omnex.integrations.langchain import OmnexRetriever

    units, references = _corpus()
    kernel = index(units, references)
    retriever = OmnexRetriever(kernel=kernel, config=_config(), budget_tokens=400)

    docs = retriever.invoke("retrieval indexing")

    assert docs, "expected at least one document"
    assert all(isinstance(doc, Document) for doc in docs)
    for doc in docs:
        assert doc.page_content
        assert doc.metadata["unit_id"] in {"u_a", "u_b"}
        assert doc.metadata["mode"] in {"INCLUDE", "COMPRESS", "ELIDE"}
        assert doc.metadata["document_id"] == "doc:1"
        receipt = doc.metadata["omnex_receipt"]
        assert isinstance(receipt, dict)
        assert receipt["determinism_class"] == "byte_exact"


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="requires the [langchain] extra")
def test_langchain_retriever_preserves_the_omnex_returned_set() -> None:
    """The retriever emits exactly omnex's packed chunks, in order, unaltered."""
    from omnex.integrations.langchain import OmnexRetriever

    units, references = _corpus()
    kernel = index(units, references)
    retriever = OmnexRetriever(kernel=kernel, config=_config(), budget_tokens=400)

    docs = retriever.invoke("retrieval indexing")

    bundle, _ = _expected("retrieval indexing", 400)
    expected_pairs = [
        (rep.unit_id, rep.text) for rep in bundle.representations if rep.mode != "SKIP"
    ]
    actual_pairs = [(doc.metadata["unit_id"], doc.page_content) for doc in docs]
    assert actual_pairs == expected_pairs
