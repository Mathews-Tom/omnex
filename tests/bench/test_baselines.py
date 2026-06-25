"""Benchmark baselines: chunk splitter, embedders, the pinned chunk-and-embed
retrieval, recall tuning, and the product-import-isolation contract.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from omnex.bench.baselines import (
    PINNED_CHUNK_EMBED,
    ChunkEmbedConfig,
    FastEmbedEmbedder,
    TfidfEmbedder,
    _encoder,
    chunk_and_embed_baseline,
    chunk_text,
    chunks_for_recall,
)
from omnex.bench.metrics import covered_labels, recall


def _token_count(text: str) -> int:
    return len(_encoder().encode(text))


def test_chunk_text_keeps_short_text_in_one_chunk() -> None:
    assert chunk_text("hello world", 256, 32) == ["hello world"]


def test_chunk_text_splits_long_text_into_multiple_windows() -> None:
    text = " ".join(str(i) for i in range(300))
    chunks = chunk_text(text, 50, 10)
    assert len(chunks) > 1
    assert all(chunks)


def test_chunk_overlap_duplicates_boundary_tokens() -> None:
    text = " ".join(str(i) for i in range(300))
    total = _token_count(text)
    overlapped = sum(_token_count(chunk) for chunk in chunk_text(text, 50, 10))
    disjoint = sum(_token_count(chunk) for chunk in chunk_text(text, 50, 0))
    # No overlap covers each token once; overlap re-emits the shared boundary.
    assert disjoint == total
    assert overlapped > total


def test_chunk_text_is_deterministic_on_repeat() -> None:
    text = " ".join(str(i) for i in range(120))
    assert chunk_text(text, 40, 8) == chunk_text(text, 40, 8)


def test_chunk_text_rejects_bad_sizes() -> None:
    for tokens, overlap in ((0, 0), (-1, 0)):
        try:
            chunk_text("x", tokens, overlap)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for chunk_tokens={tokens}")
    for overlap in (-1, 50, 60):
        try:
            chunk_text("x", 50, overlap)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for overlap={overlap}")


def test_chunk_text_handles_empty_string() -> None:
    assert chunk_text("", 256, 32) == []


def test_tfidf_embedder_is_deterministic_and_shares_one_space() -> None:
    embedder = TfidfEmbedder()
    first = embedder.embed(["payment charge", "weather forecast"])
    second = embedder.embed(["payment charge", "weather forecast"])
    assert first == second
    assert len(first) == 2
    assert len(first[0]) == len(first[1])  # dense vectors over one shared vocabulary


def test_chunk_and_embed_ranks_the_query_relevant_chunk_first() -> None:
    documents = ["payment charge record", "weather forecast sunny", "monetary currency amount"]
    ranked = chunk_and_embed_baseline(
        documents,
        "payment charge",
        TfidfEmbedder(),
        chunk_tokens=256,
        chunk_overlap=32,
    )
    assert ranked[0] == "payment charge record"
    assert set(ranked) == set(documents)  # every chunk returned, just reordered


def test_chunk_and_embed_is_deterministic_on_repeat() -> None:
    documents = [" ".join(str(i) for i in range(80)), "payment charge record"]
    first = chunk_and_embed_baseline(
        documents, "payment", TfidfEmbedder(), chunk_tokens=40, chunk_overlap=8
    )
    second = chunk_and_embed_baseline(
        documents, "payment", TfidfEmbedder(), chunk_tokens=40, chunk_overlap=8
    )
    assert first == second


def test_chunk_and_embed_handles_empty_corpus() -> None:
    ranked = chunk_and_embed_baseline([], "q", TfidfEmbedder(), chunk_tokens=256, chunk_overlap=32)
    assert ranked == []


def test_fastembed_embedder_embeds_with_the_pinned_model_when_installed() -> None:
    pytest.importorskip("fastembed")
    embedder = FastEmbedEmbedder()
    assert embedder.name == "BAAI/bge-small-en-v1.5"
    vectors = embedder.embed(["create a payment", "a monetary value"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1]) > 0


def test_baselines_imports_and_constructs_without_fastembed() -> None:
    # The lazy import is a contract: importing the module and constructing the
    # embedder must never require fastembed; only .embed() does. Force fastembed
    # to be unimportable in a fresh interpreter and prove that contract holds.
    code = (
        "import sys\n"
        "sys.modules['fastembed'] = None\n"  # makes `import fastembed` raise ImportError
        "import omnex.bench.baselines as b\n"
        "embedder = b.FastEmbedEmbedder()\n"
        "assert embedder.name == 'BAAI/bge-small-en-v1.5'\n"
        "try:\n"
        "    embedder.embed(['x'])\n"
        "except ImportError:\n"
        "    print('ok')\n"
        "else:\n"
        "    raise AssertionError('expected ImportError when fastembed is absent')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_pinned_chunk_embed_config_records_the_strong_configuration() -> None:
    assert ChunkEmbedConfig() == PINNED_CHUNK_EMBED
    assert PINNED_CHUNK_EMBED.chunk_tokens == 256
    assert PINNED_CHUNK_EMBED.chunk_overlap == 32
    assert PINNED_CHUNK_EMBED.embedder == "BAAI/bge-small-en-v1.5"
    assert PINNED_CHUNK_EMBED.rerank is None


def test_covered_labels_detects_markers_by_substring() -> None:
    text = "components / schemas / Money:\nA monetary value."
    labels = {"A monetary value.", "A postal location.", "A buyer."}
    assert covered_labels(text, labels) == frozenset({"A monetary value."})


def test_chunks_for_recall_is_tunable_and_monotonic() -> None:
    ranked = ["alpha a-marker", "beta b-marker", "gamma c-marker"]
    labels = {"a-marker", "b-marker", "c-marker"}
    third = chunks_for_recall(ranked, labels, 1.0 / 3.0)
    two_thirds = chunks_for_recall(ranked, labels, 2.0 / 3.0)
    full = chunks_for_recall(ranked, labels, 1.0)
    assert len(third) <= len(two_thirds) <= len(full)  # higher target -> longer prefix
    assert recall(covered_labels("\n".join(full), labels), labels) == 1.0
    assert chunks_for_recall(ranked, labels, 0.0) == []


def test_chunks_for_recall_returns_full_ranking_when_target_unreachable() -> None:
    ranked = ["alpha a-marker", "beta b-marker"]
    labels = {"a-marker", "b-marker", "missing-marker"}
    prefix = chunks_for_recall(ranked, labels, 1.0)
    assert prefix == ranked  # never reaches full recall; caller checks achieved recall
    assert recall(covered_labels("\n".join(prefix), labels), labels) < 1.0


def test_no_product_module_imports_omnex_bench() -> None:
    # The dependency runs one way: the benchmark may import the product, never the
    # reverse. Walk every product module in a fresh interpreter (so this test
    # process having imported omnex.bench cannot mask a violation) and assert none
    # pulled in omnex.bench.
    code = (
        "import importlib, pkgutil, sys\n"
        "import omnex\n"
        "def import_all(package, prefix):\n"
        "    for info in pkgutil.iter_modules(package.__path__, prefix):\n"
        "        # omnex.bench is the benchmark (never a product import target);\n"
        "        # omnex.integrations.* require optional extras and are import-guarded.\n"
        "        skip = ('omnex.bench', 'omnex.integrations')\n"
        "        if info.name in skip or any(info.name.startswith(s + '.') for s in skip):\n"
        "            continue\n"
        "        module = importlib.import_module(info.name)\n"
        "        if info.ispkg:\n"
        "            import_all(module, info.name + '.')\n"
        "import_all(omnex, 'omnex.')\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules if m == 'omnex.bench' or m.startswith('omnex.bench.')\n"
        ")\n"
        "assert not leaked, leaked\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
