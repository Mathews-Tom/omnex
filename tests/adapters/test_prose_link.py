"""Tests for the prose adapter's edge recovery: CONTAINS/SIBLING/CROSS_REF/CITES."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnex.adapters.prose import ProseAdapter
from omnex.ir.graph import build_graph
from omnex.ir.types import Reference, Unit

_DOCS = (Path(__file__).resolve().parent.parent / "fixtures" / "tls_docs").resolve()
_INGRESS = (_DOCS / "ingress.md").resolve()
_SECURING = (_DOCS / "securing-traffic.md").resolve()


def _linked() -> tuple[list[Unit], list[Reference]]:
    """Ingest, parse, and link both TLS pages into one combined corpus."""
    adapter = ProseAdapter()
    units: list[Unit] = []
    references: list[Reference] = []
    for path in (_INGRESS, _SECURING):
        document = adapter.ingest(path)
        page_units = adapter.parse(document)
        units.extend(page_units)
        references.extend(adapter.link(document, page_units))
    return units, references


def _section(units: list[Unit], title: str) -> Unit:
    return next(unit for unit in units if unit.kind == "SECTION" and unit.title == title)


def _content(units: list[Unit], substring: str) -> Unit:
    return next(unit for unit in units if unit.kind != "SECTION" and substring in unit.text)


def _has(refs: list[Reference], source: Unit, kind: str, target: Unit) -> bool:
    return any(
        ref.source_id == source.id and ref.kind == kind and ref.target_id == target.id
        for ref in refs
    )


def _confidence(refs: list[Reference], source: Unit, kind: str, target: Unit) -> float:
    return next(
        ref.confidence
        for ref in refs
        if ref.source_id == source.id and ref.kind == kind and ref.target_id == target.id
    )


def test_contains_tree_links_sections_to_children() -> None:
    units, refs = _linked()
    ingress = _section(units, "Ingress")
    tls = _section(units, "TLS secrets")
    verification = _section(units, "Verification")
    manifest = _content(units, "apiVersion")
    tls_paragraph = _content(units, "Store the TLS")
    assert _has(refs, ingress, "CONTAINS", tls)
    assert _has(refs, ingress, "CONTAINS", verification)
    assert _has(refs, tls, "CONTAINS", manifest)
    assert _has(refs, tls, "CONTAINS", tls_paragraph)


def test_sibling_links_adjacent_sections_both_ways() -> None:
    units, refs = _linked()
    tls = _section(units, "TLS secrets")
    verification = _section(units, "Verification")
    assert _has(refs, tls, "SIBLING", verification)
    assert _has(refs, verification, "SIBLING", tls)
    # A lone top-level section has no sibling across documents.
    ingress = _section(units, "Ingress")
    securing = _section(units, "Securing traffic with certificates")
    assert not _has(refs, ingress, "SIBLING", securing)


def test_crossref_resolves_intra_and_inter_document() -> None:
    units, refs = _linked()
    tls = _section(units, "TLS secrets")
    securing = _section(units, "Securing traffic with certificates")
    preamble = _content(units, "routes external traffic")
    verify = _content(units, "Confirm the ingress")
    # Inter-document: the Ingress preamble links to the cross-linked page.
    assert _has(refs, preamble, "CROSS_REF", securing)
    assert _confidence(refs, preamble, "CROSS_REF", securing) == 0.9
    # Intra-document: the Verification step anchors back to TLS secrets.
    assert _has(refs, verify, "CROSS_REF", tls)
    assert _confidence(refs, verify, "CROSS_REF", tls) == 1.0


def test_cites_resolves_footnote_to_its_definition() -> None:
    units, refs = _linked()
    tls_paragraph = _content(units, "Store the TLS")
    footnote = _content(units, "[^tls]:")
    assert _has(refs, tls_paragraph, "CITES", footnote)
    assert _confidence(refs, tls_paragraph, "CITES", footnote) == 0.6


def test_structural_edge_confidence_is_per_kind() -> None:
    units, refs = _linked()
    ingress = _section(units, "Ingress")
    tls = _section(units, "TLS secrets")
    verification = _section(units, "Verification")
    assert _confidence(refs, ingress, "CONTAINS", tls) == 1.0
    assert _confidence(refs, tls, "SIBLING", verification) == 0.5


def test_links_build_a_structure_graph() -> None:
    units, refs = _linked()
    graph = build_graph(units, refs)
    assert len(graph) == len(units)


def test_manifest_unit_is_protected() -> None:
    units, _ = _linked()
    manifest = _content(units, "apiVersion")
    assert manifest.protect is True


def test_link_is_deterministic() -> None:
    adapter = ProseAdapter()
    document = adapter.ingest(_INGRESS)
    units = adapter.parse(document)
    assert adapter.link(document, units) == adapter.link(document, units)


def test_external_link_is_not_a_crossref(tmp_path: Path) -> None:
    source = tmp_path / "x.md"
    source.write_text(
        "# H\n\nSee the [docs](https://example.com/guide) online.\n", encoding="utf-8"
    )
    adapter = ProseAdapter()
    document = adapter.ingest(source)
    refs = adapter.link(document, adapter.parse(document))
    assert not any(ref.kind == "CROSS_REF" for ref in refs)


def test_link_to_unindexed_neighbor_emits_no_edge(tmp_path: Path) -> None:
    # A link to a sibling file that is not on disk resolves to nothing, so no
    # cross-reference edge is fabricated (it would dangle into an empty corpus).
    source = tmp_path / "a.md"
    source.write_text("# A\n\nSee [other](missing.md) for details.\n", encoding="utf-8")
    adapter = ProseAdapter()
    document = adapter.ingest(source)
    refs = adapter.link(document, adapter.parse(document))
    assert not any(ref.kind == "CROSS_REF" for ref in refs)


def test_rest_links_recover_crossref_and_cites(tmp_path: Path) -> None:
    source = tmp_path / "d.rst"
    source.write_text(
        "Intro\n=====\n\nSee `Details`_ and a note [1]_.\n\n"
        "Details\n-------\n\nBody text.\n\n.. [1] A footnote about details.\n",
        encoding="utf-8",
    )
    adapter = ProseAdapter()
    document = adapter.ingest(source)
    refs = adapter.link(document, adapter.parse(document))
    kinds = {ref.kind for ref in refs}
    assert {"CONTAINS", "CROSS_REF", "CITES"} <= kinds


def test_link_rejects_mismatched_units() -> None:
    adapter = ProseAdapter()
    document = adapter.ingest(_INGRESS)
    with pytest.raises(ValueError, match="do not match the parsed source"):
        adapter.link(document, [])
