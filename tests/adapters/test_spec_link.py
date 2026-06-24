"""Tests for the spec adapter's $ref linking: edges, dedup, cycles, evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnex.adapters.spec import SpecAdapter
from omnex.ir.graph import build_graph
from omnex.ir.types import Reference, Unit

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_PAYMENTS = _FIXTURES / "payments_openapi.json"
_FK = _FIXTURES / "fk_schema.json"


def _linked(source: Path) -> tuple[list[Unit], list[Reference]]:
    adapter = SpecAdapter()
    document = adapter.ingest(source)
    units = adapter.parse(document)
    return units, adapter.link(document, units)


def _named_edges(units: list[Unit], references: list[Reference]) -> set[tuple[str, str, str]]:
    title = {unit.id: unit.title or unit.id for unit in units}
    return {(title[ref.source_id], ref.kind, title[ref.target_id]) for ref in references}


def test_payments_reproduces_the_design_graph() -> None:
    units, references = _linked(_PAYMENTS)
    # The exact edge set is pinned so a mis-attributed source, a dropped edge, or
    # a spurious edge fails the test. The fixture's GET /health -> HealthStatus is
    # a real edge alongside the six Example-A payment edges.
    assert _named_edges(units, references) == {
        ("POST /payments", "REFERENCES", "PaymentRequest"),
        ("POST /payments", "REFERENCES", "Payment"),
        ("PaymentRequest", "REFERENCES", "Money"),
        ("PaymentRequest", "REFERENCES", "Customer"),
        ("Customer", "REFERENCES", "Address"),
        ("Payment", "REFERENCES", "Money"),
        ("GET /health", "REFERENCES", "HealthStatus"),
    }
    assert len(references) == 7


def test_payments_edges_are_confident_references() -> None:
    _, references = _linked(_PAYMENTS)
    assert references, "expected reference edges"
    assert all(ref.confidence == 1.0 for ref in references)


def test_shared_money_target_has_two_distinct_sources() -> None:
    # Money is shared: PaymentRequest and Payment each reference it. These are two
    # distinct edges to one target; the closure dedups Money itself later.
    units, references = _linked(_PAYMENTS)
    title = {unit.id: unit.title for unit in units}
    money_sources = {title[ref.source_id] for ref in references if title[ref.target_id] == "Money"}
    assert money_sources == {"PaymentRequest", "Payment"}


def test_evidence_carries_the_ref_pointer() -> None:
    _, references = _linked(_PAYMENTS)
    for ref in references:
        assert ref.evidence
        for pointer in ref.evidence:
            assert pointer.startswith("/")
            assert pointer.endswith("/$ref")


def test_link_is_deterministic() -> None:
    adapter = SpecAdapter()
    document = adapter.ingest(_PAYMENTS)
    units = adapter.parse(document)
    assert adapter.link(document, units) == adapter.link(document, units)


def test_links_build_a_structure_graph() -> None:
    units, references = _linked(_PAYMENTS)
    graph = build_graph(units, references)
    assert len(graph) == len(units)


def test_foreign_key_and_schema_to_field_edges() -> None:
    units, references = _linked(_FK)
    by_id = {unit.id: unit for unit in units}
    edges = _named_edges(units, references)
    # owner_id is a foreign-key-named reference property.
    assert ("Order", "FOREIGN_KEY", "Owner") in edges
    # status references a field, exercising the schema-to-field edge shape.
    field_targets = {
        by_id[ref.target_id].kind for ref in references if by_id[ref.target_id].title == "name"
    }
    assert "FIELD" in field_targets
    assert ("Order", "REFERENCES", "name") in edges


def test_self_reference_cycle_is_emitted_once_and_terminates() -> None:
    units, references = _linked(_FK)
    self_edges = [ref for ref in references if ref.source_id == ref.target_id]
    assert len(self_edges) == 1
    # The cyclic schema still builds a graph without diverging.
    build_graph(units, references)


def test_mutually_recursive_schemas_terminate(tmp_path: Path) -> None:
    spec = tmp_path / "mutual.json"
    spec.write_text(
        '{"$schema": "x", "$defs": {"A": {"type": "object", "properties": '
        '{"b": {"$ref": "#/$defs/B"}}}, "B": {"type": "object", "properties": '
        '{"a": {"$ref": "#/$defs/A"}}}}}',
        encoding="utf-8",
    )
    units, references = _linked(spec)
    edges = _named_edges(units, references)
    assert ("A", "REFERENCES", "B") in edges
    assert ("B", "REFERENCES", "A") in edges


def test_external_reference_is_skipped(tmp_path: Path) -> None:
    spec = tmp_path / "external.json"
    spec.write_text(
        '{"openapi": "3.1.0", "paths": {}, "components": {"schemas": {"A": '
        '{"type": "object", "properties": {"ext": {"$ref": "other.json#/X"}, '
        '"loc": {"$ref": "#/components/schemas/B"}}}, "B": {"type": "object", '
        '"properties": {"n": {"type": "string"}}}}}}',
        encoding="utf-8",
    )
    units, references = _linked(spec)
    edges = _named_edges(units, references)
    assert ("A", "REFERENCES", "B") in edges
    assert len(references) == 1


def test_dangling_internal_reference_fails_loud(tmp_path: Path) -> None:
    spec = tmp_path / "dangling.json"
    spec.write_text(
        '{"openapi": "3.1.0", "paths": {}, "components": {"schemas": {"A": '
        '{"type": "object", "properties": {"x": {"$ref": '
        '"#/components/schemas/Nope"}}}}}}',
        encoding="utf-8",
    )
    adapter = SpecAdapter()
    document = adapter.ingest(spec)
    units = adapter.parse(document)
    with pytest.raises(ValueError, match="dangling internal reference"):
        adapter.link(document, units)


def test_link_rejects_mismatched_units() -> None:
    adapter = SpecAdapter()
    document = adapter.ingest(_PAYMENTS)
    with pytest.raises(ValueError, match="do not match the parsed source"):
        adapter.link(document, [])


def test_capabilities_report_reference_kinds() -> None:
    caps = SpecAdapter().capabilities()
    assert {"REFERENCES", "FOREIGN_KEY"} <= caps.reference_kinds


def test_non_unit_internal_target_is_skipped(tmp_path: Path) -> None:
    # A $ref to a real but non-unit node (an OpenAPI parameter) is out of scope
    # and skipped, not treated as dangling; the schema ref still links.
    spec = tmp_path / "params.json"
    spec.write_text(
        '{"openapi": "3.1.0", "paths": {"/x": {"get": {"parameters": '
        '[{"$ref": "#/components/parameters/P"}], "responses": {"200": '
        '{"description": "ok", "content": {"application/json": {"schema": '
        '{"$ref": "#/components/schemas/B"}}}}}}}}, "components": {"parameters": '
        '{"P": {"name": "p", "in": "query"}}, "schemas": {"B": {"type": "object", '
        '"properties": {"n": {"type": "string"}}}}}}',
        encoding="utf-8",
    )
    units, references = _linked(spec)
    assert _named_edges(units, references) == {("GET /x", "REFERENCES", "B")}
