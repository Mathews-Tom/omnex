"""Tests for the spec adapter: detection, ingest, and deterministic parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnex.adapters.base import ModalityAdapter
from omnex.adapters.spec import SpecAdapter
from omnex.ir.types import Document, normalize_content

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_PAYMENTS = _FIXTURES / "payments_openapi.json"


def _ingested() -> tuple[SpecAdapter, Document, str]:
    adapter = SpecAdapter()
    document = adapter.ingest(_PAYMENTS)
    source = normalize_content(_PAYMENTS.read_text(encoding="utf-8"))
    return adapter, document, source


def test_adapter_satisfies_protocol() -> None:
    adapter: ModalityAdapter = SpecAdapter()
    assert isinstance(adapter, ModalityAdapter)


def test_claims_detects_openapi() -> None:
    assert SpecAdapter().claims(_PAYMENTS) is True


def test_claims_rejects_non_spec_json(tmp_path: Path) -> None:
    plain = tmp_path / "plain.json"
    plain.write_text('{"hello": "world"}', encoding="utf-8")
    assert SpecAdapter().claims(plain) is False


def test_claims_rejects_non_json(tmp_path: Path) -> None:
    not_json = tmp_path / "note.txt"
    not_json.write_text("this is not json", encoding="utf-8")
    assert SpecAdapter().claims(not_json) is False


def test_claims_detects_jsonschema(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(
        '{"$schema": "https://json-schema.org/draft/2020-12/schema", "$defs": {}}',
        encoding="utf-8",
    )
    assert SpecAdapter().claims(schema) is True


def test_ingest_sets_identity_hash_and_raw_token_count() -> None:
    _, document, _ = _ingested()
    assert document.modality == "spec"
    assert document.content_hash.startswith("sha256:")
    assert document.id.startswith("doc:")
    assert document.raw_token_count > 0


def test_ingest_is_deterministic() -> None:
    adapter = SpecAdapter()
    assert adapter.ingest(_PAYMENTS) == adapter.ingest(_PAYMENTS)


def test_parse_emits_operations_schemas_and_fields() -> None:
    adapter, document, _ = _ingested()
    units = adapter.parse(document)
    by_kind: dict[str, set[str | None]] = {}
    for unit in units:
        by_kind.setdefault(unit.kind, set()).add(unit.title)
    assert "POST /payments" in by_kind["OPERATION"]
    assert {"PaymentRequest", "Payment", "Money", "Customer", "Address"} <= by_kind["SCHEMA"]
    # Inline scalar properties become FIELD units.
    assert {"amount", "currency"} <= by_kind["FIELD"]


def test_reference_property_is_not_a_field() -> None:
    # PaymentRequest.amount / .customer are bare $ref properties: they are schema
    # edges (recovered in link), never FIELD units. Money still has a real
    # inline 'amount' field, so the FIELD must belong to a Money breadcrumb.
    adapter, document, _ = _ingested()
    units = adapter.parse(document)
    amount_fields = [u for u in units if u.kind == "FIELD" and u.title == "amount"]
    assert amount_fields, "Money.amount inline field must be emitted"
    assert all(field.breadcrumb[-2] == "Money" for field in amount_fields)


def test_unit_ids_are_stable_and_pointer_distinct() -> None:
    adapter, document, _ = _ingested()
    first = adapter.parse(document)
    second = adapter.parse(document)
    assert [u.id for u in first] == [u.id for u in second]
    # Distinct constructs (distinct pointers) get distinct ids.
    assert len({u.id for u in first}) == len(first)


def test_parse_is_byte_identical_on_repeat() -> None:
    adapter, document, _ = _ingested()
    assert adapter.parse(document) == adapter.parse(document)


def test_spans_recover_source_text() -> None:
    adapter, document, source = _ingested()
    units = adapter.parse(document)
    for unit in units:
        # The span captures a complete JSON value: it parses, and the addressed
        # bytes are exactly the unit text, proving the boundaries are well-formed
        # and aligned to a real construct rather than any arbitrary slice.
        assert source[unit.span.start : unit.span.end] == unit.text
        assert json.loads(unit.text) is not None
    money = next(u for u in units if u.kind == "SCHEMA" and u.title == "Money")
    assert set(json.loads(money.text)["properties"]) == {"amount", "currency"}
    operation = next(u for u in units if u.title == "POST /payments")
    assert "responses" in json.loads(operation.text)


def test_operations_and_schemas_are_protected() -> None:
    adapter, document, _ = _ingested()
    for unit in adapter.parse(document):
        if unit.kind in ("OPERATION", "SCHEMA"):
            assert unit.protect is True


def test_capabilities_report_spec_kinds() -> None:
    caps = SpecAdapter().capabilities()
    assert {"OPERATION", "SCHEMA", "FIELD"} <= caps.unit_kinds
    assert caps.deterministic_parse is True
    assert caps.model_extraction_opt_in is False


def test_parse_rejects_source_changed_since_ingest(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(_PAYMENTS.read_text(encoding="utf-8"), encoding="utf-8")
    adapter = SpecAdapter()
    document = adapter.ingest(spec)
    spec.write_text('{"openapi": "3.1.0", "paths": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed since ingest"):
        adapter.parse(document)


def test_claims_rejects_swagger_2() -> None:
    # Swagger 2.0 is out of scope: it must not be claimed (and then half-parsed).
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write('{"swagger": "2.0", "definitions": {"X": {"type": "object"}}}')
        swagger = Path(handle.name)
    assert SpecAdapter().claims(swagger) is False


def test_parse_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    # A duplicate key would yield two units colliding on one pointer-derived id;
    # the parser fails loud instead.
    spec = tmp_path / "dup.json"
    spec.write_text(
        '{"openapi": "3.1.0", "paths": {}, "components": {"schemas": '
        '{"X": {"type": "object", "properties": {"a": {"type": "string"}, '
        '"a": {"type": "number"}}}}}}',
        encoding="utf-8",
    )
    adapter = SpecAdapter()
    document = adapter.ingest(spec)
    with pytest.raises(ValueError, match="duplicate object key"):
        adapter.parse(document)


def test_ref_with_sibling_keys_stays_a_field(tmp_path: Path) -> None:
    # A $ref carrying sibling content keeps that inline content as a FIELD; only a
    # sole-member $ref is a pure schema edge.
    spec = tmp_path / "siblings.json"
    spec.write_text(
        '{"openapi": "3.1.0", "paths": {}, "components": {"schemas": {"Wrap": '
        '{"type": "object", "properties": {"annotated": {"$ref": '
        '"#/components/schemas/Wrap", "description": "kept"}}}}}}',
        encoding="utf-8",
    )
    adapter = SpecAdapter()
    document = adapter.ingest(spec)
    fields = [u for u in adapter.parse(document) if u.kind == "FIELD"]
    assert any(u.title == "annotated" for u in fields)
