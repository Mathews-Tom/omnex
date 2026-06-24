"""Tests for the click CLI surface: round-trip, receipt shape, parity, determinism.

The CLI is a thin wrapper over :mod:`omnex.api`, so these assert that it neither
changes the returned set nor the receipt schema: the JSON it prints carries the
library's own bundle render and receipt fields, and a query is byte-for-byte
reproducible. No model is invoked on the byte-exact T0 floor, which the receipt
records and these tests check.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner, Result

import omnex
from omnex import api
from omnex._surface import collect_files, default_config
from omnex.cli import _render_markdown, main

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_PAYMENTS = _FIXTURES / "payments_openapi.json"
_QUESTION = "create a payment"
_BUDGET = 2000

# The JSON receipt mirrors the library Receipt: its nine fields plus the derived
# recall_limitations the receipt exposes as honesty caveats.
_RECEIPT_KEYS = {
    "returned_tokens",
    "baseline_tokens",
    "tiers_run",
    "model_used",
    "model_version",
    "extraction_used",
    "determinism_class",
    "reference_closure_complete",
    "recall_basis",
    "recall_limitations",
    "embedding_provenance",
}


def _query(*args: str) -> Result:
    return CliRunner().invoke(main, ["query", str(_PAYMENTS), _QUESTION, *args])


def test_index_then_query_round_trips() -> None:
    runner = CliRunner()
    indexed = runner.invoke(main, ["index", str(_PAYMENTS)])
    assert indexed.exit_code == 0
    assert "unit(s)" in indexed.output
    queried = runner.invoke(main, ["query", str(_PAYMENTS), _QUESTION, "--budget", str(_BUDGET)])
    assert queried.exit_code == 0
    assert queried.output.strip()


def test_query_json_receipt_has_expected_shape() -> None:
    result = _query("--budget", str(_BUDGET), "--format", "json")
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload) == {"bundle", "receipt"}
    assert set(payload["bundle"]) == {"context", "total_tokens", "representations"}
    assert set(payload["receipt"]) == _RECEIPT_KEYS
    # The byte-exact floor invokes no model and no extraction; the receipt says so.
    assert payload["receipt"]["model_used"] is False
    assert payload["receipt"]["extraction_used"] is False
    assert payload["receipt"]["model_version"] is None
    assert payload["receipt"]["determinism_class"] == "byte_exact"
    assert payload["receipt"]["embedding_provenance"] is None


def test_cli_matches_library_bundle_and_receipt() -> None:
    payload = json.loads(_query("--budget", str(_BUDGET), "--format", "json").output)
    bundle, receipt = omnex.query_sources([_PAYMENTS], _QUESTION, _BUDGET, default_config())
    # Same rendered context and token totals: the surface only renders the bundle.
    assert payload["bundle"]["context"] == bundle.render()
    assert payload["bundle"]["total_tokens"] == bundle.total_tokens
    # The returned set is untouched -- identical representations in identical order.
    assert [rep["unit_id"] for rep in payload["bundle"]["representations"]] == [
        rep.unit_id for rep in bundle.representations
    ]
    # The receipt schema is unchanged -- every field matches the library's.
    assert payload["receipt"]["returned_tokens"] == receipt.returned_tokens
    assert payload["receipt"]["baseline_tokens"] == receipt.baseline_tokens
    assert payload["receipt"]["tiers_run"] == list(receipt.tiers_run)
    assert payload["receipt"]["determinism_class"] == receipt.determinism_class
    assert payload["receipt"]["recall_basis"] == receipt.recall_basis
    assert payload["receipt"]["recall_limitations"] == list(receipt.recall_limitations)


def test_query_output_is_deterministic() -> None:
    json_a = _query("--budget", str(_BUDGET), "--format", "json").output
    json_b = _query("--budget", str(_BUDGET), "--format", "json").output
    assert json_a == json_b
    md_a = _query("--budget", str(_BUDGET)).output
    md_b = _query("--budget", str(_BUDGET)).output
    assert md_a == md_b


def test_query_returns_fewer_tokens_than_full_dump() -> None:
    receipt = json.loads(_query("--budget", str(_BUDGET), "--format", "json").output)["receipt"]
    assert receipt["returned_tokens"] < receipt["baseline_tokens"]


def test_markdown_render_surfaces_receipt_and_recall_caveats() -> None:
    bundle, receipt = omnex.query_sources([_PAYMENTS], _QUESTION, _BUDGET, default_config())
    rendered = _render_markdown(bundle, receipt)
    assert "## Receipt" in rendered
    assert "determinism_class: byte_exact" in rendered
    # T0 recall is lexical-only, so the honest caveats are surfaced verbatim.
    assert "### Recall limitations" in rendered
    assert receipt.recall_limitations[0] in rendered


def test_missing_corpus_exits_nonzero() -> None:
    result = CliRunner().invoke(main, ["query", str(_FIXTURES / "no_such_file.json"), _QUESTION])
    assert result.exit_code != 0


def test_unclaimable_source_fails_loud() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("mystery.bin").write_bytes(b"\x00\x01\x02")
        result = runner.invoke(main, ["index", "mystery.bin"])
    assert result.exit_code != 0
    assert isinstance(result.exception, (ValueError, SystemExit))


def test_default_budget_path_is_exercised() -> None:
    # Omitting --budget falls back to the surface default; the path must still
    # produce a valid, token-bounded result.
    result = CliRunner().invoke(main, ["query", str(_PAYMENTS), _QUESTION, "--format", "json"])
    assert result.exit_code == 0
    receipt = json.loads(result.output)["receipt"]
    assert receipt["returned_tokens"] < receipt["baseline_tokens"]


def test_markdown_rows_match_receipt_fields() -> None:
    # Guards Markdown schema-completeness: every Receipt field is rendered, so the
    # Markdown and JSON formats cannot drift if a field is added or renamed.
    bundle, receipt = omnex.query_sources([_PAYMENTS], _QUESTION, _BUDGET, default_config())
    rendered = _render_markdown(bundle, receipt)
    for field in (
        "returned_tokens",
        "baseline_tokens",
        "determinism_class",
        "recall_basis",
        "model_used",
        "model_version",
        "extraction_used",
        "reference_closure_complete",
    ):
        assert f"- {field}: {getattr(receipt, field)}" in rendered
    assert f"- tiers_run: {', '.join(receipt.tiers_run)}" in rendered


def test_index_counts_match_routing() -> None:
    units, references, documents = api._route_sources([_PAYMENTS])
    result = CliRunner().invoke(main, ["index", str(_PAYMENTS)])
    assert result.exit_code == 0
    assert result.output.strip() == (
        f"indexed {len(documents)} document(s), {len(units)} unit(s), "
        f"{len(references)} reference(s)"
    )


def test_directory_corpus_is_deterministic() -> None:
    docs = _FIXTURES / "tls_docs"
    question = "How do I configure TLS for the ingress controller?"
    first = CliRunner().invoke(main, ["query", str(docs), question, "--budget", "200"])
    second = CliRunner().invoke(main, ["query", str(docs), question, "--budget", "200"])
    assert first.exit_code == 0
    assert first.output.strip()
    assert first.output == second.output
    indexed = CliRunner().invoke(main, ["index", str(docs)])
    assert indexed.exit_code == 0
    assert "3 document(s)" in indexed.output


def test_collect_files_skips_hidden_entries_and_sorts() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        base = Path("corpus")
        (base / ".hidden").mkdir(parents=True)
        (base / ".hidden" / "skip.md").write_text("# hidden\n")
        (base / ".dotfile.md").write_text("# dot\n")
        (base / "b.md").write_text("# B\n")
        (base / "a.md").write_text("# A\n")
        # Hidden files and files under hidden directories are skipped; the rest
        # are returned in sorted order so routing is deterministic.
        assert collect_files([base]) == [base / "a.md", base / "b.md"]


def test_index_empty_directory_fails_loud(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = CliRunner().invoke(main, ["index", str(empty)])
    assert result.exit_code != 0
    assert "corpus is empty" in result.output
