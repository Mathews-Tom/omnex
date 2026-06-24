"""Tests for the prose adapter: detection, ingest, and deterministic parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnex.adapters.base import ModalityAdapter
from omnex.adapters.prose import _SECTION_TOKEN_BUDGET, ProseAdapter, _encoder
from omnex.ir.types import Document, Unit, read_source

_MARKDOWN = """\
# Ingress

The ingress controller routes external traffic to backend services.

## TLS secrets

Store the certificate and key in a Kubernetes secret and reference it.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ingress-tls
type: kubernetes.io/tls
```

### Verification

Confirm the listener is healthy.

| Field | Meaning |
| --- | --- |
| crt | certificate |
| key | private key |

![architecture](arch.png)
"""

_REST = """\
Ingress
=======

The ingress controller routes traffic.

TLS secrets
-----------

Store the certificate like so::

    apiVersion: v1
    kind: Secret
    type: kubernetes.io/tls

Done.
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    source = tmp_path / name
    source.write_text(text, encoding="utf-8")
    return source


def _parsed(tmp_path: Path, name: str, text: str) -> tuple[ProseAdapter, Document, list[Unit]]:
    adapter = ProseAdapter()
    document = adapter.ingest(_write(tmp_path, name, text))
    return adapter, document, adapter.parse(document)


def test_adapter_satisfies_protocol() -> None:
    adapter: ModalityAdapter = ProseAdapter()
    assert isinstance(adapter, ModalityAdapter)


def test_claims_detects_markdown_and_rest(tmp_path: Path) -> None:
    assert ProseAdapter().claims(_write(tmp_path, "doc.md", _MARKDOWN)) is True
    assert ProseAdapter().claims(_write(tmp_path, "doc.rst", _REST)) is True


def test_claims_detects_structured_plain_text(tmp_path: Path) -> None:
    structured = _write(tmp_path, "notes.txt", "# Title\n\nbody text here\n")
    assert ProseAdapter().claims(structured) is True


def test_claims_rejects_unstructured_plain_text(tmp_path: Path) -> None:
    plain = _write(tmp_path, "plain.txt", "just one line of words, no heading at all\n")
    assert ProseAdapter().claims(plain) is False


def test_claims_rejects_source_and_config_files_with_hash_lines(tmp_path: Path) -> None:
    # A source or config file whose extension is not prose is never sniffed, so a
    # bare "#" comment or a divider line does not get it mis-claimed as prose; it
    # falls through to fail-loud routing instead.
    py = _write(tmp_path, "conf.py", "# configuration\nproject = 'x'\n")
    yaml = _write(tmp_path, "mkdocs.yml", "site_name: Docs\n# ----\n")
    assert ProseAdapter().claims(py) is False
    assert ProseAdapter().claims(yaml) is False


def test_ingest_sets_identity_hash_and_raw_token_count(tmp_path: Path) -> None:
    _, document, _ = _parsed(tmp_path, "doc.md", _MARKDOWN)
    assert document.modality == "prose"
    assert document.content_hash.startswith("sha256:")
    assert document.id.startswith("doc:")
    assert document.raw_token_count > 0


def test_ingest_is_deterministic(tmp_path: Path) -> None:
    source = _write(tmp_path, "doc.md", _MARKDOWN)
    adapter = ProseAdapter()
    assert adapter.ingest(source) == adapter.ingest(source)


def test_heading_tree_maps_to_section_units(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "doc.md", _MARKDOWN)
    sections = [unit for unit in units if unit.kind == "SECTION"]
    titles = [unit.title for unit in sections]
    assert titles == ["Ingress", "TLS secrets", "Verification"]
    breadcrumbs = {unit.title: unit.breadcrumb for unit in sections}
    # The breadcrumb of a section is the path of its ancestors (not itself).
    assert breadcrumbs["Ingress"] == ()
    assert breadcrumbs["TLS secrets"] == ("Ingress",)
    assert breadcrumbs["Verification"] == ("Ingress", "TLS secrets")


def test_children_carry_the_section_path_breadcrumb(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "doc.md", _MARKDOWN)
    figure = next(unit for unit in units if unit.kind == "FIGURE_CAPTION")
    table = next(unit for unit in units if unit.kind == "TABLE")
    # Both live under the deepest open section, so the breadcrumb is the full path.
    assert figure.breadcrumb == ("Ingress", "TLS secrets", "Verification")
    assert table.breadcrumb == ("Ingress", "TLS secrets", "Verification")


def test_emits_paragraph_table_and_figure_children(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "doc.md", _MARKDOWN)
    kinds = {unit.kind for unit in units}
    assert {"SECTION", "PARAGRAPH", "TABLE", "FIGURE_CAPTION"} <= kinds


def test_code_blocks_and_tables_are_protected(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "doc.md", _MARKDOWN)
    code = next(unit for unit in units if unit.text.startswith("```"))
    table = next(unit for unit in units if unit.kind == "TABLE")
    assert code.protect is True
    assert table.protect is True
    # Ordinary prose stays compressible.
    prose = next(unit for unit in units if unit.text.startswith("The ingress"))
    assert prose.protect is False


def test_spans_round_trip_from_source(tmp_path: Path) -> None:
    _, document, units = _parsed(tmp_path, "doc.md", _MARKDOWN)
    source = read_source(document)
    for unit in units:
        assert source[unit.span.start : unit.span.end] == unit.text


def test_parse_is_byte_identical_on_repeat(tmp_path: Path) -> None:
    adapter, document, units = _parsed(tmp_path, "doc.md", _MARKDOWN)
    assert adapter.parse(document) == units


def test_unit_ids_are_unique(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "doc.md", _MARKDOWN)
    assert len({unit.id for unit in units}) == len(units)


def test_long_section_splits_on_boundaries_within_budget(tmp_path: Path) -> None:
    sentence = "The controller terminates TLS using a certificate stored in a secret object. "
    body = sentence * 40
    encode = _encoder().encode
    assert len(encode(body)) > _SECTION_TOKEN_BUDGET, "fixture must exceed the budget"
    _, _, units = _parsed(tmp_path, "long.md", f"# Big\n\n{body}\n")
    pieces = [unit for unit in units if unit.kind == "PARAGRAPH"]
    assert len(pieces) > 1, "an over-budget body must split into several units"
    for piece in pieces:
        assert len(encode(piece.text)) <= _SECTION_TOKEN_BUDGET
        assert piece.text == piece.text.strip()
    # No word is split or dropped: the pieces' words reconstruct the body in order.
    rejoined = " ".join(word for piece in pieces for word in piece.text.split())
    assert rejoined == " ".join(body.split())


def test_single_over_budget_word_is_emitted_whole(tmp_path: Path) -> None:
    encode = _encoder().encode
    word = "x" * 6000
    assert len(encode(word)) > _SECTION_TOKEN_BUDGET, "fixture must exceed the budget"
    _, _, units = _parsed(tmp_path, "word.md", f"# W\n\n{word}\n")
    pieces = [unit for unit in units if unit.kind == "PARAGRAPH"]
    # A single contiguous run wider than the budget is never cut mid-token.
    assert len(pieces) == 1
    assert pieces[0].text == word


def test_thematic_break_emits_no_unit(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "rule.md", "# H\n\nfirst\n\n---\n\nsecond\n")
    assert not any(unit.text.strip() in {"---", "***", "___"} for unit in units)
    paragraphs = [unit.text for unit in units if unit.kind == "PARAGRAPH"]
    assert paragraphs == ["first", "second"]


def test_setext_headings_become_sections(tmp_path: Path) -> None:
    # Single-line and multi-line setext, plus a paragraph underlined by '-'
    # (a setext heading per CommonMark, not a thematic break).
    _, _, single = _parsed(tmp_path, "s1.md", "Title One\n=========\n\nbody\n")
    assert [u.title for u in single if u.kind == "SECTION"] == ["Title One"]
    _, _, multi = _parsed(tmp_path, "s2.md", "Line A\nLine B\n======\n\nbody\n")
    sections = [u for u in multi if u.kind == "SECTION"]
    assert [u.title for u in sections] == ["Line A\nLine B"]
    assert not any(u.text.strip() == "======" for u in multi)
    _, _, dash = _parsed(tmp_path, "s3.md", "Subhead\n---\n\nbody\n")
    assert [u.title for u in dash if u.kind == "SECTION"] == ["Subhead"]


def test_content_before_first_heading_has_empty_breadcrumb(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "pre.md", "intro before any heading\n\n# First\n\nbody\n")
    assert units[0].kind == "PARAGRAPH"
    assert units[0].breadcrumb == ()


def test_unterminated_fence_runs_to_eof_as_one_protected_unit(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "open.md", "# C\n\n```yaml\nkey: value\nstill open\n")
    protected = [unit for unit in units if unit.protect]
    assert len(protected) == 1
    assert protected[0].text.startswith("```yaml")


def test_protected_block_is_never_split(tmp_path: Path) -> None:
    line = "tls.crt: a-very-long-base64-looking-value-repeated-many-times "
    fenced = "```yaml\n" + (line * 60) + "\n```"
    encode = _encoder().encode
    inner = line * 60
    assert len(encode(inner)) > _SECTION_TOKEN_BUDGET, "fixture must exceed the budget"
    _, _, units = _parsed(tmp_path, "fence.md", f"# Cfg\n\n{fenced}\n")
    code = [unit for unit in units if unit.protect]
    assert len(code) == 1, "a protected fence stays one unit even over budget"


def test_rest_headings_map_to_section_units(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "doc.rst", _REST)
    sections = [unit for unit in units if unit.kind == "SECTION"]
    assert [unit.title for unit in sections] == ["Ingress", "TLS secrets"]
    assert sections[0].breadcrumb == ()
    assert sections[1].breadcrumb == ("Ingress",)


def test_rest_literal_block_is_protected(tmp_path: Path) -> None:
    _, _, units = _parsed(tmp_path, "doc.rst", _REST)
    literal = next(unit for unit in units if "apiVersion" in unit.text)
    assert literal.protect is True


def test_capabilities_report_prose_kinds() -> None:
    caps = ProseAdapter().capabilities()
    assert {"SECTION", "PARAGRAPH", "TABLE", "FIGURE_CAPTION"} <= caps.unit_kinds
    assert {"CONTAINS", "SIBLING", "CROSS_REF", "CITES"} <= caps.reference_kinds
    assert caps.deterministic_parse is True
    assert caps.model_extraction_opt_in is False


def test_parse_rejects_source_changed_since_ingest(tmp_path: Path) -> None:
    source = _write(tmp_path, "doc.md", _MARKDOWN)
    adapter = ProseAdapter()
    document = adapter.ingest(source)
    source.write_text("# Different\n\nnew body\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since ingest"):
        adapter.parse(document)
