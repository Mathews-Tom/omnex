"""Markdown / reStructuredText modality adapter.

Detects, ingests, and parses prose documents into the modality-agnostic IR. The
adapter is deterministic and model-free: it never calls a language model on any
path. It builds the document's heading tree into ``SECTION`` units whose children
are ``PARAGRAPH`` / ``TABLE`` / ``FIGURE_CAPTION`` units, sets each unit's
breadcrumb to its section path, and marks verbatim blocks (code fences, tables)
``protect=True`` so the packer can never compress or elide them. A long section
body is split on natural boundaries within a token budget (see
:mod:`omnex.adapters._split`), because a unit is the packing atom and must fit.

``raw_token_count`` is measured with the ``tiktoken`` ``cl100k_base`` tokenizer,
a deterministic offline encoder loaded once and reused (a tokenizer, not a model
lane). Edge recovery (``CONTAINS`` / ``SIBLING`` / ``CROSS_REF`` / ``CITES``)
lives in :meth:`ProseAdapter.link`.

Scope is Markdown (``.md`` and kin) and reStructuredText (``.rst``); a structured
plain-text file with a heading is also claimed. Adapters depend on the kernel and
IR, never the reverse.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from omnex.adapters._split import split_on_budget
from omnex.adapters.base import AdapterCapabilities
from omnex.ir.types import (
    Document,
    Reference,
    Span,
    Unit,
    UnitKind,
    compute_content_hash,
    make_document_id,
    make_unit_id,
    normalize_content,
    read_source,
)
from omnex.kernel.packer import count_tokens

if TYPE_CHECKING:
    import tiktoken

# Identifier of the deterministic offline tiktoken encoding used for raw counts
# and token-aware splitting. This is a tokenizer, not a model lane.
_ENCODING = "cl100k_base"

# Token budget a single section-body unit may reach before it is split on natural
# boundaries. A unit is the packing atom, so an over-budget body is divided into
# several units rather than packed whole.
_SECTION_TOKEN_BUDGET = 512

# Recognized prose file extensions (lowercased).
_MARKDOWN_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown", ".mdown", ".mkd", ".mdwn"})
_REST_SUFFIXES: frozenset[str] = frozenset({".rst", ".rest"})
_PROSE_SUFFIXES: frozenset[str] = _MARKDOWN_SUFFIXES | _REST_SUFFIXES

Flavor = Literal["markdown", "rest"]

# --- Markdown line grammar (a focused, deterministic block subset) -----------
_ATX_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+.*?)?(?:[ \t]+#+)?[ \t]*$")
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_THEMATIC_BREAK_RE = re.compile(r"^ {0,3}([-*_])[ \t]*(?:\1[ \t]*){2,}$")
_TABLE_DELIM_RE = re.compile(r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$")
_IMAGE_ONLY_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)$")

# --- reStructuredText line grammar -------------------------------------------
_REST_UNDERLINE_RE = re.compile(r"""^([=\-~^"'#*+.:_`])\1{1,}[ \t]*$""")
_REST_GRID_RE = re.compile(r"^ {0,3}\+[-=+]+\+[ \t]*$")
_REST_SIMPLE_TABLE_RE = re.compile(r"^ {0,3}=+( +=+)+[ \t]*$")


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    """Load the deterministic offline ``tiktoken`` encoder once and cache it.

    Mirrors the spec adapter's tokenizer handling: ``cl100k_base`` is loaded from
    the local tiktoken cache and reused, so token counts are deterministic and
    the encoder is a tokenizer rather than a retrieval model lane.
    """
    import tiktoken

    return tiktoken.get_encoding(_ENCODING)


def _flavor(uri: str) -> Flavor:
    """Classify a source as reST or Markdown by its file extension."""
    return "rest" if Path(uri).suffix.lower() in _REST_SUFFIXES else "markdown"


def _has_prose_structure(text: str) -> bool:
    """True when extension-less text still looks like a structured prose document.

    Used only as the secondary detection path: a recognized prose extension claims
    directly, but a plain-text file carrying a Markdown ATX heading or a reST
    section underline is claimed too, so structure -- not just suffix -- routes.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _ATX_RE.match(line) and line.lstrip().startswith("#"):
            return True
        if index > 0 and _REST_UNDERLINE_RE.match(line) and lines[index - 1].strip():
            return True
    return False


# ---------------------------------------------------------------------------
# Block scanning
#
# Both flavors are reduced to the same ordered list of typed blocks with exact
# character spans, so heading-tree assembly and unit construction are shared.
# ---------------------------------------------------------------------------

BlockKind = Literal["HEADING", "PARAGRAPH", "CODE", "TABLE", "FIGURE"]


@dataclass(frozen=True, slots=True)
class _Block:
    """One source block with its character span (and heading level/title)."""

    kind: BlockKind
    level: int
    start: int
    end: int
    title: str | None


def _scan_lines(source: str) -> list[tuple[int, int, int]]:
    """Return ``(start, content_end, raw_end)`` for each line of ``source``."""
    lines: list[tuple[int, int, int]] = []
    position = 0
    length = len(source)
    while position < length:
        newline = source.find("\n", position)
        if newline == -1:
            lines.append((position, length, length))
            break
        lines.append((position, newline, newline + 1))
        position = newline + 1
    return lines


def _trim_span(source: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``(start, end)`` past surrounding whitespace."""
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return start, end


def _atx_title(line: str) -> str:
    """Extract the heading text from an ATX heading line."""
    stripped = line.strip().lstrip("#").strip()
    return re.sub(r"[ \t]+#+$", "", stripped).strip()


def _indent(line: str) -> int:
    """Return the count of leading spaces (tabs expand to one) on ``line``."""
    return len(line) - len(line.lstrip(" \t"))


def _parse_markdown(source: str) -> list[_Block]:
    """Scan Markdown ``source`` into ordered typed blocks with exact spans."""
    lines = _scan_lines(source)
    blocks: list[_Block] = []
    index = 0
    while index < len(lines):
        start, content_end, raw_end = lines[index]
        line = source[start:content_end]
        if not line.strip():
            index += 1
            continue
        if _THEMATIC_BREAK_RE.match(line):
            # A thematic break (horizontal rule) carries no content: emit no unit.
            # A paragraph followed by such a line is handled as a setext heading
            # below, so reaching one here means it stands alone.
            index += 1
            continue
        fence = _FENCE_OPEN_RE.match(line)
        if fence:
            marker = fence.group(1)
            block_end = raw_end
            index += 1
            while index < len(lines):
                inner_start, inner_end, inner_raw = lines[index]
                block_end = inner_raw
                index += 1
                if source[inner_start:inner_end].strip().startswith(marker[0] * len(marker)):
                    break
            blocks.append(_Block("CODE", 0, start, block_end, None))
            continue
        if _ATX_RE.match(line) and line.lstrip().startswith("#"):
            level = len(line.strip()) - len(line.strip().lstrip("#"))
            blocks.append(_Block("HEADING", level, start, content_end, _atx_title(line)))
            index += 1
            continue
        if (
            "|" in line
            and index + 1 < len(lines)
            and _TABLE_DELIM_RE.match(source[lines[index + 1][0] : lines[index + 1][1]])
        ):
            block_end = lines[index + 1][2]
            index += 2
            while index < len(lines):
                row_start, row_end, row_raw = lines[index]
                row = source[row_start:row_end]
                if "|" not in row or not row.strip():
                    break
                block_end = row_raw
                index += 1
            blocks.append(_Block("TABLE", 0, start, block_end, None))
            continue
        para_end = content_end
        index += 1
        setext_level = 0
        while index < len(lines):
            next_start, next_end, _ = lines[index]
            nxt = source[next_start:next_end]
            if not nxt.strip() or _ATX_RE.match(nxt) or _FENCE_OPEN_RE.match(nxt):
                break
            if _SETEXT_RE.match(nxt):
                # A paragraph underlined by = or - is a setext heading; it wins
                # over a thematic break per CommonMark.
                setext_level = 1 if nxt.strip()[0] == "=" else 2
                break
            if _THEMATIC_BREAK_RE.match(nxt):
                break
            para_end = next_end
            index += 1
        block_text = source[start:para_end]
        if setext_level and "|" not in block_text:
            blocks.append(_Block("HEADING", setext_level, start, para_end, block_text.strip()))
            index += 1
            continue
        text = block_text.strip()
        kind: BlockKind = "FIGURE" if _IMAGE_ONLY_RE.match(text) else "PARAGRAPH"
        blocks.append(_Block(kind, 0, start, para_end, None))
    return blocks


def _parse_rest(source: str) -> list[_Block]:
    """Scan reStructuredText ``source`` into ordered typed blocks with exact spans."""
    lines = _scan_lines(source)
    blocks: list[_Block] = []
    level_for_char: dict[str, int] = {}
    index = 0
    while index < len(lines):
        start, content_end, raw_end = lines[index]
        line = source[start:content_end]
        if not line.strip():
            index += 1
            continue
        if index + 1 < len(lines):
            under = source[lines[index + 1][0] : lines[index + 1][1]]
            is_section = (
                _REST_UNDERLINE_RE.match(under)
                and not _REST_UNDERLINE_RE.match(line)
                and len(under.strip()) >= len(line.strip())
            )
            if is_section:
                char = under.strip()[0]
                level = level_for_char.setdefault(char, len(level_for_char) + 1)
                blocks.append(_Block("HEADING", level, start, content_end, line.strip()))
                index += 2
                continue
        if _REST_GRID_RE.match(line) or _REST_SIMPLE_TABLE_RE.match(line):
            block_end = raw_end
            index += 1
            while index < len(lines):
                row_start, row_end, row_raw = lines[index]
                if not source[row_start:row_end].strip():
                    break
                block_end = row_raw
                index += 1
            blocks.append(_Block("TABLE", 0, start, block_end, None))
            continue
        para_end = content_end
        ends_with_literal = line.rstrip().endswith("::")
        index += 1
        while index < len(lines):
            next_start, next_end, _ = lines[index]
            nxt = source[next_start:next_end]
            if not nxt.strip():
                break
            if index + 1 < len(lines) and _REST_UNDERLINE_RE.match(
                source[lines[index + 1][0] : lines[index + 1][1]]
            ):
                break
            para_end = next_end
            ends_with_literal = nxt.rstrip().endswith("::")
            index += 1
        blocks.append(_Block("PARAGRAPH", 0, start, para_end, None))
        if ends_with_literal:
            index = _consume_rest_literal(source, lines, index, blocks)
    return blocks


def _consume_rest_literal(
    source: str, lines: list[tuple[int, int, int]], index: int, blocks: list[_Block]
) -> int:
    """Consume an indented reST literal block after a ``::`` paragraph, if present."""
    while index < len(lines) and not source[lines[index][0] : lines[index][1]].strip():
        index += 1
    if index >= len(lines):
        return index
    base_indent = _indent(source[lines[index][0] : lines[index][1]])
    if base_indent == 0:
        return index
    literal_start = lines[index][0]
    literal_end = lines[index][2]
    index += 1
    while index < len(lines):
        row_start, row_end, row_raw = lines[index]
        row = source[row_start:row_end]
        if row.strip() and _indent(row) < base_indent:
            break
        literal_end = row_raw
        index += 1
    blocks.append(_Block("CODE", 0, literal_start, literal_end, None))
    return index


# ---------------------------------------------------------------------------
# Heading-tree assembly
# ---------------------------------------------------------------------------

# Block kind -> (IR unit kind, protected). Code fences and tables are verbatim:
# they are protected so the packer can never compress or elide them.
_BLOCK_UNIT: dict[BlockKind, tuple[UnitKind, bool]] = {
    "PARAGRAPH": ("PARAGRAPH", False),
    "CODE": ("PARAGRAPH", True),
    "TABLE": ("TABLE", True),
    "FIGURE": ("FIGURE_CAPTION", False),
}


def _build_unit(
    document: Document,
    source: str,
    kind: UnitKind,
    start: int,
    end: int,
    title: str | None,
    breadcrumb: tuple[str, ...],
    protect: bool,
) -> Unit:
    """Construct one IR unit over the trimmed ``(start, end)`` source span."""
    span_start, span_end = _trim_span(source, start, end)
    text = source[span_start:span_end]
    return Unit(
        id=make_unit_id(document_id=document.id, span=Span(span_start, span_end), text=text),
        document_id=document.id,
        span=Span(span_start, span_end),
        text=text,
        token_count=count_tokens(text),
        title=title,
        breadcrumb=breadcrumb,
        kind=kind,
        summary=None,
        protect=protect,
    )


def _assemble(document: Document, source: str, blocks: Sequence[_Block]) -> list[Unit]:
    """Build the heading tree from ``blocks`` into breadcrumb-stamped units."""
    units: list[Unit] = []
    stack: list[tuple[int, str]] = []
    encode = _encoder().encode
    for block in blocks:
        if block.kind == "HEADING":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            breadcrumb = tuple(title for _, title in stack)
            units.append(
                _build_unit(
                    document,
                    source,
                    "SECTION",
                    block.start,
                    block.end,
                    block.title,
                    breadcrumb,
                    False,
                )
            )
            stack.append((block.level, block.title or ""))
            continue
        unit_kind, protect = _BLOCK_UNIT[block.kind]
        breadcrumb = tuple(title for _, title in stack)
        text = source[block.start : block.end]
        if not protect and len(encode(text)) > _SECTION_TOKEN_BUDGET:
            for piece_start, piece_end in split_on_budget(text, _SECTION_TOKEN_BUDGET, encode):
                units.append(
                    _build_unit(
                        document,
                        source,
                        unit_kind,
                        block.start + piece_start,
                        block.start + piece_end,
                        None,
                        breadcrumb,
                        protect,
                    )
                )
        else:
            units.append(
                _build_unit(
                    document, source, unit_kind, block.start, block.end, None, breadcrumb, protect
                )
            )
    return units


def _blocks(source: str, flavor: Flavor) -> list[_Block]:
    """Scan ``source`` into typed blocks for the given flavor."""
    return _parse_rest(source) if flavor == "rest" else _parse_markdown(source)


class ProseAdapter:
    """Deterministic Markdown / reStructuredText adapter emitting the IR."""

    __slots__ = ()

    def claims(self, source: Path) -> bool:
        """Return True for a prose source, by extension or heading structure."""
        if source.suffix.lower() in _PROSE_SUFFIXES:
            return True
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        return _has_prose_structure(text)

    def ingest(self, source: Path) -> Document:
        """Establish document identity, content hash, and raw token count."""
        text = source.read_text(encoding="utf-8")
        content_hash = compute_content_hash(text)
        raw_token_count = len(_encoder().encode(normalize_content(text)))
        uri = str(source)
        return Document(
            id=make_document_id(uri=uri, content_hash=content_hash),
            uri=uri,
            modality="prose",
            content_hash=content_hash,
            raw_token_count=raw_token_count,
        )

    def parse(self, document: Document) -> list[Unit]:
        """Emit the retrievable units of a prose document.

        Deterministic and model-free. The heading tree becomes ``SECTION`` units
        whose direct children are ``PARAGRAPH`` / ``TABLE`` / ``FIGURE_CAPTION``
        units; each unit's breadcrumb is its section path. Code fences and tables
        are ``protect=True``; a long body is split on natural boundaries within
        the token budget. Spans are character offsets into the normalized source.
        """
        source = read_source(document)
        blocks = _blocks(source, _flavor(document.uri))
        return _assemble(document, source, blocks)

    def link(self, document: Document, units: Sequence[Unit]) -> list[Reference]:
        """Recover the typed edges among ``units``.

        The structural tree (``CONTAINS`` / ``SIBLING``) and the cross-reference
        edges (``CROSS_REF`` / ``CITES``) are recovered by the linking pass added
        next in this stack; this parse slice emits units only.
        """
        return []

    def capabilities(self) -> AdapterCapabilities:
        """Report emittable kinds, determinism, and model-extraction opt-in."""
        return AdapterCapabilities(
            unit_kinds=frozenset({"SECTION", "PARAGRAPH", "TABLE", "FIGURE_CAPTION"}),
            reference_kinds=frozenset({"CONTAINS", "SIBLING", "CROSS_REF", "CITES"}),
            deterministic_parse=True,
            model_extraction_opt_in=False,
        )
