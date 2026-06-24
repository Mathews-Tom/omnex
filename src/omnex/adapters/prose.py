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
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from omnex.adapters._split import split_on_budget
from omnex.adapters.base import AdapterCapabilities
from omnex.ir.types import (
    Document,
    Reference,
    ReferenceKind,
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
# Extensions whose contents are sniffed for prose structure. Only genuine
# plain-text files (and extension-less files) are sniffed, so a source or config
# file that merely contains a ``#`` comment or a ``----`` divider is not claimed
# as prose -- it falls through to fail loud, as the fail-loud routing requires.
_PLAINTEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".text"})

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


def _claims_prose(source: Path) -> bool:
    """True if ``source`` is a prose document, by extension or heading structure.

    A recognized prose extension claims directly. A genuine plain-text or
    extension-less file is sniffed for prose structure; any other extension (a
    source or config file) is rejected so it falls through to fail-loud routing
    rather than being mis-parsed as prose. This is the single routing predicate,
    shared by the adapter and inter-document neighbor resolution.
    """
    suffix = source.suffix.lower()
    if suffix in _PROSE_SUFFIXES:
        return True
    if suffix and suffix not in _PLAINTEXT_SUFFIXES:
        return False
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return _has_prose_structure(text)


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


@dataclass(frozen=True, slots=True)
class _Node:
    """An assembled unit with its tree position: enclosing section and level."""

    unit: Unit
    parent: int | None
    level: int
    is_section: bool


def _build_nodes(document: Document, source: str, blocks: Sequence[_Block]) -> list[_Node]:
    """Assemble ``blocks`` into tree nodes: units plus parent/section structure.

    The heading stack tracks open sections; each block becomes a node whose
    ``parent`` is the index of its enclosing section node (None at the top
    level). Breadcrumbs are the section path, identical to what parsing emits. A
    long unprotected body is split into several sibling content nodes.
    """
    nodes: list[_Node] = []
    stack: list[tuple[int, int, str]] = []  # (heading level, node index, title)
    encode = _encoder().encode
    for block in blocks:
        if block.kind == "HEADING":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            breadcrumb = tuple(title for _, _, title in stack)
            parent = stack[-1][1] if stack else None
            unit = _build_unit(
                document, source, "SECTION", block.start, block.end, block.title, breadcrumb, False
            )
            nodes.append(_Node(unit, parent, block.level, True))
            stack.append((block.level, len(nodes) - 1, block.title or ""))
            continue
        unit_kind, protect = _BLOCK_UNIT[block.kind]
        breadcrumb = tuple(title for _, _, title in stack)
        parent = stack[-1][1] if stack else None
        text = source[block.start : block.end]
        if not protect and len(encode(text)) > _SECTION_TOKEN_BUDGET:
            spans = [
                (block.start + piece_start, block.start + piece_end)
                for piece_start, piece_end in split_on_budget(text, _SECTION_TOKEN_BUDGET, encode)
            ]
        else:
            spans = [(block.start, block.end)]
        for span_start, span_end in spans:
            unit = _build_unit(
                document, source, unit_kind, span_start, span_end, None, breadcrumb, protect
            )
            nodes.append(_Node(unit, parent, 0, False))
    return nodes


def _assemble(document: Document, source: str, blocks: Sequence[_Block]) -> list[Unit]:
    """Build the heading tree from ``blocks`` into breadcrumb-stamped units."""
    return [node.unit for node in _build_nodes(document, source, blocks)]


def _blocks(source: str, flavor: Flavor) -> list[_Block]:
    """Scan ``source`` into typed blocks for the given flavor."""
    return _parse_rest(source) if flavor == "rest" else _parse_markdown(source)


# ---------------------------------------------------------------------------
# Edge recovery
# ---------------------------------------------------------------------------

# Per-kind edge confidence. CONTAINS is structural and certain; SIBLING is a
# weaker adjacency signal; cross-references decay with document distance and
# citations are weaker still.
_CONTAINS_CONF = 1.0
_SIBLING_CONF = 0.5


def _ref(
    source_id: str, target_id: str, kind: ReferenceKind, confidence: float, evidence: str
) -> Reference:
    """Build one typed edge with a single evidence string."""
    return Reference(
        source_id=source_id,
        target_id=target_id,
        kind=kind,
        confidence=confidence,
        evidence=(evidence,),
    )


def _structural_edges(nodes: Sequence[_Node]) -> list[Reference]:
    """Recover CONTAINS (section to child) and SIBLING (adjacent sections) edges."""
    edges: list[Reference] = []
    for node in nodes:
        if node.parent is not None:
            parent = nodes[node.parent].unit
            crumb = " / ".join(node.unit.breadcrumb) or parent.title or parent.id
            edges.append(_ref(parent.id, node.unit.id, "CONTAINS", _CONTAINS_CONF, crumb))
    siblings: dict[int | None, list[int]] = {}
    for index, node in enumerate(nodes):
        if node.is_section:
            siblings.setdefault(node.parent, []).append(index)
    for members in siblings.values():
        for left, right in pairwise(members):
            first, second = nodes[left].unit, nodes[right].unit
            shared = " / ".join(first.breadcrumb) or "(document root)"
            edges.append(_ref(first.id, second.id, "SIBLING", _SIBLING_CONF, shared))
            edges.append(_ref(second.id, first.id, "SIBLING", _SIBLING_CONF, shared))
    return edges


# Inline Markdown link: [text](dest), excluding images; captures the dest URL.
_INLINE_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+[^)]*)?\)")
# A URL scheme (http:, mailto:, ...) marks an external link.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")
# reST internal references: `Phrase`_ and word_ (single trailing underscore).
_REST_PHRASE_REF_RE = re.compile(r"`([^`]+)`_")
_REST_SIMPLE_REF_RE = re.compile(r"(?<![\w`])([A-Za-z][\w.\-]*)_(?![\w_])")

_CROSSREF_INTRA_CONF = 1.0
_CROSSREF_INTER_CONF = 0.9

# Within one link() call, neighbor documents are parsed once and reused: maps a
# resolved neighbor path to its SECTION units (empty when missing or non-prose).
NeighborCache = dict[str, list[Unit]]


def _slug(title: str) -> str:
    """GitHub-style heading slug: lowercase, drop punctuation, spaces to hyphens."""
    lowered = re.sub(r"[^\w\s-]", "", title.strip().lower())
    return re.sub(r"\s+", "-", lowered)


def _section_slugs(nodes: Sequence[_Node]) -> dict[str, str]:
    """Map each section's anchor slug to its unit id (first occurrence wins)."""
    slugs: dict[str, str] = {}
    for node in nodes:
        if node.is_section and node.unit.title:
            slugs.setdefault(_slug(node.unit.title), node.unit.id)
    return slugs


def _section_titles(nodes: Sequence[_Node]) -> dict[str, str]:
    """Map each section's lowercased title to its unit id (for reST references)."""
    titles: dict[str, str] = {}
    for node in nodes:
        if node.is_section and node.unit.title:
            titles.setdefault(node.unit.title.strip().lower(), node.unit.id)
    return titles


def _neighbor_sections(target: Path, cache: NeighborCache) -> list[Unit]:
    """Return the neighbor's SECTION units, parsing it at most once per link call.

    A missing or non-prose target caches and returns an empty list, so repeated
    links to the same neighbor neither re-read nor re-parse it within one call.
    """
    key = str(target)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not target.is_file() or not _claims_prose(target):
        cache[key] = []
        return []
    document = ProseAdapter().ingest(target)
    sections = [unit for unit in ProseAdapter().parse(document) if unit.kind == "SECTION"]
    cache[key] = sections
    return sections


def _neighbor_section_id(base_uri: str, dest: str, cache: NeighborCache) -> str | None:
    """Resolve an inter-document link to a neighbor section's unit id, or None.

    The link path is resolved relative to the source document; the neighbor is
    ingested and parsed exactly as the corpus would, so the returned id matches
    the neighbor's own parsed unit when both are indexed under the same path. A
    non-prose, missing, or unresolvable target yields None (no edge), never a
    fabricated one. An anchor selects the section whose slug matches; without an
    anchor the neighbor's first (root) section is used.
    """
    path_part, _, anchor = dest.partition("#")
    if not path_part:
        return None
    sections = _neighbor_sections((Path(base_uri).parent / path_part).resolve(), cache)
    if not sections:
        return None
    if anchor:
        slug = anchor.strip().lower()
        return next((unit.id for unit in sections if _slug(unit.title or "") == slug), None)
    return sections[0].id


def _resolve_anchor(
    base_uri: str, dest: str, slugs: dict[str, str], cache: NeighborCache
) -> tuple[str, float] | None:
    """Resolve a Markdown link dest to a (unit id, confidence), or None."""
    if dest.startswith("#"):
        target = slugs.get(dest[1:].strip().lower())
        return (target, _CROSSREF_INTRA_CONF) if target is not None else None
    if _SCHEME_RE.match(dest) or dest.startswith("//"):
        return None
    target = _neighbor_section_id(base_uri, dest, cache)
    return (target, _CROSSREF_INTER_CONF) if target is not None else None


def _markdown_crossref_edges(
    document: Document, nodes: Sequence[_Node], cache: NeighborCache
) -> list[Reference]:
    """Recover CROSS_REF edges from Markdown inline anchor and document links."""
    slugs = _section_slugs(nodes)
    edges: list[Reference] = []
    for node in nodes:
        if node.unit.protect:
            continue
        for dest in _INLINE_LINK_RE.findall(node.unit.text):
            resolved = _resolve_anchor(document.uri, dest, slugs, cache)
            if resolved is not None and resolved[0] != node.unit.id:
                edges.append(
                    Reference(node.unit.id, resolved[0], "CROSS_REF", resolved[1], (dest,))
                )
    return edges


def _rest_crossref_edges(nodes: Sequence[_Node]) -> list[Reference]:
    """Recover CROSS_REF edges from reST internal section references."""
    titles = _section_titles(nodes)
    edges: list[Reference] = []
    for node in nodes:
        if node.unit.protect:
            continue
        names = _REST_PHRASE_REF_RE.findall(node.unit.text)
        names += _REST_SIMPLE_REF_RE.findall(node.unit.text)
        for name in names:
            target = titles.get(name.strip().lower())
            if target is not None and target != node.unit.id:
                edges.append(
                    Reference(node.unit.id, target, "CROSS_REF", _CROSSREF_INTRA_CONF, (name,))
                )
    return edges


def _crossref_edges(
    document: Document, nodes: Sequence[_Node], flavor: Flavor, cache: NeighborCache
) -> list[Reference]:
    """Recover CROSS_REF edges for the document's flavor."""
    if flavor == "rest":
        return _rest_crossref_edges(nodes)
    return _markdown_crossref_edges(document, nodes, cache)


# Footnote reference [^id] and its definition line [^id]: ...
_FOOTNOTE_REF_RE = re.compile(r"\[\^([^\]\s]+)\]")
_FOOTNOTE_DEF_RE = re.compile(r"^[ \t]*\[\^([^\]\s]+)\]:", re.MULTILINE)
# Reference-style link [text][label] (label empty => collapsed to text) and a
# link reference definition [label]: dest (a footnote's ^label is excluded).
_REF_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\[([^\]]*)\]")
_REF_DEF_RE = re.compile(r"^[ \t]*\[([^\]^]+)\]:[ \t]*(\S+)", re.MULTILINE)
# reST footnote/citation reference [id]_ and its definition .. [id] ...
_REST_LABEL_REF_RE = re.compile(r"\[([\w#.\-]+)\]_")
_REST_LABEL_DEF_RE = re.compile(r"^[ \t]*\.\.[ \t]+\[([\w#.\-]+)\]", re.MULTILINE)

_CITES_FOOTNOTE_CONF = 0.6
_CITES_REFLINK_CONF = 0.7


def _footnote_defs(nodes: Sequence[_Node]) -> dict[str, str]:
    """Map each footnote id to the unit that defines it (first occurrence wins)."""
    defs: dict[str, str] = {}
    for node in nodes:
        if node.unit.protect:
            continue
        for match in _FOOTNOTE_DEF_RE.finditer(node.unit.text):
            defs.setdefault(match.group(1), node.unit.id)
    return defs


def _reflink_defs(nodes: Sequence[_Node]) -> dict[str, str]:
    """Map each link reference label (lowercased) to its destination string."""
    defs: dict[str, str] = {}
    for node in nodes:
        if node.unit.protect:
            continue
        for match in _REF_DEF_RE.finditer(node.unit.text):
            defs.setdefault(match.group(1).strip().lower(), match.group(2))
    return defs


def _markdown_cites_edges(
    document: Document, nodes: Sequence[_Node], cache: NeighborCache
) -> list[Reference]:
    """Recover CITES edges from Markdown footnotes and reference-style links."""
    footnotes = _footnote_defs(nodes)
    reflinks = _reflink_defs(nodes)
    slugs = _section_slugs(nodes)
    edges: list[Reference] = []
    for node in nodes:
        if node.unit.protect:
            continue
        text = node.unit.text
        for match in _FOOTNOTE_REF_RE.finditer(text):
            line_start = text.rfind("\n", 0, match.start()) + 1
            is_definition = (
                text[line_start : match.start()].strip() == ""
                and text[match.end() : match.end() + 1] == ":"
            )
            if is_definition:
                continue  # a line-anchored [^id]: is the definition's own marker
            target = footnotes.get(match.group(1))
            if target is not None and target != node.unit.id:
                edges.append(
                    Reference(
                        node.unit.id,
                        target,
                        "CITES",
                        _CITES_FOOTNOTE_CONF,
                        (f"[^{match.group(1)}]",),
                    )
                )
        for match in _REF_LINK_RE.finditer(text):
            label = (match.group(2) or match.group(1)).strip().lower()
            dest = reflinks.get(label)
            if dest is None:
                continue
            resolved = _resolve_anchor(document.uri, dest, slugs, cache)
            if resolved is not None and resolved[0] != node.unit.id:
                edges.append(
                    Reference(
                        node.unit.id, resolved[0], "CITES", _CITES_REFLINK_CONF, (f"[{label}]",)
                    )
                )
    return edges


def _rest_cites_edges(nodes: Sequence[_Node]) -> list[Reference]:
    """Recover CITES edges from reST footnote and citation references."""
    defs: dict[str, str] = {}
    for node in nodes:
        if node.unit.protect:
            continue
        for match in _REST_LABEL_DEF_RE.finditer(node.unit.text):
            defs.setdefault(match.group(1), node.unit.id)
    edges: list[Reference] = []
    for node in nodes:
        if node.unit.protect:
            continue
        for match in _REST_LABEL_REF_RE.finditer(node.unit.text):
            target = defs.get(match.group(1))
            if target is not None and target != node.unit.id:
                edges.append(
                    Reference(
                        node.unit.id,
                        target,
                        "CITES",
                        _CITES_FOOTNOTE_CONF,
                        (f"[{match.group(1)}]",),
                    )
                )
    return edges


def _cites_edges(
    document: Document, nodes: Sequence[_Node], flavor: Flavor, cache: NeighborCache
) -> list[Reference]:
    """Recover CITES edges for the document's flavor."""
    if flavor == "rest":
        return _rest_cites_edges(nodes)
    return _markdown_cites_edges(document, nodes, cache)


def _dedup_sort(refs: Sequence[Reference]) -> list[Reference]:
    """Collapse duplicate (source, target, kind) edges and sort canonically.

    Several occurrences that resolve to the same edge fold into one Reference
    whose evidence is the union of contributing strings and whose confidence is
    the strongest seen, so the edge set is deterministic regardless of scan order.
    """
    grouped: dict[tuple[str, str, ReferenceKind], tuple[float, set[str]]] = {}
    for ref in refs:
        key = (ref.source_id, ref.target_id, ref.kind)
        confidence, evidence = grouped.get(key, (0.0, set()))
        grouped[key] = (max(confidence, ref.confidence), evidence | set(ref.evidence))
    out = [
        Reference(
            source_id=source_id,
            target_id=target_id,
            kind=kind,
            confidence=confidence,
            evidence=tuple(sorted(evidence)),
        )
        for (source_id, target_id, kind), (confidence, evidence) in grouped.items()
    ]
    out.sort(key=lambda ref: (ref.source_id, ref.target_id, ref.kind, ref.evidence))
    return out


class ProseAdapter:
    """Deterministic Markdown / reStructuredText adapter emitting the IR."""

    __slots__ = ()

    def claims(self, source: Path) -> bool:
        """Return True for a prose source, by extension or heading structure."""
        return _claims_prose(source)

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
        """Recover the typed edges among ``units`` as Reference values.

        Re-parses the source into the heading tree and recovers, deterministically
        and model-free: ``CONTAINS`` from a section to each direct child,
        ``SIBLING`` between adjacent sections sharing a parent, ``CROSS_REF`` for
        intra-document anchor links and inter-document links resolved to a
        neighbor section, and ``CITES`` for footnotes and reference-style links.
        link fails loud when the units it is given do not match the parsed source.
        """
        flavor = _flavor(document.uri)
        source = read_source(document)
        nodes = _build_nodes(document, source, _blocks(source, flavor))
        if {node.unit.id for node in nodes} != {unit.id for unit in units}:
            raise ValueError("link received units that do not match the parsed source")
        cache: NeighborCache = {}
        edges = _structural_edges(nodes)
        edges += _crossref_edges(document, nodes, flavor, cache)
        edges += _cites_edges(document, nodes, flavor, cache)
        return _dedup_sort(edges)

    def capabilities(self) -> AdapterCapabilities:
        """Report emittable kinds, determinism, and model-extraction opt-in."""
        return AdapterCapabilities(
            unit_kinds=frozenset({"SECTION", "PARAGRAPH", "TABLE", "FIGURE_CAPTION"}),
            reference_kinds=frozenset({"CONTAINS", "SIBLING", "CROSS_REF", "CITES"}),
            deterministic_parse=True,
            model_extraction_opt_in=False,
        )
