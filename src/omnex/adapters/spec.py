"""OpenAPI / JSON-Schema modality adapter.

Detects, ingests, and parses JSON-encoded OpenAPI and JSON-Schema documents into
the modality-agnostic IR. The adapter is deterministic and model-free: it never
calls a language model on any path. ``raw_token_count`` is measured with the
``tiktoken`` ``cl100k_base`` tokenizer, a deterministic offline encoder loaded
once and reused (a tokenizer, not a model lane), and ``$ref`` edge recovery lives
in :meth:`SpecAdapter.link`.

Scope is JSON-encoded specs only: OpenAPI 2/3 (``openapi``/``swagger`` key) and
JSON-Schema (``$schema``/``$defs``/``definitions`` or a root object schema). Unit
ids are derived from each construct's RFC 6901 JSON pointer, so they are stable
across reformatting and let :meth:`link` resolve ``$ref`` targets to unit ids
without a second pass. Spans are character offsets into the normalized source,
consistent with the IR ``Span`` contract; the JSON pointer is carried in the
breadcrumb and in edge evidence.

Adapters depend on the kernel and IR, never the reverse.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

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
    normalize_content,
)
from omnex.kernel.packer import count_tokens

if TYPE_CHECKING:
    import tiktoken

# Identifier of the deterministic offline tiktoken encoding used for raw counts.
_ENCODING = "cl100k_base"

# Path-item members that denote an OPERATION unit.
_HTTP_METHODS: frozenset[str] = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    """Load the deterministic offline ``tiktoken`` encoder once and cache it.

    The import and load are deferred to first use, so importing this module (and
    ``omnex``) is cheap; the cl100k_base vocab is loaded from the local tiktoken
    cache on first use. The retrieval path never touches it (it budgets in the
    kernel's whitespace ``count_tokens``); only ``raw_token_count`` uses it.
    """
    import tiktoken

    return tiktoken.get_encoding(_ENCODING)


def _escape(segment: str) -> str:
    """Escape one JSON pointer reference token per RFC 6901 (``~`` then ``/``)."""
    return segment.replace("~", "~0").replace("/", "~1")


def _pointer(segments: Sequence[str]) -> str:
    """Build an RFC 6901 JSON pointer from raw (unescaped) path segments."""
    return "".join("/" + _escape(segment) for segment in segments)


def _unit_id(document_id: str, pointer: str) -> str:
    """Derive a stable unit id from its document and RFC 6901 JSON pointer."""
    body = f"{document_id}\x00{pointer}".encode()
    return f"unit:{sha256(body).hexdigest()[:16]}"


def _classify(has_key: Callable[[str], bool]) -> str | None:
    """Classify a spec by its top-level keys (the one detection predicate).

    Shared by ``claims`` (over a decoded dict) and parsing (over a parsed node),
    so detection never drifts between the two. Swagger 2.0 is out of scope and is
    rejected here rather than partially parsed, so a ``swagger`` document is never
    claimed.
    """
    if has_key("swagger"):
        return None
    if has_key("openapi"):
        return "openapi"
    if (
        has_key("$schema")
        or has_key("$defs")
        or has_key("definitions")
        or (has_key("type") and has_key("properties"))
    ):
        return "jsonschema"
    return None


# ---------------------------------------------------------------------------
# Position-tracking JSON parser
#
# The stdlib decoder discards source offsets, but the IR needs a Span back to
# source for every construct. This minimal recursive-descent parser records the
# character span of every value while preserving object member order, so units
# carry exact, deterministic spans into the normalized source.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _JsonNode:
    """A parsed JSON value with its character span in the normalized source."""

    start: int
    end: int
    members: tuple[tuple[str, _JsonNode], ...] | None
    elements: tuple[_JsonNode, ...] | None

    @property
    def is_object(self) -> bool:
        return self.members is not None

    def get(self, key: str) -> _JsonNode | None:
        if self.members is None:
            return None
        for name, node in self.members:
            if name == key:
                return node
        return None


class _SpanParser:
    """Minimal recursive-descent JSON parser recording per-value char spans."""

    __slots__ = ("_i", "_n", "_s")

    def __init__(self, text: str) -> None:
        self._s = text
        self._i = 0
        self._n = len(text)

    def parse(self) -> _JsonNode:
        self._ws()
        node = self._value()
        self._ws()
        if self._i != self._n:
            raise ValueError(f"trailing data at offset {self._i}")
        return node

    def _ws(self) -> None:
        while self._i < self._n and self._s[self._i] in " \t\n\r":
            self._i += 1

    def _value(self) -> _JsonNode:
        if self._i >= self._n:
            raise ValueError("unexpected end of input")
        ch = self._s[self._i]
        if ch == "{":
            return self._object()
        if ch == "[":
            return self._array()
        if ch == '"':
            start = self._i
            self._skip_string()
            return _JsonNode(start, self._i, None, None)
        if ch == "-" or ch.isdigit():
            return self._number()
        for literal in ("true", "false", "null"):
            if self._s.startswith(literal, self._i):
                start = self._i
                self._i += len(literal)
                return _JsonNode(start, self._i, None, None)
        raise ValueError(f"unexpected character {ch!r} at offset {self._i}")

    def _object(self) -> _JsonNode:
        start = self._i
        self._i += 1
        members: list[tuple[str, _JsonNode]] = []
        seen: set[str] = set()
        self._ws()
        if self._i < self._n and self._s[self._i] == "}":
            self._i += 1
            return _JsonNode(start, self._i, (), None)
        while True:
            self._ws()
            if self._i >= self._n or self._s[self._i] != '"':
                raise ValueError(f"expected object key at offset {self._i}")
            key = self._string_value()
            if key in seen:
                raise ValueError(f"duplicate object key {key!r} at offset {self._i}")
            seen.add(key)
            self._ws()
            if self._i >= self._n or self._s[self._i] != ":":
                raise ValueError(f"expected ':' at offset {self._i}")
            self._i += 1
            self._ws()
            members.append((key, self._value()))
            self._ws()
            if self._i >= self._n:
                raise ValueError("unterminated object")
            ch = self._s[self._i]
            self._i += 1
            if ch == ",":
                continue
            if ch == "}":
                return _JsonNode(start, self._i, tuple(members), None)
            raise ValueError(f"expected ',' or '}}' at offset {self._i - 1}")

    def _array(self) -> _JsonNode:
        start = self._i
        self._i += 1
        elements: list[_JsonNode] = []
        self._ws()
        if self._i < self._n and self._s[self._i] == "]":
            self._i += 1
            return _JsonNode(start, self._i, None, ())
        while True:
            self._ws()
            elements.append(self._value())
            self._ws()
            if self._i >= self._n:
                raise ValueError("unterminated array")
            ch = self._s[self._i]
            self._i += 1
            if ch == ",":
                continue
            if ch == "]":
                return _JsonNode(start, self._i, None, tuple(elements))
            raise ValueError(f"expected ',' or ']' at offset {self._i - 1}")

    def _number(self) -> _JsonNode:
        start = self._i
        if self._s[self._i] == "-":
            self._i += 1
        while self._i < self._n and self._s[self._i] in "0123456789.eE+-":
            self._i += 1
        return _JsonNode(start, self._i, None, None)

    def _skip_string(self) -> None:
        self._i += 1
        while self._i < self._n:
            ch = self._s[self._i]
            if ch == "\\":
                self._i += 2
                continue
            if ch == '"':
                self._i += 1
                return
            self._i += 1
        raise ValueError("unterminated string")

    def _string_value(self) -> str:
        start = self._i
        self._skip_string()
        decoded = json.loads(self._s[start : self._i])
        if not isinstance(decoded, str):  # pragma: no cover - guarded by _skip_string
            raise ValueError(f"expected string at offset {start}")
        return decoded


@dataclass(frozen=True, slots=True)
class _UnitNode:
    """An emitted unit's location and metadata, before IR construction."""

    pointer: str
    kind: UnitKind
    title: str
    breadcrumb: tuple[str, ...]
    span: Span
    summary: str | None


def _string_member(node: _JsonNode, key: str, source: str) -> str | None:
    """Return the decoded string value of ``node[key]``, or None if absent."""
    member = node.get(key)
    if member is None or member.members is not None or member.elements is not None:
        return None
    raw = source[member.start : member.end]
    if not raw.startswith('"'):
        return None
    value = json.loads(raw)
    return value if isinstance(value, str) else None


def _collect_operations(paths: _JsonNode, source: str, units: list[_UnitNode]) -> None:
    """Emit an OPERATION unit for every HTTP method under each path item."""
    if paths.members is None:
        return
    for path_key, item in paths.members:
        if not item.is_object or item.members is None:
            continue
        for method, operation in item.members:
            if method.lower() not in _HTTP_METHODS or not operation.is_object:
                continue
            summary = _string_member(operation, "summary", source) or _string_member(
                operation, "description", source
            )
            units.append(
                _UnitNode(
                    pointer=_pointer(["paths", path_key, method]),
                    kind="OPERATION",
                    title=f"{method.upper()} {path_key}",
                    breadcrumb=("paths",),
                    span=Span(operation.start, operation.end),
                    summary=summary,
                )
            )


def _is_reference_property(value: _JsonNode) -> bool:
    """True when a property value is exactly a ``$ref`` (no inline shape of its own).

    A ``$ref`` carrying sibling keys has inline content of its own, so it is kept
    as a FIELD; only a sole-member ``$ref`` is treated as a pure schema edge.
    """
    return value.members is not None and len(value.members) == 1 and value.get("$ref") is not None


def _collect_fields(
    schema: _JsonNode, base: Sequence[str], source: str, units: list[_UnitNode]
) -> None:
    """Emit a FIELD unit for each inline (non-``$ref``) property of a schema.

    A property that is a bare ``$ref`` has no inline shape; it is recovered as a
    schema edge in ``link`` instead of a FIELD, so the reference is attributed to
    the enclosing schema rather than to a synthetic field.
    """
    properties = schema.get("properties")
    if properties is None or properties.members is None:
        return
    for name, value in properties.members:
        if _is_reference_property(value):
            continue
        units.append(
            _UnitNode(
                pointer=_pointer([*base, "properties", name]),
                kind="FIELD",
                title=name,
                breadcrumb=(*base, "properties"),
                span=Span(value.start, value.end),
                summary=_string_member(value, "description", source) if value.is_object else None,
            )
        )


def _collect_schemas(
    container: _JsonNode, base: Sequence[str], source: str, units: list[_UnitNode]
) -> None:
    """Emit a SCHEMA unit and its FIELD units for each named schema in a container."""
    if container.members is None:
        return
    for name, schema in container.members:
        if not schema.is_object:
            continue
        schema_base = [*base, name]
        units.append(
            _UnitNode(
                pointer=_pointer(schema_base),
                kind="SCHEMA",
                title=name,
                breadcrumb=tuple(base),
                span=Span(schema.start, schema.end),
                summary=_string_member(schema, "description", source)
                or _string_member(schema, "title", source),
            )
        )
        _collect_fields(schema, schema_base, source, units)


def _collect_root_schema(root: _JsonNode, source: str, units: list[_UnitNode]) -> None:
    """Emit the root JSON-Schema as a SCHEMA unit when it has its own properties."""
    if root.get("properties") is None:
        return
    units.append(
        _UnitNode(
            pointer="",
            kind="SCHEMA",
            title=_string_member(root, "title", source) or "(root)",
            breadcrumb=(),
            span=Span(root.start, root.end),
            summary=_string_member(root, "description", source),
        )
    )
    _collect_fields(root, [], source, units)


def _collect_unit_nodes(root: _JsonNode, source: str, flavor: str) -> list[_UnitNode]:
    """Collect the OPERATION/SCHEMA/FIELD unit nodes of a parsed spec.

    Shared by ``parse`` (to build units) and ``link`` (to know which pointers are
    emitted units), so the two never disagree about what a unit is.
    """
    nodes: list[_UnitNode] = []
    if flavor == "openapi":
        paths = root.get("paths")
        if paths is not None and paths.is_object:
            _collect_operations(paths, source, nodes)
        components = root.get("components")
        schemas = components.get("schemas") if components is not None else None
        if schemas is not None and schemas.is_object:
            _collect_schemas(schemas, ["components", "schemas"], source, nodes)
    else:
        for defs_key in ("$defs", "definitions"):
            defs = root.get(defs_key)
            if defs is not None and defs.is_object:
                _collect_schemas(defs, [defs_key], source, nodes)
        _collect_root_schema(root, source, nodes)
    return nodes


@dataclass(frozen=True, slots=True)
class _RefSite:
    """One ``$ref`` occurrence: its pointer, target string, and owning property."""

    ref_pointer: str
    target: str
    property_name: str | None


def _resolve_target(reference: str) -> str | None:
    """Return the local JSON pointer a ``$ref`` addresses, or None if external.

    Only internal ``#/...`` references resolve to a local unit; an external
    reference (a file or URL) is out of scope and yields no local edge.
    """
    if not reference.startswith("#"):
        return None
    return reference[1:]


def _resolve_pointer(root: _JsonNode, pointer: str) -> _JsonNode | None:
    """Resolve an RFC 6901 JSON pointer against the parsed tree, or None.

    Used to tell a target that exists but is not a retrievable unit (an OpenAPI
    parameter/response/requestBody, etc.) apart from a truly dangling reference.
    """
    if pointer == "":
        return root
    node = root
    for token in pointer.split("/")[1:]:
        segment = token.replace("~1", "/").replace("~0", "~")
        if node.members is not None:
            child = node.get(segment)
            if child is None:
                return None
            node = child
        elif node.elements is not None:
            try:
                index = int(segment)
            except ValueError:
                return None
            if not 0 <= index < len(node.elements):
                return None
            node = node.elements[index]
        else:
            return None
    return node


def _is_foreign_key(property_name: str) -> bool:
    """True for a conventional foreign-key property name (``*_id`` / ``*Id``)."""
    return property_name.endswith("_id") or property_name.endswith("Id")


def _nearest_emitted(ref_pointer: str, emitted: frozenset[str]) -> str | None:
    """Return the longest emitted unit pointer that encloses ``ref_pointer``.

    The reference is attributed to its nearest enclosing emitted unit, so a
    ``$ref`` inside an operation is an operation edge, and a bare reference
    property is attributed to its schema rather than a (non-emitted) field.
    """
    parts = ref_pointer.split("/")
    best: str | None = None
    best_depth = -1
    for pointer in emitted:
        candidate = pointer.split("/")
        depth = len(candidate)
        if depth < len(parts) and parts[:depth] == candidate and depth > best_depth:
            best = pointer
            best_depth = depth
    return best


def _collect_refs(root: _JsonNode, source: str) -> list[_RefSite]:
    """Collect every ``$ref`` occurrence with its pointer and owning property."""
    sites: list[_RefSite] = []
    _walk_refs(root, [], source, sites)
    return sites


def _walk_refs(node: _JsonNode, segments: list[str], source: str, sites: list[_RefSite]) -> None:
    if node.members is not None:
        ref = node.get("$ref")
        if ref is not None and ref.members is None and ref.elements is None:
            raw = source[ref.start : ref.end]
            if raw.startswith('"'):
                target = json.loads(raw)
                if isinstance(target, str):
                    property_name = (
                        segments[-1]
                        if len(segments) >= 2 and segments[-2] == "properties"
                        else None
                    )
                    sites.append(_RefSite(_pointer([*segments, "$ref"]), target, property_name))
        for key, child in node.members:
            _walk_refs(child, [*segments, key], source, sites)
    elif node.elements is not None:
        for index, child in enumerate(node.elements):
            _walk_refs(child, [*segments, str(index)], source, sites)


class SpecAdapter:
    """Deterministic OpenAPI / JSON-Schema adapter emitting the IR."""

    __slots__ = ()

    def claims(self, source: Path) -> bool:
        """Return True for a JSON-encoded OpenAPI or JSON-Schema source."""
        try:
            text = source.read_text(encoding="utf-8")
        except OSError:
            return False
        try:
            data = json.loads(text)
        except ValueError:
            return False
        if not isinstance(data, dict):
            return False
        return _classify(data.__contains__) is not None

    def ingest(self, source: Path) -> Document:
        """Establish document identity, content hash, and raw token count."""
        text = source.read_text(encoding="utf-8")
        content_hash = compute_content_hash(text)
        raw_token_count = len(_encoder().encode(normalize_content(text)))
        uri = str(source)
        return Document(
            id=make_document_id(uri=uri, content_hash=content_hash),
            uri=uri,
            modality="spec",
            content_hash=content_hash,
            raw_token_count=raw_token_count,
        )

    def parse(self, document: Document) -> list[Unit]:
        """Emit the retrievable units of a spec document.

        Deterministic and model-free. Units are emitted in source order; their
        ids are pointer-derived and their spans are character offsets into the
        normalized source.
        """
        source = _read_source(document)
        root = _SpanParser(source).parse()
        flavor = _flavor_node(root)
        if flavor is None:
            raise ValueError(f"source is not a supported spec: {document.uri}")

        nodes = _collect_unit_nodes(root, source, flavor)
        return [self._build_unit(document, source, node) for node in nodes]

    def link(self, document: Document, units: Sequence[Unit]) -> list[Reference]:
        """Recover ``$ref`` edges among ``units`` as typed Reference values.

        Each internal ``$ref`` becomes one edge from its nearest enclosing
        emitted unit to the referenced unit, with ``confidence = 1.0`` and the
        ``$ref`` JSON pointer as evidence. A reference property named like a
        foreign key (``*_id`` / ``*Id``) is classified ``FOREIGN_KEY``; every
        other reference is ``REFERENCES``. An external reference, or an internal
        reference whose target is a real node that is not a retrievable unit, is
        skipped; only a reference whose target resolves to nothing fails loud.
        Self- and mutually-recursive schemas terminate because references are
        resolved one hop and never traversed; a self-reference is a real
        ``source == target`` edge emitted once.
        """
        source = _read_source(document)
        root = _SpanParser(source).parse()
        flavor = _flavor_node(root)
        if flavor is None:
            raise ValueError(f"source is not a supported spec: {document.uri}")
        nodes = _collect_unit_nodes(root, source, flavor)
        if {_unit_id(document.id, node.pointer) for node in nodes} != {unit.id for unit in units}:
            raise ValueError("link received units that do not match the parsed source")
        emitted = frozenset(node.pointer for node in nodes)
        edges: dict[tuple[str, str, ReferenceKind], set[str]] = {}
        for site in _collect_refs(root, source):
            target_pointer = _resolve_target(site.target)
            if target_pointer is None:
                continue
            if target_pointer not in emitted:
                # A target that resolves to a real but non-unit node (an OpenAPI
                # parameter/response/requestBody, etc.) is out of scope, like an
                # external ref; only a target resolving to nothing is dangling.
                if _resolve_pointer(root, target_pointer) is None:
                    raise ValueError(f"dangling internal reference: {site.target}")
                continue
            source_pointer = _nearest_emitted(site.ref_pointer, emitted)
            if source_pointer is None:
                # The $ref sits outside any retrievable unit (e.g. root-level
                # composition with no emitted root schema): no source to anchor.
                continue
            # Cyclic-reference guard: references are resolved one hop and never
            # traversed, so a self- or mutually-recursive schema yields a finite
            # edge set. A self-reference is a real edge (source == target) and is
            # emitted once via the dedup above; downstream closure terminates on
            # the visited set.
            kind: ReferenceKind = (
                "FOREIGN_KEY"
                if site.property_name is not None and _is_foreign_key(site.property_name)
                else "REFERENCES"
            )
            key = (
                _unit_id(document.id, source_pointer),
                _unit_id(document.id, target_pointer),
                kind,
            )
            # Dedup shared targets: several $ref sites that resolve to the same
            # (source, target, kind) edge collapse into one Reference whose
            # evidence carries every contributing pointer.
            edges.setdefault(key, set()).add(site.ref_pointer)
        references = [
            Reference(
                source_id=source_id,
                target_id=target_id,
                kind=kind,
                confidence=1.0,
                evidence=tuple(sorted(evidence)),
            )
            for (source_id, target_id, kind), evidence in edges.items()
        ]
        references.sort(key=lambda ref: (ref.source_id, ref.target_id, ref.kind, ref.evidence))
        return references

    def capabilities(self) -> AdapterCapabilities:
        """Report emittable kinds, determinism, and model-extraction opt-in."""
        return AdapterCapabilities(
            unit_kinds=frozenset({"OPERATION", "SCHEMA", "FIELD"}),
            reference_kinds=frozenset({"REFERENCES", "FOREIGN_KEY"}),
            deterministic_parse=True,
            model_extraction_opt_in=False,
        )

    @staticmethod
    def _build_unit(document: Document, source: str, node: _UnitNode) -> Unit:
        text = source[node.span.start : node.span.end]
        return Unit(
            id=_unit_id(document.id, node.pointer),
            document_id=document.id,
            span=node.span,
            text=text,
            token_count=count_tokens(text),
            title=node.title,
            breadcrumb=node.breadcrumb,
            kind=node.kind,
            summary=node.summary,
            protect=node.kind in ("OPERATION", "SCHEMA"),
        )


def _flavor_node(root: _JsonNode) -> str | None:
    """Classify a parsed root node, reusing the shared detection predicate."""
    if not root.is_object:
        return None
    return _classify(lambda key: root.get(key) is not None)


def _read_source(document: Document) -> str:
    """Re-read and normalize a document's source, failing loud if it changed.

    ``parse`` and ``link`` reconstruct the source from ``document.uri`` (the IR
    carries identity, not text). The content hash is re-verified so a source that
    changed between ``ingest`` and parsing is rejected rather than silently
    yielding stale structure.
    """
    text = Path(document.uri).read_text(encoding="utf-8")
    if compute_content_hash(text) != document.content_hash:
        raise ValueError(f"source changed since ingest: {document.uri}")
    return normalize_content(text)
