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
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from omnex.adapters.base import AdapterCapabilities
from omnex.ir.types import (
    Document,
    Reference,
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

    The import and load are deferred to first use so importing this module (and
    ``omnex``) stays cheap and free of any download attempt.
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


def _flavor_of_value(data: object) -> str | None:
    """Classify a decoded JSON value as ``openapi``, ``jsonschema``, or unknown."""
    if not isinstance(data, dict):
        return None
    if "openapi" in data or "swagger" in data:
        return "openapi"
    if "$schema" in data or "$defs" in data or "definitions" in data:
        return "jsonschema"
    if "type" in data and "properties" in data:
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
        self._ws()
        if self._i < self._n and self._s[self._i] == "}":
            self._i += 1
            return _JsonNode(start, self._i, (), None)
        while True:
            self._ws()
            if self._i >= self._n or self._s[self._i] != '"':
                raise ValueError(f"expected object key at offset {self._i}")
            key = self._string_value()
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
        return _flavor_of_value(data) is not None

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

        nodes: list[_UnitNode] = []
        if flavor == "openapi":
            paths = root.get("paths")
            if paths is not None and paths.is_object:
                _collect_operations(paths, source, nodes)
        return [self._build_unit(document, source, node) for node in nodes]

    def link(self, document: Document, units: Sequence[Unit]) -> list[Reference]:
        """Recover ``$ref`` edges among ``units``."""
        raise NotImplementedError("spec $ref linking lands in the spec-adapter-link change")

    def capabilities(self) -> AdapterCapabilities:
        """Report emittable kinds, determinism, and model-extraction opt-in."""
        return AdapterCapabilities(
            unit_kinds=frozenset({"OPERATION", "SCHEMA", "FIELD"}),
            reference_kinds=frozenset(),
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
    """Classify a parsed root node as ``openapi``, ``jsonschema``, or unknown."""
    if not root.is_object:
        return None
    if root.get("openapi") is not None or root.get("swagger") is not None:
        return "openapi"
    if (
        root.get("$schema") is not None
        or root.get("$defs") is not None
        or root.get("definitions") is not None
        or (root.get("type") is not None and root.get("properties") is not None)
    ):
        return "jsonschema"
    return None


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
