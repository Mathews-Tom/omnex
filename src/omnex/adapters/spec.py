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
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from omnex.adapters.base import AdapterCapabilities
from omnex.ir.types import (
    Document,
    Reference,
    Unit,
    compute_content_hash,
    make_document_id,
    normalize_content,
)

if TYPE_CHECKING:
    import tiktoken

# Identifier of the deterministic offline tiktoken encoding used for raw counts.
_ENCODING = "cl100k_base"


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
        """Emit the retrievable units of a spec document."""
        raise NotImplementedError("spec parsing lands in a following commit")

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
