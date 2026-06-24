"""Stable IR contract shared by every modality adapter and the kernel.

The four dataclasses (``Document``, ``Span``, ``Unit``, ``Reference``) are the
system boundary: adapters may be modality-specific, the kernel never is. All
four are frozen and slotted so IR values are immutable, hashable, and cheap.

Construction validates the invariants the rest of the system relies on, and the
content-address helpers derive deterministic, content-addressed identifiers so
identical normalized content always yields identical ids and hashes.

This module is import-safe: no model load, no network, no file-system read.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Literal

Modality = Literal["prose", "code", "spec", "config", "pdf", "image", "audio", "video"]

UnitKind = Literal[
    "SECTION",
    "PARAGRAPH",
    "TABLE",
    "FIGURE_CAPTION",
    "FUNCTION",
    "CLASS",
    "OPERATION",
    "SCHEMA",
    "FIELD",
]

ReferenceKind = Literal[
    "CONTAINS",
    "SIBLING",
    "CROSS_REF",
    "CITES",
    "LINKS_TO",
    "IMPORTS",
    "CALLS",
    "REFERENCES",
    "FOREIGN_KEY",
]


@dataclass(frozen=True, slots=True)
class Document:
    """A single ingested source, identified by its content address.

    ``content_hash`` is the normalized content address and ``id`` is stable for
    a given ``(uri, content_hash)``; both are produced by the helpers below.
    """

    id: str
    uri: str
    modality: Modality
    content_hash: str
    raw_token_count: int

    def __post_init__(self) -> None:
        if self.raw_token_count < 0:
            raise ValueError(f"raw_token_count must be >= 0, got {self.raw_token_count}")


@dataclass(frozen=True, slots=True)
class Span:
    """A character-offset range within a document's text, with ``start <= end``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"span start ({self.start}) must be <= end ({self.end})")


@dataclass(frozen=True, slots=True)
class Unit:
    """The retrievable, packable atom; never split during retrieval.

    ``protect=True`` marks units the packer must never compress or elide (code
    blocks, tables, formal definitions, payload fragments). ``breadcrumb`` is
    the ordered ancestry of section/container titles above this unit.
    """

    id: str
    document_id: str
    span: Span
    text: str
    token_count: int
    title: str | None
    breadcrumb: tuple[str, ...]
    kind: UnitKind
    summary: str | None
    protect: bool

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError(f"token_count must be >= 0, got {self.token_count}")


@dataclass(frozen=True, slots=True)
class Reference:
    """A typed, directed edge from one unit to another, with recovery evidence.

    ``confidence`` is the adapter's belief that the edge is real, in [0.0, 1.0];
    low confidence is surfaced as evidence, never hidden behind guessed structure.
    """

    source_id: str
    target_id: str
    kind: ReferenceKind
    confidence: float
    evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


def normalize_content(text: str) -> str:
    """Normalize text for content addressing.

    Unifies line endings (CRLF/CR to LF) and applies Unicode NFC so cosmetic
    differences hash identically. Pure and deterministic.
    """
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", unified)


def compute_content_hash(text: str) -> str:
    """Return the SHA-256 content address of ``text`` after normalization."""
    digest = hashlib.sha256(normalize_content(text).encode()).hexdigest()
    return f"sha256:{digest}"


def make_document_id(*, uri: str, content_hash: str) -> str:
    """Derive a stable document id from its uri and content hash.

    Distinct documents at the same uri but with different content (or the same
    content at different uris) receive distinct ids.
    """
    payload = f"{uri}\x00{content_hash}".encode()
    return f"doc:{hashlib.sha256(payload).hexdigest()[:16]}"


def make_unit_id(*, document_id: str, span: Span, text: str) -> str:
    """Derive a stable unit id from its document, span, and normalized text."""
    body = f"{document_id}\x00{span.start}\x00{span.end}\x00{normalize_content(text)}"
    return f"unit:{hashlib.sha256(body.encode()).hexdigest()[:16]}"
