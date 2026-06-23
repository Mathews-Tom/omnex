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
