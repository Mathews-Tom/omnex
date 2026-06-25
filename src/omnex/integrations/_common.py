"""Framework-free projection of an omnex retrieval into per-chunk records.

Shared by the LangChain and LlamaIndex retriever adapters so both map the same
``ContextBundle`` representations and ``Receipt`` provenance into their
framework's document/node type. This module imports no integration framework and
changes no retrieval ranking, returned set, or receipt schema: it only reshapes a
``(ContextBundle, Receipt)`` pair the kernel already produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from omnex._surface import receipt_dict
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.receipt import Receipt


@dataclass(frozen=True, slots=True)
class Chunk:
    """One emitted unit of a retrieval: the packed text plus its unit provenance.

    ``mode`` is the packer's representation mode (``INCLUDE``/``COMPRESS``/
    ``ELIDE``); ``SKIP`` representations are never emitted as chunks.
    """

    unit_id: str
    text: str
    mode: str
    token_count: int
    title: str | None
    breadcrumb: tuple[str, ...]
    kind: str
    document_id: str


def bundle_chunks(bundle: ContextBundle) -> list[Chunk]:
    """One ``Chunk`` per included (non-``SKIP``) representation, in packed order.

    Preserves the bundle's order and returned set exactly: an adapter emits one
    document/node per chunk, so the framework receives the same units omnex
    packed, never a re-ranked or re-chunked view.
    """
    chunks: list[Chunk] = []
    for rep in bundle.representations:
        if rep.mode == "SKIP":
            continue
        unit = bundle.units[rep.unit_id]
        chunks.append(
            Chunk(
                unit_id=rep.unit_id,
                text=rep.text,
                mode=rep.mode,
                token_count=rep.token_count,
                title=unit.title,
                breadcrumb=unit.breadcrumb,
                kind=unit.kind,
                document_id=unit.document_id,
            )
        )
    return chunks


def receipt_provenance(receipt: Receipt) -> dict[str, object]:
    """The receipt as JSON-serializable provenance carried on every doc/node."""
    return receipt_dict(receipt)


def chunk_metadata(chunk: Chunk, provenance: Mapping[str, object]) -> dict[str, object]:
    """The metadata an adapter attaches to a chunk's document/node.

    Carries the chunk's unit provenance (id, packer mode, token cost, title,
    breadcrumb, kind, source document) plus the shared retrieval ``provenance``
    (the receipt) under ``omnex_receipt`` so every emitted item is auditable.
    """
    return {
        "unit_id": chunk.unit_id,
        "mode": chunk.mode,
        "token_count": chunk.token_count,
        "title": chunk.title,
        "breadcrumb": list(chunk.breadcrumb),
        "kind": chunk.kind,
        "document_id": chunk.document_id,
        "omnex_receipt": dict(provenance),
    }
