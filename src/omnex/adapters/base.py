"""The modality adapter contract.

Every modality adapter is responsible for modality-specific detection, parsing,
and edge recovery, and emits only the IR (``Document``, ``Unit``, ``Reference``).
The kernel is modality-blind and accepts nothing else.

This module defines the contract only -- no concrete adapter. Model-backed
extraction (OCR, captioning, transcription) lives inside an adapter, never in the
kernel: it is opt-in, off by default, and any invocation must be recorded in the
``Receipt`` with the model version. An adapter that needs extraction it cannot
perform fails loud rather than fabricating structure.

This module is import-safe: no model load, no network, no file-system read.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """What an adapter can emit and whether it is deterministic.

    ``unit_kinds`` and ``reference_kinds`` are the ``UnitKind`` / ``ReferenceKind``
    values the adapter can produce. ``deterministic_parse`` is True when parsing
    is byte-exact and model-free. ``model_extraction_opt_in`` is True when the
    adapter has a model-backed extraction lane; that lane is off by default and,
    when invoked, must be recorded in the ``Receipt`` with the model version.
    """

    unit_kinds: frozenset[str]
    reference_kinds: frozenset[str]
    deterministic_parse: bool
    model_extraction_opt_in: bool
