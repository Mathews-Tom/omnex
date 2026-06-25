"""Modality adapters: the IR-emitting boundary between sources and the kernel.

Adapters are modality-specific and depend on the kernel and IR, never the
reverse. The kernel only ever sees the IR these adapters emit. ``select_adapter``
routes a source to the first adapter that claims it, failing loud when none does.
"""

from __future__ import annotations

from pathlib import Path

from omnex.adapters.base import AdapterCapabilities, ModalityAdapter
from omnex.adapters.prose import ProseAdapter
from omnex.adapters.spec import SpecAdapter

# Adapters are tried in order; the first whose ``claims`` returns True handles the
# source. SpecAdapter precedes ProseAdapter so a JSON-encoded spec is never
# claimed as prose. This is the single routing registry the public API dispatches
# through.
_ADAPTERS: tuple[ModalityAdapter, ...] = (SpecAdapter(), ProseAdapter())

__all__ = [
    "AdapterCapabilities",
    "ModalityAdapter",
    "ProseAdapter",
    "SpecAdapter",
    "available_adapters",
    "select_adapter",
]


def select_adapter(source: Path) -> ModalityAdapter:
    """Return the adapter that claims ``source``, failing loud if none does."""
    for adapter in _ADAPTERS:
        if adapter.claims(source):
            return adapter
    raise ValueError(f"no adapter claims source: {source}")


def available_adapters() -> tuple[ModalityAdapter, ...]:
    """The routing registry, in priority order -- the single source surfaces probe."""
    return _ADAPTERS
