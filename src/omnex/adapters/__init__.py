"""Modality adapters: the IR-emitting boundary between sources and the kernel.

Adapters are modality-specific and depend on the kernel and IR, never the
reverse. The kernel only ever sees the IR these adapters emit. ``select_adapter``
routes a source to the first adapter that claims it, failing loud when none does.
"""

from __future__ import annotations

from pathlib import Path

from omnex.adapters.base import AdapterCapabilities, ModalityAdapter
from omnex.adapters.spec import SpecAdapter

# Adapters are tried in order; the first whose ``claims`` returns True handles the
# source. This is the single routing registry the public API dispatches through.
_ADAPTERS: tuple[ModalityAdapter, ...] = (SpecAdapter(),)

__all__ = ["AdapterCapabilities", "ModalityAdapter", "SpecAdapter", "select_adapter"]


def select_adapter(source: Path) -> ModalityAdapter:
    """Return the adapter that claims ``source``, failing loud if none does."""
    for adapter in _ADAPTERS:
        if adapter.claims(source):
            return adapter
    raise ValueError(f"no adapter claims source: {source}")
