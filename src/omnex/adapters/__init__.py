"""Modality adapters: the IR-emitting boundary between sources and the kernel.

Adapters are modality-specific and depend on the kernel and IR, never the
reverse. The kernel only ever sees the IR these adapters emit.
"""

from __future__ import annotations

from omnex.adapters.base import AdapterCapabilities, ModalityAdapter
from omnex.adapters.spec import SpecAdapter

__all__ = ["AdapterCapabilities", "ModalityAdapter", "SpecAdapter"]
