"""Optional retriever adapters that plug omnex into external RAG frameworks.

Each adapter lives in its own submodule and imports its framework at module load,
so importing it without the matching extra fails loud -- exactly like
:mod:`omnex.mcp`. This package's ``__init__`` imports no framework, so neither
``import omnex`` nor ``import omnex.integrations`` pulls a heavy optional
dependency; only ``import omnex.integrations.langchain`` (the ``[langchain]``
extra) or ``import omnex.integrations.llamaindex`` (the ``[llamaindex]`` extra)
does. The adapters change no retrieval behavior: they reshape a
``ContextBundle``/``Receipt`` the kernel already produced into the framework's
document/node type, carrying the omnex chunks and the receipt as provenance.
"""

from __future__ import annotations
