"""LangChain retriever over an omnex ``RetrievalKernel`` (the ``[langchain]`` extra).

Wraps the same retrieval the library API (``omnex.query``/``query_sources``) runs
-- index once, then ``kernel.retrieve`` per query -- and maps each packed chunk to
a ``langchain_core`` ``Document``. The document's ``page_content`` is the chunk's
packed text; its ``metadata`` carries the unit provenance and the run's receipt
(under ``omnex_receipt``). Ranking, the returned set, and the receipt are exactly
omnex's; this adapter only reshapes them into LangChain's type.

Requires the optional ``langchain-core`` dependency (the ``[langchain]`` extra).
It is never imported by ``import omnex``; importing this module without the extra
fails loud with an ``ImportError``.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from omnex.integrations._common import bundle_chunks, chunk_metadata, receipt_provenance
from omnex.kernel.config import KernelConfig
from omnex.kernel.kernel import RetrievalKernel


class OmnexRetriever(BaseRetriever):
    """A LangChain retriever backed by a prebuilt omnex kernel.

    Build the kernel once with :func:`omnex.index` (in-memory IR) or
    :func:`omnex.index_sources` (file/directory paths), then pass it here with the
    retrieval ``config`` and a per-query token ``budget_tokens``. Each call returns
    one ``Document`` per packed chunk, in omnex's packed order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kernel: RetrievalKernel
    config: KernelConfig
    budget_tokens: int = 4000

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        bundle, receipt = self.kernel.retrieve(query, self.budget_tokens, self.config)
        provenance = receipt_provenance(receipt)
        return [
            Document(page_content=chunk.text, metadata=chunk_metadata(chunk, provenance))
            for chunk in bundle_chunks(bundle)
        ]
