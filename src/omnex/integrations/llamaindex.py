"""LlamaIndex retriever over an omnex ``RetrievalKernel`` (the ``[llamaindex]`` extra).

Wraps the same retrieval the library API (``omnex.query``/``query_sources``) runs
-- index once, then ``kernel.retrieve`` per query -- and maps each packed chunk to
a ``llama_index`` ``TextNode`` wrapped in a ``NodeWithScore``. The node's ``text``
is the chunk's packed text; its ``metadata`` carries the unit provenance and the
run's receipt (under ``omnex_receipt``). omnex returns a complete, ordered set
rather than similarity scores, so the node score is left unset and packed order is
preserved. Ranking, the returned set, and the receipt are exactly omnex's; this
adapter only reshapes them into LlamaIndex's type.

Requires the optional ``llama-index-core`` dependency (the ``[llamaindex]``
extra). It is never imported by ``import omnex``; importing this module without
the extra fails loud with an ``ImportError``.
"""

from __future__ import annotations

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from omnex.integrations._common import bundle_chunks, chunk_metadata, receipt_provenance
from omnex.kernel.config import KernelConfig
from omnex.kernel.kernel import RetrievalKernel


class OmnexLlamaRetriever(BaseRetriever):
    """A LlamaIndex retriever backed by a prebuilt omnex kernel.

    Build the kernel once with :func:`omnex.index` (in-memory IR) or
    :func:`omnex.index_sources` (file/directory paths), then pass it here with the
    retrieval ``config`` and a per-query token ``budget_tokens``. Each call returns
    one ``NodeWithScore`` per packed chunk, in omnex's packed order.
    """

    def __init__(
        self,
        kernel: RetrievalKernel,
        config: KernelConfig,
        budget_tokens: int = 4000,
    ) -> None:
        self._kernel = kernel
        self._config = config
        self._budget_tokens = budget_tokens
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        bundle, receipt = self._kernel.retrieve(
            query_bundle.query_str, self._budget_tokens, self._config
        )
        provenance = receipt_provenance(receipt)
        return [
            NodeWithScore(
                node=TextNode(
                    id_=chunk.unit_id,
                    text=chunk.text,
                    metadata=chunk_metadata(chunk, provenance),
                )
            )
            for chunk in bundle_chunks(bundle)
        ]
