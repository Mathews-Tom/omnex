"""Receipt: the auditable record of a retrieval run.

The receipt makes every run inspectable: how many tokens were returned against
the full-dump baseline, which tiers ran, whether a model or extraction was
invoked (and its version), and which determinism class the run may claim. For a
T0 run the receipt reports ``byte_exact`` determinism and zero model use, so the
deterministic headline never bleeds across tiers.

A receipt is a frozen value: two runs with the same inputs produce equal,
byte-identical receipts. No model load, network, or file-system access.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnex.kernel.config import DeterminismClass, EmbeddingProvenance, RecallBasis


@dataclass(frozen=True, slots=True)
class Receipt:
    """The audit trail for one retrieval.

    ``returned_tokens`` is the emitted token total measured against
    ``baseline_tokens``, the full-dump upper bound. The modality-blind kernel
    sets it to the sum of every indexed unit's text (the naive dump-every-unit
    cost); the source-routed ``query_sources`` refines it to the true
    whole-document dump. ``tiers_run`` lists the
    tiers exercised. ``model_used``/``model_version`` and ``extraction_used``
    record any opt-in lane outside the byte-exact floor. ``determinism_class`` is
    the reproducibility guarantee this run may claim. ``reference_closure_complete``
    is True only when a tier computed a reference closure and every unit in that
    closure was emitted in full (an exact set-membership fact, never a threshold);
    it is False on a tier that computes no closure and on an empty closure (no
    reachable units, e.g. no lexical seeds). ``recall_basis`` records what recall
    rested on -- the lexical lane alone, or the lexical plus vector lane -- so a
    reader never mistakes a lexical-only run for one with semantic recall; the
    plain-language caveats follow from it in :attr:`recall_limitations`.
    ``embedding_provenance`` is set only on a pinned-reproducible (T2) run and
    records the model, tokenizer, runtime, and architecture its embeddings depend
    on; it is None on the byte-exact tiers, which load no model.
    """

    returned_tokens: int
    baseline_tokens: int
    tiers_run: tuple[str, ...]
    model_used: bool
    model_version: str | None
    extraction_used: bool
    determinism_class: DeterminismClass
    reference_closure_complete: bool
    recall_basis: RecallBasis
    embedding_provenance: EmbeddingProvenance | None = None

    @property
    def recall_limitations(self) -> tuple[str, ...]:
        """Plain-language recall caveats implied by ``recall_basis``.

        For a lexical-only run these state that recall is lexical, so a
        semantically distant unit can be missed unless a structural edge such as
        CROSS_REF reaches it, and that lexical recall trails embedding-based
        retrieval where query and content vocabulary diverge. An empty tuple means
        no recall caveat applies (the vector lane contributed semantic recall).
        """
        if self.recall_basis == "lexical":
            return (
                "Recall is lexical-only: a semantically distant unit can be missed "
                "unless a structural edge such as CROSS_REF reaches it.",
                "Lexical recall trails embedding-based retrieval where query and "
                "content vocabulary diverge; this run makes no claim to beat embeddings.",
            )
        return ()
