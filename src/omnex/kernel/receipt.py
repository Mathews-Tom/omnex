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

from omnex.kernel.config import DeterminismClass


@dataclass(frozen=True, slots=True)
class Receipt:
    """The audit trail for one retrieval.

    ``returned_tokens`` is the emitted token total and ``baseline_tokens`` the
    full-document dump upper bound it is measured against. ``tiers_run`` lists the
    tiers exercised. ``model_used``/``model_version`` and ``extraction_used``
    record any opt-in lane outside the byte-exact floor. ``determinism_class`` is
    the reproducibility guarantee this run may claim. ``reference_closure_complete``
    is True only when a tier computed a reference closure and every unit in that
    closure was emitted in full (an exact set-membership fact, never a threshold);
    it is False on tiers that compute no closure.
    """

    returned_tokens: int
    baseline_tokens: int
    tiers_run: tuple[str, ...]
    model_used: bool
    model_version: str | None
    extraction_used: bool
    determinism_class: DeterminismClass
    reference_closure_complete: bool
