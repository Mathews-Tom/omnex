"""Kernel configuration types: tiers, determinism classes, and KernelConfig.

These are pure data shapes shared by the packer and the kernel orchestration.
Behavior is selected by configuration rather than by modality-specific code
paths, so the same kernel serves every modality. The tier also fixes the
determinism class a run may claim.

No model load, network, or file-system access.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

# Retrieval tiers. T0/T1 are byte-exact and model-free; T2 adds an opt-in vector
# lane (reproducible only with a pinned model); T3 adds model extraction.
Tier = Literal["T0", "T1", "T2", "T3"]

# The reproducibility guarantee a tier may claim in the Receipt.
DeterminismClass = Literal["byte_exact", "pinned_reproducible", "model_versioned"]

# What a run's recall rests on: the lexical (FTS/BM25F) lane alone, or the
# lexical lane plus the opt-in vector lane.
RecallBasis = Literal["lexical", "lexical_plus_vector"]


@dataclass(frozen=True, slots=True)
class KernelConfig:
    """Configuration selecting kernel behavior for one retrieval run.

    ``bm25_profile`` is the per-modality BM25F column weighting,
    ``hop_budget_by_kind`` bounds graph expansion per reference kind, and
    ``confidence_decay`` weights far neighbors below near ones. ``enable_vector_lane``
    and ``enable_rerank`` are opt-in lanes outside the byte-exact floor.
    """

    tier: Tier
    bm25_profile: Mapping[str, float]
    hop_budget_by_kind: Mapping[str, int]
    confidence_decay: float
    enable_vector_lane: bool
    enable_rerank: bool

    def __post_init__(self) -> None:
        # Defensive copy behind a read-only view: a config can never be mutated
        # after construction, so "identical config -> identical output" holds even
        # if the caller later mutates the dict it passed in.
        object.__setattr__(self, "bm25_profile", MappingProxyType(dict(self.bm25_profile)))
        object.__setattr__(
            self, "hop_budget_by_kind", MappingProxyType(dict(self.hop_budget_by_kind))
        )
