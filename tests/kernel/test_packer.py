"""Tests for the efficiency packer: scoring, the representation chain, guards."""

from __future__ import annotations

import socket

import pytest

from omnex.ir.types import Span, Unit
from omnex.kernel.config import KernelConfig
from omnex.kernel.packer import (
    Candidate,
    Representation,
    count_tokens,
    pack_efficiently,
    score_candidate,
)


def _unit(
    uid: str,
    text: str,
    *,
    token_count: int | None = None,
    title: str | None = None,
    breadcrumb: tuple[str, ...] = (),
    protect: bool = False,
) -> Unit:
    return Unit(
        id=uid,
        document_id="doc:1",
        span=Span(0, max(len(text), 1)),
        text=text,
        token_count=token_count if token_count is not None else len(text.split()),
        title=title,
        breadcrumb=breadcrumb,
        kind="SECTION",
        summary=None,
        protect=protect,
    )


def _config(tier: str = "T0") -> KernelConfig:
    return KernelConfig(
        tier=tier,  # type: ignore[arg-type]
        bm25_profile={},
        hop_budget_by_kind={},
        confidence_decay=1.0,
        enable_vector_lane=False,
        enable_rerank=False,
    )


# --- scoring ---


def test_score_candidate_applies_graph_distance_penalty() -> None:
    unit = _unit("u1", "alpha beta gamma")  # token_count 3
    near = score_candidate(unit, {"u1": 9.0}, 0)
    far = score_candidate(unit, {"u1": 9.0}, 2)
    assert near == 3.0  # 9 / 3 tokens / (1 + 0)
    assert far == 1.0  # 9 / 3 tokens / (1 + 2)
    assert near > far


def test_score_candidate_prefers_smaller_unit_at_equal_relevance() -> None:
    small = _unit("s", "x y")  # 2 tokens
    big = _unit("b", "x y z w")  # 4 tokens
    assert score_candidate(small, {"s": 4.0}, 0) > score_candidate(big, {"b": 4.0}, 0)


def test_score_candidate_missing_signal_is_zero() -> None:
    assert score_candidate(_unit("u1", "alpha"), {}, 0) == 0.0


# --- packing chain ---


def test_chain_descends_include_compress_elide_skip() -> None:
    unit = _unit(
        "c",
        "alpha beta gamma delta\n\nsecond paragraph words here more more more",
        title="Cfg",
    )
    candidate = Candidate(unit, score=9.0, graph_distance=0)
    config = _config()
    assert pack_efficiently([candidate], 50, config)[0].mode == "INCLUDE"
    assert pack_efficiently([candidate], 5, config)[0].mode == "COMPRESS"
    assert pack_efficiently([candidate], 4, config)[0].mode == "ELIDE"
    assert pack_efficiently([candidate], 0, config)[0].mode == "SKIP"


def test_packed_total_stays_within_budget() -> None:
    candidates = [
        Candidate(_unit(f"u{i}", " ".join(["word"] * 5)), score=float(10 - i), graph_distance=0)
        for i in range(6)
    ]
    budget = 12
    reps = pack_efficiently(candidates, budget, _config())
    assert sum(rep.token_count for rep in reps) <= budget


def test_tight_budget_drops_far_neighbors_before_near() -> None:
    # Equal relevance and equal size: only graph distance differs, so the
    # distance penalty in score_candidate is what demotes the far neighbor.
    near_unit = _unit("near", "near relevant text here")
    far_unit = _unit("far", "far neighbor filler text")
    signals = {"near": 4.0, "far": 4.0}
    near = Candidate(near_unit, score=score_candidate(near_unit, signals, 0), graph_distance=0)
    far = Candidate(far_unit, score=score_candidate(far_unit, signals, 5), graph_distance=5)
    assert near.score > far.score
    reps = {rep.unit_id: rep.mode for rep in pack_efficiently([near, far], 4, _config())}
    assert reps["near"] == "INCLUDE"
    assert reps["far"] == "SKIP"


def test_packing_is_deterministic() -> None:
    candidates = [
        Candidate(_unit("a", "alpha beta"), score=2.0, graph_distance=0),
        Candidate(_unit("b", "gamma delta"), score=2.0, graph_distance=1),
        Candidate(_unit("c", "epsilon zeta eta"), score=5.0, graph_distance=0),
    ]
    first = pack_efficiently(candidates, 6, _config())
    second = pack_efficiently(candidates, 6, _config())
    assert first == second


# --- protect guard ---


def test_protected_unit_is_never_compressed_or_elided() -> None:
    protected = Candidate(
        _unit("p", "one two three four five six", title="P", breadcrumb=("R",), protect=True),
        score=9.0,
        graph_distance=0,
    )
    # Budget is too small for the full six-token unit, so an unprotected unit
    # would COMPRESS or ELIDE; the protected unit must SKIP instead.
    reps = pack_efficiently([protected], 3, _config())
    assert [rep.mode for rep in reps] == ["SKIP"]
    assert all(rep.mode not in ("COMPRESS", "ELIDE") for rep in reps)


def test_protected_unit_still_includes_when_it_fits() -> None:
    protected = Candidate(_unit("p", "one two three", protect=True), score=9.0, graph_distance=0)
    reps = pack_efficiently([protected], 10, _config())
    assert reps == [Representation("p", "INCLUDE", "one two three", 3)]


# --- determinism / no-model invariant ---


def test_compress_invokes_no_model_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("COMPRESS must not open a network connection or load a model file")

    # A model could load over the network (socket) or from local disk (open); a
    # deterministic COMPRESS must touch neither.
    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr("builtins.open", _forbidden)

    unit = _unit(
        "c",
        "alpha beta gamma delta\n\nsecond paragraph words here more more more",
        title="Cfg",
    )
    reps = pack_efficiently([Candidate(unit, score=9.0, graph_distance=0)], 5, _config())
    # The COMPRESS rung was exercised and produced the deterministic stub.
    assert reps[0].mode == "COMPRESS"
    assert reps[0].text == "Cfg\n\nalpha beta gamma delta"
    assert count_tokens(reps[0].text) == reps[0].token_count


# --- guards ---


def test_rejects_model_extraction_tier() -> None:
    near = Candidate(_unit("u1", "alpha"), score=1.0, graph_distance=0)
    with pytest.raises(NotImplementedError, match="model-backed extraction"):
        pack_efficiently([near], 100, _config("T3"))


def test_packs_the_t2_vector_tier_deterministically() -> None:
    # T2 only changes which candidates arrive; packing them is the same
    # deterministic, model-free chain as the byte-exact tiers.
    near = Candidate(_unit("u1", "alpha"), score=1.0, graph_distance=0)
    assert pack_efficiently([near], 100, _config("T2")) == [
        Representation("u1", "INCLUDE", "alpha", 1)
    ]


def test_rejects_negative_budget() -> None:
    near = Candidate(_unit("u1", "alpha"), score=1.0, graph_distance=0)
    with pytest.raises(ValueError, match="budget must be non-negative"):
        pack_efficiently([near], -1, _config())
