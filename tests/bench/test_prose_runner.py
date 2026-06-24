"""Prose benchmark family: the T0-versus-full-dump run and its honest recall.

T0 is the deterministic floor. The assertions: the prose family runs, T0 tokens
are far below full-dump at equal recall, recall is held equal in the token
comparison, and the run is deterministic. Recall itself is reported honestly --
on a semantically-distant query T0 trails embeddings and the family says so, with
no claim of beating embeddings at T0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import omnex
from omnex.bench.metrics import covered_labels, recall
from omnex.bench.runner import (
    CHUNK_AND_EMBED,
    FULL_DUMP,
    OMNEX_T0,
    OMNEX_T2,
    PROSE_OMNEX_CONFIG,
    run_prose_family,
    run_prose_t2_family,
)
from omnex.bench.tasks import load_family

_PROSE = Path(__file__).resolve().parents[1].parent / "benchmarks" / "prose"
_TASKS = _PROSE / "tasks.json"
_RESULTS = Path(__file__).resolve().parents[1].parent / "benchmarks" / "results"


def _drop_latency(artifact: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in artifact.items() if key != "latency"}


def test_prose_family_runs() -> None:
    artifact = run_prose_family(load_family(_TASKS))
    assert artifact["family"] == "prose"
    assert artifact["tier"] == "T0"
    ids = [task["id"] for task in artifact["tasks"]]
    assert ids == ["configure_tls", "provision_storage"]


def test_t0_tokens_are_far_below_full_dump_at_equal_recall() -> None:
    artifact = run_prose_family(load_family(_TASKS))
    for task in artifact["tasks"]:
        tokens = task["tokens_at_equal_recall"]
        # equal recall: both paths reach the task's equal_recall figure.
        assert task["recall"][OMNEX_T0] == task["equal_recall"]
        assert task["recall"][FULL_DUMP] >= task["equal_recall"]
        assert task["omnex_below_full_dump_at_equal_recall"] is True
        # "far below": at least 3x fewer tokens than the full dump.
        assert tokens[OMNEX_T0] * 3 <= tokens[FULL_DUMP]


def test_recall_is_held_equal_in_the_token_comparison() -> None:
    artifact = run_prose_family(load_family(_TASKS))
    for task in artifact["tasks"]:
        assert task["tokens_at_equal_recall"][OMNEX_T0] is not None
        assert task["tokens_at_equal_recall"][FULL_DUMP] is not None


def test_prose_run_is_deterministic_except_for_latency() -> None:
    first = run_prose_family(load_family(_TASKS))
    second = run_prose_family(load_family(_TASKS))
    assert _drop_latency(first) == _drop_latency(second)


def test_recall_is_reported_honestly_without_claiming_to_beat_embeddings() -> None:
    artifact = run_prose_family(load_family(_TASKS))
    for task in artifact["tasks"]:
        assert task["recall_basis"] == "lexical"
        limitations = " ".join(task["recall_limitations"])
        assert "no claim to beat embeddings" in limitations
    note = artifact["recall_honesty"]
    assert "makes no claim to beat embeddings at T0" in note
    # T2 is now implemented: the note points to it as the lane that closes the gap,
    # labeled the weaker pinned_reproducible class rather than a parity claim.
    assert "T2" in note and "pinned_reproducible" in note


def test_semantically_distant_query_shows_a_lexical_recall_ceiling() -> None:
    artifact = run_prose_family(load_family(_TASKS))
    tls = next(task for task in artifact["tasks"] if task["id"] == "configure_tls")
    # T0 misses the semantically-distant page: recall below 1.0, honestly reported.
    assert tls["recall"][OMNEX_T0] < 1.0
    assert tls["reaches_full_recall"] is False
    assert tls["recall"][FULL_DUMP] == 1.0


def test_checked_in_prose_artifact_shows_floor_win_and_honest_recall() -> None:
    artifact = json.loads((_RESULTS / "prose.json").read_text())
    fresh = run_prose_family(load_family(_TASKS))
    # The byte-exact T0 portion (T0 + full-dump, no embedder) must match a clean run
    # exactly, modulo the latency section and the pinned-reproducible t2 section,
    # which depends on the embedding model and is checked separately.
    t0_artifact = {key: value for key, value in artifact.items() if key != "t2"}
    assert _drop_latency(t0_artifact) == _drop_latency(fresh)
    assert "makes no claim to beat embeddings at T0" in artifact["recall_honesty"]
    for task in artifact["tasks"]:
        tokens = task["tokens_at_equal_recall"]
        assert tokens[OMNEX_T0] * 3 <= tokens[FULL_DUMP]


def test_lexical_ceiling_is_genuine_not_a_budget_artifact() -> None:
    # The distant securing-traffic page shares no vocabulary with the configure_tls
    # query, so it is never a lexical candidate: T0 cannot reach full recall at ANY
    # budget. This is what makes the "T2 closes the gap" framing honest rather than
    # a consequence of the chosen floor budget.
    family = load_family(_TASKS)
    tls = next(task for task in family.tasks if task.id == "configure_tls")
    sources = sorted(family.corpus.glob("*.md"))
    for budget in (200, 1000, 10000):
        bundle, _ = omnex.query_sources(sources, tls.query, budget, PROSE_OMNEX_CONFIG)
        achieved = recall(covered_labels(bundle.render(), tls.markers), tls.markers)
        assert achieved < 1.0, f"T0 unexpectedly reached full recall at budget {budget}"


def test_checked_in_t2_section_closes_recall_at_competitive_tokens() -> None:
    # Reads the checked-in artifact (no embedder needed): the recorded t2 section
    # proves the opt-in vector lane reaches the recall target the lexical floor
    # cannot, at no more tokens than the strong chunk-and-embed headline.
    artifact = json.loads((_RESULTS / "prose.json").read_text())
    t2 = artifact["t2"]
    assert t2["determinism_class"] == "pinned_reproducible"
    assert t2["model_provenance"]["model"]
    for task in t2["tasks"]:
        assert task["omnex_t2_reaches_target"] is True
        assert task["recall"][OMNEX_T2] >= task["recall_target"]
        tokens = task["tokens_at_equal_recall"]
        assert tokens[OMNEX_T2] <= tokens[CHUNK_AND_EMBED]
        assert task["omnex_t2_at_or_below_chunk_embed_at_equal_recall"] is True
    assert t2["totals"]["omnex_t2_at_or_below_chunk_embed_at_equal_recall"] is True


def test_t2_section_closes_recall_above_the_t0_ceiling() -> None:
    # The configure_tls task has a genuine T0 lexical ceiling below 1.0; the T2 lane
    # closes it to the recall target, which is the whole point of the stack.
    artifact = json.loads((_RESULTS / "prose.json").read_text())
    t0_tls = next(t for t in artifact["tasks"] if t["id"] == "configure_tls")
    t2_tls = next(t for t in artifact["t2"]["tasks"] if t["id"] == "configure_tls")
    assert t0_tls["recall"][OMNEX_T0] < 1.0  # lexical floor cannot reach it
    assert t2_tls["recall"][OMNEX_T2] >= t2_tls["recall_target"]  # T2 does


def test_t2_prose_run_reaches_target_at_competitive_tokens() -> None:
    pytest.importorskip("fastembed")
    # A live run: the vector lane reaches the recall target and spends no more
    # tokens than chunk-and-embed at equal recall (a property robust across
    # architectures, unlike the exact pinned-reproducible token counts).
    t2 = run_prose_t2_family(load_family(_TASKS))
    assert t2["determinism_class"] == "pinned_reproducible"
    for task in t2["tasks"]:
        assert task["recall"][OMNEX_T2] >= task["recall_target"]
        tokens = task["tokens_at_equal_recall"]
        assert tokens[OMNEX_T2] is not None and tokens[CHUNK_AND_EMBED] is not None
        assert tokens[OMNEX_T2] <= tokens[CHUNK_AND_EMBED]
