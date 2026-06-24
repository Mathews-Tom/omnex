"""Spec benchmark family: task loading, the equal-recall runner, and the artifact.

The load-bearing assertions: recall is held equal across every compared path,
the deterministic-embedder run is byte-stable, and on the labeled spec family
omnex T1 spends no more tokens than chunk-and-embed at equal recall.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnex.bench.runner import (
    CHUNK_AND_EMBED,
    FULL_DUMP,
    OMNEX,
    run_family,
    write_artifact,
)
from omnex.bench.tasks import Family, load_family

_SPECS = Path(__file__).resolve().parents[1].parent / "benchmarks" / "specs"
_TASKS = _SPECS / "tasks.json"
_RESULTS = Path(__file__).resolve().parents[1].parent / "benchmarks" / "results"


def test_load_family_reads_the_spec_tasks() -> None:
    family = load_family(_TASKS)
    assert isinstance(family, Family)
    assert family.name == "specs"
    assert family.recall_target == 1.0
    assert family.corpus.name == "commerce_api.json"
    assert family.corpus.is_file()
    ids = [task.id for task in family.tasks]
    assert ids == ["create_payment", "dispatch_shipment", "enroll_subscriber"]


def test_task_markers_are_the_gold_label_set() -> None:
    family = load_family(_TASKS)
    payment = next(task for task in family.tasks if task.id == "create_payment")
    assert "A monetary value." in payment.markers
    assert "A postal location." in payment.markers
    assert len(payment.markers) == len(payment.gold) == 6


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_family_rejects_out_of_range_recall(tmp_path: Path) -> None:
    path = _write(tmp_path, {"family": "x", "corpus": "c.json", "recall_target": 1.5, "tasks": []})
    with pytest.raises(ValueError, match="recall_target"):
        load_family(path)


def test_load_family_rejects_empty_gold(tmp_path: Path) -> None:
    payload = {
        "family": "x",
        "corpus": "c.json",
        "recall_target": 1.0,
        "tasks": [{"id": "t", "query": "q", "gold": []}],
    }
    with pytest.raises(ValueError, match="no gold labels"):
        load_family(_write(tmp_path, payload))


def test_load_family_rejects_duplicate_markers(tmp_path: Path) -> None:
    payload = {
        "family": "x",
        "corpus": "c.json",
        "recall_target": 1.0,
        "tasks": [
            {
                "id": "t",
                "query": "q",
                "gold": [
                    {"label": "A", "marker": "dup"},
                    {"label": "B", "marker": "dup"},
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="duplicate gold markers"):
        load_family(_write(tmp_path, payload))


def _drop_latency(artifact: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in artifact.items() if key != "latency"}


def test_spec_family_runs_with_the_deterministic_embedder() -> None:
    artifact = run_family(load_family(_TASKS), "tfidf")
    assert artifact["family"] == "specs"
    assert len(artifact["tasks"]) == 3
    assert artifact["headline_baseline"]["embedder"] == "tfidf-cosine"


def test_recall_is_held_equal_across_compared_paths() -> None:
    artifact = run_family(load_family(_TASKS), "tfidf")
    for task in artifact["tasks"]:
        assert task["recall_held_equal"] is True
        recalls = task["recall"]
        assert recalls[OMNEX] == recalls[CHUNK_AND_EMBED] == recalls[FULL_DUMP] == 1.0


def test_omnex_tokens_at_most_chunk_and_embed_at_equal_recall() -> None:
    artifact = run_family(load_family(_TASKS), "tfidf")
    for task in artifact["tasks"]:
        omnex_tokens = task["tokens_at_recall"][OMNEX]
        headline_tokens = task["tokens_at_recall"][CHUNK_AND_EMBED]
        assert omnex_tokens is not None and headline_tokens is not None
        assert omnex_tokens <= headline_tokens
    assert artifact["totals"]["omnex_at_or_below_headline_at_equal_recall"] is True


def test_omnex_closure_is_complete_at_its_budget() -> None:
    artifact = run_family(load_family(_TASKS), "tfidf")
    assert all(task["omnex_closure_complete"] for task in artifact["tasks"])


def test_artifact_is_deterministic_except_for_latency() -> None:
    first = run_family(load_family(_TASKS), "tfidf")
    second = run_family(load_family(_TASKS), "tfidf")
    assert _drop_latency(first) == _drop_latency(second)


def test_write_artifact_round_trips(tmp_path: Path) -> None:
    artifact = run_family(load_family(_TASKS), "tfidf")
    path = write_artifact(artifact, tmp_path)
    assert path == tmp_path / "specs.json"
    assert json.loads(path.read_text()) == artifact


def test_checked_in_artifact_shows_the_headline_win_at_equal_recall() -> None:
    # The checked-in headline artifact is the v0 defensible number: it must show
    # omnex T1 <= chunk-and-embed at equal recall, against the pinned strong model.
    artifact = json.loads((_RESULTS / "specs.json").read_text())
    assert artifact["headline_baseline"]["embedder"] == "BAAI/bge-small-en-v1.5"
    assert artifact["headline_baseline"]["determinism"] == "pinned_reproducible"
    assert artifact["totals"]["omnex_at_or_below_headline_at_equal_recall"] is True
    for task in artifact["tasks"]:
        assert task["recall_held_equal"] is True
        assert task["recall"][OMNEX] == task["recall"][CHUNK_AND_EMBED] == 1.0
        assert task["tokens_at_recall"][OMNEX] <= task["tokens_at_recall"][CHUNK_AND_EMBED]


def test_checked_in_artifact_omnex_numbers_match_a_clean_run() -> None:
    # The omnex T1 and full-dump figures are embedder-independent, so a clean
    # tfidf run reproduces them byte-for-byte. Pinning them here means a hand-edit
    # of the checked-in metric values (the one number the headline rests on) fails
    # CI even when the inequality it asserts stays trivially true.
    artifact = json.loads((_RESULTS / "specs.json").read_text())
    fresh = run_family(load_family(_TASKS), "tfidf")
    assert artifact["corpus"]["full_dump_tokens"] == fresh["corpus"]["full_dump_tokens"]
    fresh_by_id = {task["id"]: task for task in fresh["tasks"]}
    for task in artifact["tasks"]:
        clean = fresh_by_id[task["id"]]
        assert task["omnex_budget"] == clean["omnex_budget"]
        assert task["tokens_at_recall"][OMNEX] == clean["tokens_at_recall"][OMNEX]
        assert task["tokens_at_recall"][FULL_DUMP] == clean["tokens_at_recall"][FULL_DUMP]
        assert task["recall"][OMNEX] == clean["recall"][OMNEX]
        assert task["f1"][OMNEX] == clean["f1"][OMNEX]
