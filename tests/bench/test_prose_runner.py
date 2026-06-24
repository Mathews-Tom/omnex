"""Prose benchmark family: the T0-versus-full-dump run and its honest recall.

T0 is the deterministic floor. The assertions: the prose family runs, T0 tokens
are far below full-dump at equal recall, recall is held equal in the token
comparison, and the run is deterministic. Recall itself is reported honestly --
on a semantically-distant query T0 trails embeddings and the family says so, with
no claim of beating embeddings at T0.
"""

from __future__ import annotations

from pathlib import Path

from omnex.bench.runner import FULL_DUMP, OMNEX_T0, run_prose_family
from omnex.bench.tasks import load_family

_PROSE = Path(__file__).resolve().parents[1].parent / "benchmarks" / "prose"
_TASKS = _PROSE / "tasks.json"


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
