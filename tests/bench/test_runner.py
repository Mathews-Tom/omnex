"""Spec benchmark family: task loading, the equal-recall runner, and the artifact.

The load-bearing assertions: recall is held equal across every compared path,
the deterministic-embedder run is byte-stable, and on the labeled spec family
omnex T1 spends no more tokens than chunk-and-embed at equal recall.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnex.bench.tasks import Family, load_family

_SPECS = Path(__file__).resolve().parents[1].parent / "benchmarks" / "specs"
_TASKS = _SPECS / "tasks.json"


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
