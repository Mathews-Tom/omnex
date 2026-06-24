"""Labeled benchmark task families: load the checked-in task JSON into typed data.

A *family* is a corpus plus a set of labeled retrieval *tasks*. Each task carries
a query and its gold labels; a gold label is a human-readable name and the
distinctive *marker* -- the substring identifying that relevant unit's definition
in the corpus. The marker is the label the metrics grade against, so the same
marker serves omnex and every baseline identically.

The task file is the only place labels live; the runner owns the pinned method
(omnex config and the chunk-and-embed config), so labels and method stay
separable and the labels are auditable on their own. Benchmark-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GoldLabel:
    """One relevant unit: a human-readable ``label`` and its grading ``marker``."""

    label: str
    marker: str


@dataclass(frozen=True, slots=True)
class Task:
    """A labeled retrieval task: a query and the gold closure it must recall."""

    id: str
    query: str
    gold: tuple[GoldLabel, ...]

    @property
    def markers(self) -> frozenset[str]:
        """The gold markers, the label set the metrics grade recall against."""
        return frozenset(label.marker for label in self.gold)


@dataclass(frozen=True, slots=True)
class Family:
    """A corpus, an equal recall target, and the labeled tasks graded against it."""

    name: str
    corpus: Path
    recall_target: float
    tasks: tuple[Task, ...]


def load_family(path: Path) -> Family:
    """Load a family from its task JSON, resolving the corpus path relative to it.

    Fails loud on a missing field, an empty gold set, a duplicate marker within a
    task (which would make recall un-gradeable), or a recall target outside
    ``[0, 1]`` -- a malformed family must never silently grade as a passing one.
    """
    data: Any = json.loads(path.read_text())
    recall_target = float(data["recall_target"])
    if not 0.0 <= recall_target <= 1.0:
        raise ValueError(f"recall_target must be in [0.0, 1.0], got {recall_target}")
    tasks: list[Task] = []
    for entry in data["tasks"]:
        gold = tuple(GoldLabel(item["label"], item["marker"]) for item in entry["gold"])
        if not gold:
            raise ValueError(f"task {entry['id']!r} has no gold labels")
        markers = [label.marker for label in gold]
        if len(set(markers)) != len(markers):
            raise ValueError(f"task {entry['id']!r} has duplicate gold markers")
        tasks.append(Task(entry["id"], entry["query"], gold))
    if not tasks:
        raise ValueError(f"family {path} has no tasks")
    corpus = (path.parent / data["corpus"]).resolve()
    return Family(data["family"], corpus, recall_target, tuple(tasks))
