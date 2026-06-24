"""Token-efficiency runner: compare omnex against the baselines at equal recall.

For each labeled task the runner retrieves three ways over one corpus -- omnex T1,
the pinned chunk-and-embed headline, and the full-dump upper bound -- grades each
against the gold markers, and measures tokens at the family's recall target. The
load-bearing discipline lives here:

- **Equal recall.** Every path is graded against the same gold markers and the
  token figure is ``tokens_at_recall`` at the one shared target. A path that never
  reaches the target reports ``None`` tokens, so a delta is never read at unequal
  recall.
- **omnex's operating point.** omnex T1 is queried at the smallest budget that
  yields a complete reference closure (binary-searched), the natural minimal-token
  point for the "complete reference-closure at budget" claim.
- **Pinned and recorded.** The omnex config and the pinned chunk-and-embed config
  are recorded verbatim in the artifact, along with the embedder identity and its
  determinism class. Latency is environment-dependent and kept in a separate
  section, excluded from the determinism guarantee.

Benchmark-only. Imports the product (one way); the product never imports this.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

import omnex
from omnex.bench.baselines import (
    PINNED_CHUNK_EMBED,
    Embedder,
    FastEmbedEmbedder,
    TfidfEmbedder,
    chunk_and_embed_baseline,
    full_dump_baseline,
)
from omnex.bench.metrics import (
    RetrievedItem,
    covered_labels,
    f1,
    p95_latency,
    recall,
    tokens_at_recall,
)
from omnex.bench.report import Comparison, PathResult, render_report, verdict
from omnex.bench.tasks import Family, Task, load_family
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.config import KernelConfig
from omnex.kernel.packer import Representation, count_tokens

# Path-name keys every comparison reports under.
OMNEX = "omnex_t1"
CHUNK_AND_EMBED = "chunk_and_embed"
FULL_DUMP = "full_dump"
_PATHS = (OMNEX, CHUNK_AND_EMBED, FULL_DUMP)

# The pinned omnex configuration the spec family is measured under. Closure follows
# the hard reference edges only (CONTAINS/SIBLING/CROSS_REF off), so the candidate
# set is exactly the $ref closure; recorded in the artifact. Not a product default.
SPEC_OMNEX_CONFIG = KernelConfig(
    tier="T1",
    bm25_profile={"text": 1.0, "title": 2.0, "breadcrumb": 1.5, "summary": 1.0},
    hop_budget_by_kind={
        "CONTAINS": 0,
        "SIBLING": 0,
        "CROSS_REF": 0,
        "REFERENCES": 8,
        "FOREIGN_KEY": 8,
    },
    confidence_decay=0.9,
    enable_vector_lane=False,
    enable_rerank=False,
)

# Known benchmark families and the task file each loads from.
_FAMILY_TASKS: Mapping[str, Path] = {
    "specs": Path("benchmarks/specs/tasks.json"),
    "prose": Path("benchmarks/prose/tasks.json"),
}


def make_embedder(alias: str) -> tuple[Embedder, str]:
    """Resolve an embedder alias to an embedder and its determinism class.

    ``tfidf`` is the deterministic, offline, byte-exact lexical embedder (CI and
    the reproducible artifact); ``bge-small`` is the pinned strong embedding lane,
    reproducible only with the pinned model/runtime/arch (``pinned_reproducible``).
    """
    if alias == "tfidf":
        return TfidfEmbedder(), "byte_exact"
    if alias == "bge-small":
        return FastEmbedEmbedder(PINNED_CHUNK_EMBED.embedder), "pinned_reproducible"
    raise ValueError(f"unknown embedder alias {alias!r}; use 'tfidf' or 'bge-small'")


@dataclass(frozen=True, slots=True)
class PathOutcome:
    """One retrieval path's standing on one task: tokens at recall, recall, F1."""

    tokens_at_recall: int | None
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """A task's outcomes across the three paths, plus omnex's operating point."""

    id: str
    query: str
    gold_label_count: int
    omnex_budget: int
    omnex_closure_complete: bool
    paths: Mapping[str, PathOutcome]


def _block(bundle: ContextBundle, representation: Representation) -> str:
    """Render one representation exactly as it appears in the bundle."""
    unit = bundle.units[representation.unit_id]
    return ContextBundle((representation,), {representation.unit_id: unit}).render()


def _omnex_passages(bundle: ContextBundle) -> list[str]:
    """omnex's emitted (non-SKIP) units as ranked passages, in pack order."""
    return [
        _block(bundle, representation)
        for representation in bundle.representations
        if representation.mode != "SKIP"
    ]


def _min_complete_closure_budget(corpus: Path, query: str, ceiling: int) -> int:
    """Smallest budget at which omnex T1 returns a complete reference closure.

    Under this config the candidate set is exactly the ``$ref`` closure
    (CONTAINS/SIBLING/CROSS_REF hops are zeroed), so ``reference_closure_complete``
    is monotone in budget -- a larger budget packs a superset of the same closure
    candidates -- and a binary search over ``[0, ceiling]`` finds the threshold.
    (With non-closure candidates present the greedy packer could upgrade a
    high-score distractor and starve a low-score closure unit, breaking
    monotonicity; the search would then still return a verified-complete budget,
    just not necessarily the minimum.) Fails loud if even the full-corpus budget
    does not complete the closure, rather than reporting a win at an incomplete one.
    """
    _, top = omnex.query_sources([corpus], query, ceiling, SPEC_OMNEX_CONFIG)
    if not top.reference_closure_complete:
        raise RuntimeError(f"closure never completes for query {query!r} at budget {ceiling}")
    low, high = 0, ceiling
    while low < high:
        mid = (low + high) // 2
        _, receipt = omnex.query_sources([corpus], query, mid, SPEC_OMNEX_CONFIG)
        if receipt.reference_closure_complete:
            high = mid
        else:
            low = mid + 1
    return high


def _grade(
    passages: Sequence[str],
    task: Task,
    universe: frozenset[str],
    target: float,
) -> PathOutcome:
    """Grade a ranked retrieval against one task: tokens at recall, recall, and F1.

    Tokens at recall walks the ranked passages; recall is over the whole retrieval;
    F1 measures precision against the family-wide label ``universe`` (a path that
    drags in other tasks' gold scores lower precision), so F1 surfaces the
    precision gap the token figure alone does not.
    """
    items = [
        RetrievedItem(count_tokens(text), covered_labels(text, task.markers)) for text in passages
    ]
    full_text = "\n".join(passages)
    covered_task = covered_labels(full_text, task.markers)
    covered_universe = covered_labels(full_text, universe)
    return PathOutcome(
        tokens_at_recall=tokens_at_recall(items, task.markers, target),
        recall=recall(covered_task, task.markers),
        f1=f1(covered_universe, task.markers),
    )


def run_family(family: Family, embedder_alias: str) -> dict[str, Any]:
    """Run every task in ``family`` three ways and return the artifact dict."""
    embedder, determinism = make_embedder(embedder_alias)
    corpus_text = family.corpus.read_text()
    full_dump_tokens = count_tokens(corpus_text)
    universe = frozenset[str]().union(*(task.markers for task in family.tasks))

    outcomes: list[TaskOutcome] = []
    omnex_latencies: list[float] = []
    for task in family.tasks:
        budget = _min_complete_closure_budget(family.corpus, task.query, full_dump_tokens)
        start = time.perf_counter()
        bundle, receipt = omnex.query_sources(
            [family.corpus], task.query, budget, SPEC_OMNEX_CONFIG
        )
        omnex_latencies.append(time.perf_counter() - start)
        chunks = chunk_and_embed_baseline(
            [corpus_text],
            task.query,
            embedder,
            chunk_tokens=PINNED_CHUNK_EMBED.chunk_tokens,
            chunk_overlap=PINNED_CHUNK_EMBED.chunk_overlap,
        )
        target = family.recall_target
        paths = {
            OMNEX: _grade(_omnex_passages(bundle), task, universe, target),
            CHUNK_AND_EMBED: _grade(chunks, task, universe, target),
            FULL_DUMP: _grade(full_dump_baseline([corpus_text]), task, universe, target),
        }
        outcomes.append(
            TaskOutcome(
                id=task.id,
                query=task.query,
                gold_label_count=len(task.markers),
                omnex_budget=budget,
                omnex_closure_complete=receipt.reference_closure_complete,
                paths=paths,
            )
        )
    return _build_artifact(
        family, embedder, determinism, full_dump_tokens, outcomes, omnex_latencies
    )


def _make_comparison(
    task_id: str,
    recall_target: float,
    tokens: Mapping[str, int | None],
) -> Comparison:
    return Comparison(
        task=task_id,
        recall_target=recall_target,
        subject=PathResult("omnex T1", tokens[OMNEX]),
        headline=PathResult("chunk-and-embed", tokens[CHUNK_AND_EMBED]),
        upper_bound=PathResult("full-dump", tokens[FULL_DUMP]),
    )


def _build_artifact(
    family: Family,
    embedder: Embedder,
    determinism: str,
    full_dump_tokens: int,
    outcomes: Sequence[TaskOutcome],
    omnex_latencies: Sequence[float],
) -> dict[str, Any]:
    tasks_json: list[dict[str, Any]] = []
    for outcome in outcomes:
        tokens = {key: outcome.paths[key].tokens_at_recall for key in _PATHS}
        recalls = {key: outcome.paths[key].recall for key in _PATHS}
        held_equal = all(value >= family.recall_target for value in recalls.values())
        tasks_json.append(
            {
                "id": outcome.id,
                "query": outcome.query,
                "gold_label_count": outcome.gold_label_count,
                "omnex_budget": outcome.omnex_budget,
                "omnex_closure_complete": outcome.omnex_closure_complete,
                "recall_held_equal": held_equal,
                "tokens_at_recall": tokens,
                "recall": recalls,
                "f1": {key: outcome.paths[key].f1 for key in _PATHS},
                "verdict": verdict(_make_comparison(outcome.id, family.recall_target, tokens)),
            }
        )

    def total(key: str) -> int | None:
        values = [outcome.paths[key].tokens_at_recall for outcome in outcomes]
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)

    omnex_total, headline_total = total(OMNEX), total(CHUNK_AND_EMBED)
    omnex_wins = (
        omnex_total is not None and headline_total is not None and omnex_total <= headline_total
    )
    return {
        "family": family.name,
        "generated_by": "omnex-bench",
        "recall_target": family.recall_target,
        "token_ledger": "whitespace word count (omnex.kernel.packer.count_tokens)",
        "corpus": {"file": family.corpus.name, "full_dump_tokens": full_dump_tokens},
        "omnex": {
            "tier": SPEC_OMNEX_CONFIG.tier,
            "budget_policy": "min_complete_closure",
            "bm25_profile": dict(SPEC_OMNEX_CONFIG.bm25_profile),
            "hop_budget_by_kind": dict(SPEC_OMNEX_CONFIG.hop_budget_by_kind),
            "confidence_decay": SPEC_OMNEX_CONFIG.confidence_decay,
        },
        "headline_baseline": {
            "name": CHUNK_AND_EMBED,
            "role": "headline",
            "embedder": embedder.name,
            "determinism": determinism,
            "chunk_tokens": PINNED_CHUNK_EMBED.chunk_tokens,
            "chunk_overlap": PINNED_CHUNK_EMBED.chunk_overlap,
            "rerank": PINNED_CHUNK_EMBED.rerank,
        },
        "upper_bound_baseline": {"name": FULL_DUMP, "role": "demoted_upper_bound"},
        "tasks": tasks_json,
        "totals": {
            "tokens_at_recall": {
                OMNEX: omnex_total,
                CHUNK_AND_EMBED: headline_total,
                FULL_DUMP: total(FULL_DUMP),
            },
            "omnex_at_or_below_headline_at_equal_recall": omnex_wins,
        },
        "latency": {
            "omnex_p95_seconds": p95_latency(omnex_latencies),
            "note": "environment-dependent; excluded from the determinism guarantee",
        },
    }


def write_artifact(artifact: dict[str, Any], out_dir: Path) -> Path:
    """Write ``artifact`` to ``out_dir/<family>.json`` deterministically; return the path.

    The ``latency`` section is the only non-deterministic field; everything else is
    byte-stable for a fixed corpus, config, and embedder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifact['family']}.json"
    path.write_text(json.dumps(artifact, indent=2) + "\n")
    return path


def _resolve_tasks(family: str) -> Path:
    known = _FAMILY_TASKS.get(family)
    if known is not None:
        return known
    candidate = Path(family)
    if candidate.is_file():
        return candidate
    raise click.ClickException(
        f"unknown family {family!r}; use one of {sorted(_FAMILY_TASKS)} or a task path"
    )


@click.group()
def main() -> None:
    """omnex token-efficiency benchmark."""


@main.command()
@click.option("--family", required=True, help="family name (specs/prose) or a task JSON path")
@click.option("--out", default="benchmarks/results", help="output directory for the artifact")
@click.option("--embedder", default="bge-small", type=click.Choice(["bge-small", "tfidf"]))
def run(family: str, out: str, embedder: str) -> None:
    """Run a benchmark family and write its artifact."""
    loaded = load_family(_resolve_tasks(family))
    artifact = run_family(loaded, embedder)
    path = write_artifact(artifact, Path(out))
    comparisons = [
        _make_comparison(task["id"], loaded.recall_target, task["tokens_at_recall"])
        for task in artifact["tasks"]
    ]
    click.echo(render_report(f"{loaded.name} family @ embedder={embedder}", comparisons))
    click.echo(f"\nartifact: {path}")
