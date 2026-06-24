"""click CLI surface over the omnex public library API.

The CLI is a thin wrapper: ``index`` routes a corpus into IR and reports its
shape, and ``query`` (added alongside this group) answers a question under a
token budget. Neither command changes retrieval ranking, the returned set, or
the receipt schema -- they delegate to :mod:`omnex.api` and render its results.
omnex stays modality-blind here: a directory argument is expanded to its files
and each file is routed by its claiming adapter, never by the CLI.

No model is loaded on any path; the deterministic T0 floor is the only tier the
surface drives.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

import click

from omnex import ContextBundle, KernelConfig, Receipt, api

# Default token budget for a query when ``--budget`` is omitted. A query is a
# budgeted retrieval, so the surface always passes a concrete budget through to
# the kernel rather than leaving it implicit.
_DEFAULT_BUDGET = 4000


def default_config() -> KernelConfig:
    """The CLI's fixed retrieval config: the byte-exact, model-free T0 floor.

    The config is modality-agnostic: the BM25F profile names every indexed FTS
    column and the hop budget names every reference kind an adapter can emit, so
    one config serves prose and spec corpora identically. The surface never
    picks a tier or tunes ranking -- the T0 floor is the documented default and
    keeps every CLI run byte-exact and model-free. Exposed (not inlined) so a
    caller, and the parity tests, can drive the library with the exact config the
    CLI uses.
    """
    return KernelConfig(
        tier="T0",
        bm25_profile={"title": 2.0, "breadcrumb": 1.5, "text": 1.0, "summary": 1.0},
        hop_budget_by_kind={
            "CONTAINS": 2,
            "SIBLING": 0,
            "CROSS_REF": 1,
            "CITES": 1,
            "LINKS_TO": 1,
            "REFERENCES": 1,
            "FOREIGN_KEY": 1,
            "IMPORTS": 1,
            "CALLS": 1,
        },
        confidence_decay=0.8,
        enable_vector_lane=False,
        enable_rerank=False,
    )


def _collect_files(paths: Iterable[Path]) -> list[Path]:
    """Expand each path to files: a file is itself, a directory its sorted files.

    Sorting the files of a directory makes the routed order -- and therefore the
    packed output -- stable for a fixed corpus, which the deterministic-output
    contract requires. Files are kept in the caller's given order.
    """
    collected: list[Path] = []
    for path in paths:
        if path.is_dir():
            collected.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            collected.append(path)
    return collected


def _receipt_dict(receipt: Receipt) -> dict[str, object]:
    """The receipt as a JSON-serializable mapping, with its recall caveats.

    The receipt schema is unchanged: this serializes its fields verbatim and
    appends the derived ``recall_limitations`` so the rendered audit trail
    carries the same honesty caveats the library exposes.
    """
    data: dict[str, object] = dataclasses.asdict(receipt)
    data["recall_limitations"] = list(receipt.recall_limitations)
    return data


def _render_json(bundle: ContextBundle, receipt: Receipt) -> str:
    """Render the bundle and receipt as a deterministic, key-sorted JSON document."""
    payload = {
        "bundle": {
            "context": bundle.render(),
            "total_tokens": bundle.total_tokens,
            "representations": [
                {"unit_id": rep.unit_id, "mode": rep.mode, "token_count": rep.token_count}
                for rep in bundle.representations
            ],
        },
        "receipt": _receipt_dict(receipt),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _render_markdown(bundle: ContextBundle, receipt: Receipt) -> str:
    """Render the bundle context followed by a human-readable receipt section."""
    rows = [
        f"- returned_tokens: {receipt.returned_tokens}",
        f"- baseline_tokens: {receipt.baseline_tokens}",
        f"- tiers_run: {', '.join(receipt.tiers_run)}",
        f"- determinism_class: {receipt.determinism_class}",
        f"- model_used: {receipt.model_used}",
        f"- model_version: {receipt.model_version}",
        f"- extraction_used: {receipt.extraction_used}",
        f"- reference_closure_complete: {receipt.reference_closure_complete}",
        f"- recall_basis: {receipt.recall_basis}",
    ]
    blocks = [bundle.render(), "## Receipt", "\n".join(rows)]
    if receipt.recall_limitations:
        caveats = "\n".join(f"- {item}" for item in receipt.recall_limitations)
        blocks.append("### Recall limitations\n\n" + caveats)
    return "\n\n".join(block for block in blocks if block)


@click.group()
@click.version_option(package_name="omnex")
def main() -> None:
    """omnex: universal, structure-aware retrieval at a fraction of the tokens."""


@main.command(name="index")
@click.argument(
    "paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
def index_command(paths: tuple[Path, ...]) -> None:
    """Ingest, parse, and link PATHS into IR and report the indexed corpus shape.

    Each source is routed through its claiming adapter -- failing loud when none
    claims it -- then built into the FTS index and StructureGraph to validate the
    full index path. No state is persisted; the command reports the corpus shape
    (documents, units, references) it would index.
    """
    sources = _collect_files(paths)
    units, references, documents = api._route_sources(sources)
    # Build the index and graph so a corpus that routes but cannot be indexed
    # (e.g. a malformed edge) fails here rather than silently at query time.
    api.index(units, references)
    click.echo(
        f"indexed {len(documents)} document(s), {len(units)} unit(s), "
        f"{len(references)} reference(s)"
    )


@main.command(name="query")
@click.argument("corpus", type=click.Path(exists=True, path_type=Path))
@click.argument("question")
@click.option(
    "--budget",
    type=int,
    default=_DEFAULT_BUDGET,
    show_default=True,
    help="Token budget the packed context must fit within.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"]),
    default="markdown",
    show_default=True,
    help="Render the ContextBundle and Receipt as JSON or Markdown.",
)
def query_command(corpus: Path, question: str, budget: int, output_format: str) -> None:
    """Answer QUESTION over CORPUS under a token budget and render the result.

    Routes CORPUS through its adapters and runs the same T0 kernel pipeline the
    library does, then renders the ContextBundle and Receipt in the chosen
    format. The retrieval, ranking, and returned set are exactly the library's;
    the CLI only renders them, so output is deterministic for a fixed corpus,
    question, and budget.
    """
    sources = _collect_files([corpus])
    bundle, receipt = api.query_sources(sources, question, budget, default_config())
    if output_format == "json":
        click.echo(_render_json(bundle, receipt))
    else:
        click.echo(_render_markdown(bundle, receipt))
