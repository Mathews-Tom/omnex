"""click CLI surface over the omnex public library API.

The CLI is a thin wrapper: ``index`` routes a corpus into IR and reports its
shape, and ``query`` answers a question under a token budget. Neither command
changes retrieval ranking, the returned set, or the receipt schema -- they
delegate to :mod:`omnex.api` (via the shared :mod:`omnex._surface` helpers) and
render its results. omnex stays modality-blind here: a directory argument is
expanded to its files and each file is routed by its claiming adapter, never by
the CLI.

No model is loaded on any path; the deterministic T0 floor is the only tier the
surface drives.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from omnex import ContextBundle, Receipt, api
from omnex._surface import (
    _DEFAULT_BUDGET,
    collect_files,
    default_config,
    index_corpus,
    receipt_dict,
    result_payload,
)


def _render_json(bundle: ContextBundle, receipt: Receipt) -> str:
    """Render the bundle and receipt as a deterministic, key-sorted JSON document."""
    return json.dumps(result_payload(bundle, receipt), indent=2, sort_keys=True)


def _format_field(value: object) -> str:
    """Format a receipt field value for a Markdown row.

    Tuples and lists (e.g. ``tiers_run``) render as comma-joined values; every
    other field renders via ``str`` so the row matches the JSON value.
    """
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _render_markdown(bundle: ContextBundle, receipt: Receipt) -> str:
    """Render the bundle context followed by a human-readable receipt section.

    The receipt rows are driven from the same ``receipt_dict`` the JSON renderer
    uses, so both formats track the Receipt schema by construction and never
    drift. ``recall_limitations`` has its own section, so it is the one field
    excluded from the row list.
    """
    rows = [
        f"- {key}: {_format_field(value)}"
        for key, value in receipt_dict(receipt).items()
        if key != "recall_limitations"
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
    try:
        documents, units, references = index_corpus(collect_files(paths))
    except ValueError as exc:
        # Routing fails loud when a source is unclaimable or its content changed
        # since ingest; surface it as a clean CLI error, never a silent fallback.
        raise click.ClickException(str(exc)) from exc
    click.echo(f"indexed {documents} document(s), {units} unit(s), {references} reference(s)")


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
    sources = collect_files([corpus])
    try:
        bundle, receipt = api.query_sources(sources, question, budget, default_config())
    except ValueError as exc:
        # Same fail-loud routing errors as `index`, surfaced as a clean CLI error.
        raise click.ClickException(str(exc)) from exc
    if output_format == "json":
        click.echo(_render_json(bundle, receipt))
    else:
        click.echo(_render_markdown(bundle, receipt))
