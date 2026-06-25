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

import dataclasses
import json
from pathlib import Path
from typing import cast

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
from omnex.client_setup import (
    ALL_CLIENTS,
    ClientName,
    ClientScope,
    append_agent_guidance,
    build_client_install_plan,
    render_agent_guidance_preview,
    render_client_install_preview,
    resolve_scope,
    write_client_install_plan,
)
from omnex.doctor import render_report_text, report_to_dict, run_doctor
from omnex.metrics import recorder, settings, store, summary


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
        files = collect_files(paths)
        documents, units, references = index_corpus(files)
    except ValueError as exc:
        # Routing fails loud when a source is unclaimable or its content changed
        # since ingest; surface it as a clean CLI error, never a silent fallback.
        raise click.ClickException(str(exc)) from exc
    click.echo(f"indexed {documents} document(s), {units} unit(s), {references} reference(s)")
    recorder.record_index(surface="cli", file_count=len(files))


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
    recorder.record_query(surface="cli", receipt=receipt, bundle=bundle, file_count=len(sources))


@main.command(name="install-client")
@click.argument("client", type=click.Choice(list(ALL_CLIENTS)))
@click.argument("source", required=False, default=None)
@click.option(
    "--scope",
    type=click.Choice(["project", "user"]),
    default=None,
    help="Install scope. Default user/global; a SOURCE path or --scope project is repo-local.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview the resolved target and config without writing anything.",
)
@click.option(
    "--agent-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Append the omnex MCP guidance prompt to this agent file (CLAUDE.md, AGENTS.md).",
)
def install_client_command(
    client: str,
    source: str | None,
    scope: str | None,
    dry_run: bool,
    agent_file: Path | None,
) -> None:
    """Write the MCP client configuration that registers the omnex-mcp server.

    CLIENT is one of the supported MCP clients; SOURCE is the repo root for a
    project-scope install (defaults to the current directory). The omnex entry
    is merged into the client's existing config without clobbering unrelated
    sections, so adopters do not hand-write MCP JSON for the existing server.
    With --dry-run the resolved target and config are printed and nothing is
    written to disk. --agent-file appends ready-to-paste omnex MCP guidance to
    an agent file once (a re-run is a no-op).
    """
    agent_path = agent_file.expanduser() if agent_file is not None else None
    try:
        resolved_scope = resolve_scope(
            cast("ClientName", client),
            source,
            cast("ClientScope | None", scope),
        )
        plan = build_client_install_plan(cast("ClientName", client), source, scope=resolved_scope)
        if dry_run:
            click.echo(render_client_install_preview(plan), nl=False)
            if agent_path is not None:
                click.echo(render_agent_guidance_preview(agent_path), nl=False)
            return
        target = write_client_install_plan(plan)
        click.echo(f"Wrote {client} config: {target}")
        if agent_path is not None:
            if append_agent_guidance(agent_path):
                click.echo(f"Appended omnex MCP guidance: {agent_path}")
            else:
                click.echo(f"omnex MCP guidance already present: {agent_path}")
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@main.group(name="metrics")
def metrics_group() -> None:
    """Inspect and manage the local, off-by-default usage ledger.

    These commands are CLI-only: the MCP server exposes no metrics tools, so an
    agent can route retrieval through omnex but cannot read, change, or delete the
    operator's local metrics state. The ledger never leaves the machine.
    """


@metrics_group.command(name="enable")
@click.option("--on/--off", "on", default=True, help="Turn usage recording on (default) or off.")
def metrics_enable_command(on: bool) -> None:
    """Persist whether usage recording is on.

    OMNEX_USAGE_METRICS still overrides this per session.
    """
    settings.set_metrics_enabled(on)
    click.echo(f"Usage metrics {'enabled' if on else 'disabled'}.")


@metrics_group.command(name="trace")
@click.option("--on/--off", "on", default=True, help="Turn detailed tracing on (default) or off.")
def metrics_trace_command(on: bool) -> None:
    """Turn detailed tracing on or off -- a second opt-in, separate from recording.

    A trace stores only anonymous diagnostics (tier, determinism, recall basis,
    closure) and takes effect only when usage recording is also on. It never
    stores source or output, and OMNEX_USAGE_TRACE overrides this per session.
    """
    settings.set_trace_enabled(on)
    click.echo(f"Usage trace {'enabled' if on else 'disabled'}.")


@metrics_group.command(name="summary")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Render the summary as text or JSON.",
)
def metrics_summary_command(output_format: str) -> None:
    """Report token savings and the CLI-vs-MCP surface split from the local ledger."""
    events = store.read_events(settings.ledger_path())
    report = summary.build_summary(
        events, enabled=settings.metrics_enabled(), trace_enabled=settings.trace_enabled()
    )
    if output_format == "json":
        click.echo(json.dumps(summary.summary_to_dict(report), indent=2, sort_keys=True))
    else:
        click.echo(summary.render_summary_text(report))


@metrics_group.command(name="export")
def metrics_export_command() -> None:
    """Export every recorded event as JSON -- anonymous counters only."""
    events = store.read_events(settings.ledger_path())
    payload = {"events": [dataclasses.asdict(event) for event in events]}
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@metrics_group.command(name="delete")
@click.option("--yes", is_flag=True, default=False, help="Delete without confirmation.")
def metrics_delete_command(yes: bool) -> None:
    """Delete the local usage ledger. The enable setting is left unchanged."""
    if not yes:
        click.confirm("Delete the local usage ledger?", abort=True)
    deleted = store.delete_ledger(settings.ledger_path())
    click.echo("Deleted the local usage ledger." if deleted else "No usage ledger to delete.")


@main.command(name="doctor")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Render the report as text or JSON.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit non-zero when any check is not ok.",
)
@click.pass_context
def doctor_command(ctx: click.Context, output_format: str, strict: bool) -> None:
    """Report installation and operational health.

    Covers MCP registration, the usage-metrics ledger state, installed extras,
    adapter sanity, and the persistence mode. With ``--strict`` the command exits
    non-zero when any check is not ``ok`` -- usable as a health gate.
    """
    report = run_doctor()
    if output_format == "json":
        click.echo(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
    else:
        click.echo(render_report_text(report))
    if strict and not report.healthy:
        ctx.exit(1)
