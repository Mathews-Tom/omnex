"""Tests for ContextBundle: modality-aware, deterministic rendering."""

from __future__ import annotations

from omnex.ir.types import Span, Unit, UnitKind
from omnex.kernel.bundle import ContextBundle
from omnex.kernel.packer import Representation


def _unit(
    uid: str, text: str, *, title: str | None, breadcrumb: tuple[str, ...], kind: UnitKind
) -> Unit:
    return Unit(
        id=uid,
        document_id="doc:1",
        span=Span(0, max(len(text), 1)),
        text=text,
        token_count=len(text.split()),
        title=title,
        breadcrumb=breadcrumb,
        kind=kind,
        summary=None,
        protect=False,
    )


def _bundle() -> ContextBundle:
    units = {
        "p": _unit(
            "p",
            "Body of the prose section.",
            title="Sec",
            breadcrumb=("Doc", "Sec"),
            kind="SECTION",
        ),
        "c": _unit("c", "def f():\n    return 1", title="f", breadcrumb=("mod",), kind="FUNCTION"),
        "s": _unit(
            "s", "amount: integer", title="amount", breadcrumb=("API", "Payment"), kind="FIELD"
        ),
    }
    reps = (
        Representation("p", "INCLUDE", "Body of the prose section.", 5),
        Representation("c", "INCLUDE", "def f():\n    return 1", 4),
        Representation("s", "INCLUDE", "amount: integer", 2),
    )
    return ContextBundle(reps, units)


def test_render_is_modality_aware_by_kind() -> None:
    rendered = _bundle().render()
    # Prose: breadcrumb blockquote then text.
    assert "> Doc / Sec\n\nBody of the prose section." in rendered
    # Code: fenced block.
    assert "```\ndef f():\n    return 1\n```" in rendered
    # Spec: path-qualified canonical fragment, no blockquote.
    assert "API / Payment / amount:\namount: integer" in rendered


def test_render_is_byte_identical_across_calls() -> None:
    bundle = _bundle()
    assert bundle.render() == bundle.render()


def test_render_omits_skipped_representations() -> None:
    units = {"p": _unit("p", "kept text", title=None, breadcrumb=(), kind="SECTION")}
    reps = (
        Representation("p", "INCLUDE", "kept text", 2),
        Representation("gone", "SKIP", "", 0),
    )
    bundle = ContextBundle(reps, units)
    # The skipped unit is absent from the render and from the token total, and the
    # missing unit id never triggers a lookup.
    assert bundle.render() == "kept text"
    assert bundle.total_tokens == 2


def test_total_tokens_sums_non_skip_representations() -> None:
    assert _bundle().total_tokens == 11


def test_bundle_units_are_immutable_view() -> None:
    bundle = _bundle()
    # The stored mapping is a read-only view; mutation is rejected.
    try:
        bundle.units["p"] = bundle.units["s"]  # type: ignore[index]
    except TypeError:
        pass
    else:  # pragma: no cover - the view must reject mutation
        raise AssertionError("ContextBundle.units must be a read-only view")
