"""ContextBundle: the rendered output of a retrieval, modality-aware by unit kind.

A bundle holds the packer's ordered representations plus the units they came from,
and renders them deterministically. Rendering is modality-aware via the IR
``Unit.kind`` (never the raw format, keeping the kernel modality-blind): prose
units render as markdown with a breadcrumb, code units render fenced, and spec
units render as a canonical path-qualified fragment. ``SKIP`` representations
carry no text and are omitted from the render but kept for receipt auditing.

The render is byte-exact: identical representations and units always produce the
same string. No model load, network, or file-system access.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from omnex.ir.types import Unit
from omnex.kernel.packer import Representation

RenderStyle = Literal["prose", "code", "spec"]

# Map each IR unit kind to a render style. This is a structural property of the
# IR, so the kernel stays modality-blind (it never inspects Document.modality or
# the raw source format).
_KIND_STYLE: Mapping[str, RenderStyle] = MappingProxyType(
    {
        "SECTION": "prose",
        "PARAGRAPH": "prose",
        "TABLE": "prose",
        "FIGURE_CAPTION": "prose",
        "FUNCTION": "code",
        "CLASS": "code",
        "OPERATION": "spec",
        "SCHEMA": "spec",
        "FIELD": "spec",
    }
)


def _render_prose(unit: Unit, text: str) -> str:
    breadcrumb = " / ".join(unit.breadcrumb)
    if breadcrumb:
        return f"> {breadcrumb}\n\n{text}"
    return text


def _render_code(unit: Unit, text: str) -> str:
    return f"```\n{text}\n```"


def _render_spec(unit: Unit, text: str) -> str:
    qualifier = " / ".join(part for part in (*unit.breadcrumb, unit.title) if part)
    if qualifier:
        return f"{qualifier}:\n{text}"
    return text


_RENDERERS = {
    "prose": _render_prose,
    "code": _render_code,
    "spec": _render_spec,
}


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """The ordered representations of a retrieval plus their source units."""

    representations: tuple[Representation, ...]
    units: Mapping[str, Unit]

    def __post_init__(self) -> None:
        # Defensive read-only copy so a bundle never aliases mutable caller state.
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    @property
    def total_tokens(self) -> int:
        """Total emitted tokens across non-SKIP representations."""
        return sum(rep.token_count for rep in self.representations if rep.mode != "SKIP")

    def render(self) -> str:
        """Render the included representations to a single deterministic string."""
        blocks: list[str] = []
        for rep in self.representations:
            if rep.mode == "SKIP":
                continue
            unit = self.units[rep.unit_id]
            style = _KIND_STYLE.get(unit.kind, "prose")
            blocks.append(_RENDERERS[style](unit, rep.text))
        return "\n\n".join(blocks)

    def dominant_style(self) -> str:
        """The most common render style among included representations.

        A coarse, content-free label -- ``"prose"``, ``"code"``, or ``"spec"`` --
        for the kind of material a retrieval returned, or ``"empty"`` when nothing
        was included. The usage-metrics layer uses it to categorize a query
        anonymously; ties break by the fixed style order so the label is
        deterministic for a fixed bundle.
        """
        counts: dict[RenderStyle, int] = {}
        for rep in self.representations:
            if rep.mode == "SKIP":
                continue
            style = _KIND_STYLE.get(self.units[rep.unit_id].kind, "prose")
            counts[style] = counts.get(style, 0) + 1
        if not counts:
            return "empty"
        order: tuple[RenderStyle, ...] = ("prose", "code", "spec")
        return max(order, key=lambda style: counts.get(style, 0))
