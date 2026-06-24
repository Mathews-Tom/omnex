"""Token-aware splitter for long prose blocks.

A long section body must be broken into retrievable units that each fit a token
budget, but a unit is the packing atom: the kernel never splits a unit further,
so the split has to land on *natural* boundaries rather than cutting mid-word.
``split_on_budget`` does exactly that. It walks a strict boundary hierarchy --
paragraph (blank line), then sentence, then word -- splitting only as finely as
needed and greedily packing adjacent pieces back together while they still fit,
so the result is the coarsest split that respects the budget.

The splitter is pure and deterministic: it owns no tokenizer and performs no
model, network, or file-system access. The caller injects an ``encode`` callable
(the ``tiktoken`` encoder the adapter already holds), so token counting stays the
adapter's single deterministic offline tokenizer and the splitter stays trivial
to test. Boundaries are returned as half-open ``(start, end)`` character ranges
into the input, trimmed of surrounding whitespace, so the caller can build exact
``Span`` values whose text round-trips from the source.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Boundary separators, coarsest first. A blank line ends a paragraph; sentence
# punctuation followed by whitespace ends a sentence; any whitespace run ends a
# word. Splitting descends this list only when a piece still exceeds the budget.
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n\s*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\s+")
_SEPARATORS: tuple[re.Pattern[str], ...] = (_PARAGRAPH_RE, _SENTENCE_RE, _WORD_RE)

Encode = Callable[[str], list[int]]


def split_on_budget(text: str, max_tokens: int, encode: Encode) -> list[tuple[int, int]]:
    """Split ``text`` into trimmed ``(start, end)`` ranges each within ``max_tokens``.

    Ranges are non-overlapping and in source order; each spans whole words on
    natural boundaries. A piece is divided only when it exceeds ``max_tokens``,
    and adjacent pieces are packed back together greedily while they still fit,
    so the split is the coarsest one the budget allows. A single word that alone
    exceeds the budget is emitted whole rather than cut. Whitespace between
    pieces is dropped from the ranges, so ``text[start:end]`` is always a clean,
    untruncated fragment of the source.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    return _split(text, 0, len(text), max_tokens, encode, 0)


def _count(text: str, start: int, end: int, encode: Encode) -> int:
    """Token count of ``text[start:end]`` under the injected encoder."""
    return len(encode(text[start:end]))


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``(start, end)`` past surrounding whitespace, preserving content."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_by(text: str, start: int, end: int, separator: re.Pattern[str]) -> list[tuple[int, int]]:
    """Divide ``text[start:end]`` at ``separator`` matches into content ranges."""
    parts: list[tuple[int, int]] = []
    pos = start
    for match in separator.finditer(text, start, end):
        if match.start() > pos:
            parts.append((pos, match.start()))
        pos = match.end()
    if pos < end:
        parts.append((pos, end))
    return parts


def _split(
    text: str, start: int, end: int, max_tokens: int, encode: Encode, level: int
) -> list[tuple[int, int]]:
    """Recursively split ``text[start:end]`` at boundary ``level`` and finer."""
    if not text[start:end].strip():
        return []
    if _count(text, start, end, encode) <= max_tokens:
        trimmed = _trim(text, start, end)
        return [trimmed] if trimmed[0] < trimmed[1] else []
    if level >= len(_SEPARATORS):
        # No finer boundary remains: this is a single over-budget word, emitted
        # whole rather than cut mid-token.
        trimmed = _trim(text, start, end)
        return [trimmed] if trimmed[0] < trimmed[1] else []
    parts = _split_by(text, start, end, _SEPARATORS[level])
    if len(parts) <= 1:
        # This boundary did not divide the range; try the next finer one.
        return _split(text, start, end, max_tokens, encode, level + 1)
    return _pack(text, parts, max_tokens, encode, level)


def _pack(
    text: str,
    parts: list[tuple[int, int]],
    max_tokens: int,
    encode: Encode,
    level: int,
) -> list[tuple[int, int]]:
    """Greedily merge adjacent ``parts`` into ranges within ``max_tokens``."""
    result: list[tuple[int, int]] = []
    group_start: int | None = None
    group_end = 0
    for part_start, part_end in parts:
        if _count(text, part_start, part_end, encode) > max_tokens:
            # A single piece is over budget on its own: flush the open group, then
            # split this piece at the next finer boundary.
            if group_start is not None:
                result.append(_trim(text, group_start, group_end))
                group_start = None
            result.extend(_split(text, part_start, part_end, max_tokens, encode, level + 1))
            continue
        if group_start is None:
            group_start, group_end = part_start, part_end
        elif _count(text, group_start, part_end, encode) <= max_tokens:
            group_end = part_end
        else:
            result.append(_trim(text, group_start, group_end))
            group_start, group_end = part_start, part_end
    if group_start is not None:
        result.append(_trim(text, group_start, group_end))
    return result
