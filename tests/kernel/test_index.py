"""Tests for the FTS5 BM25F index: ranking, profiles, determinism, safety."""

from __future__ import annotations

from omnex.ir.types import Span, Unit
from omnex.kernel.index import FtsIndex


def _unit(
    uid: str,
    *,
    text: str = "",
    title: str | None = None,
    breadcrumb: tuple[str, ...] = (),
    summary: str | None = None,
) -> Unit:
    return Unit(
        id=uid,
        document_id="doc:1",
        span=Span(0, 1),
        text=text,
        token_count=1,
        title=title,
        breadcrumb=breadcrumb,
        kind="SECTION",
        summary=summary,
        protect=False,
    )


def test_search_returns_matching_unit_ids() -> None:
    index = FtsIndex()
    index.index_units(
        [
            _unit("u1", text="alpha beta gamma"),
            _unit("u2", text="delta epsilon"),
            _unit("u3", text="gamma zeta"),
        ]
    )
    ids = [uid for uid, _ in index.search("gamma", {"text": 1.0}, limit=10)]
    assert set(ids) == {"u1", "u3"}
    assert "u2" not in ids


def test_changing_profile_changes_ranking() -> None:
    index = FtsIndex()
    index.index_units(
        [
            _unit("u_text", text="tls configuration body words padding padding"),
            _unit("u_title", text="generic body words padding padding", title="tls"),
        ]
    )
    title_heavy = {"text": 0.1, "title": 10.0}
    text_heavy = {"text": 10.0, "title": 0.1}

    title_first = [uid for uid, _ in index.search("tls", title_heavy, limit=10)]
    text_first = [uid for uid, _ in index.search("tls", text_heavy, limit=10)]

    assert title_first[0] == "u_title"
    assert text_first[0] == "u_text"
    assert title_first != text_first


def test_identical_corpus_and_query_is_deterministic() -> None:
    units = [
        _unit("u1", text="alpha beta", title="header"),
        _unit("u2", text="beta gamma", summary="beta summary"),
        _unit("u3", text="beta", breadcrumb=("root", "child")),
    ]
    # Index the second instance from a reordered copy: result order must depend
    # only on the corpus and query, not on insertion order.
    first = FtsIndex()
    first.index_units(units)
    second = FtsIndex()
    second.index_units(list(reversed(units)))
    profile = {"text": 1.0, "title": 2.0, "breadcrumb": 1.0, "summary": 1.5}
    assert first.search("beta", profile, limit=10) == second.search("beta", profile, limit=10)


def test_score_ties_break_by_unit_id() -> None:
    index = FtsIndex()
    # Identical content under different ids forces equal BM25 scores.
    body = "tls tls payload words common common"
    index.index_units(
        [
            _unit("unit:zzz", text=body),
            _unit("unit:aaa", text=body),
            _unit("unit:mmm", text=body),
        ]
    )
    result = index.search("tls", {"text": 1.0}, limit=10)
    ids = [uid for uid, _ in result]
    scores = {round(score, 12) for _, score in result}
    assert len(scores) == 1  # all tied on score
    assert ids == ["unit:aaa", "unit:mmm", "unit:zzz"]


def test_unicode_and_fts_special_characters_are_safe() -> None:
    index = FtsIndex()
    index.index_units(
        [
            _unit("u_cjk", text="設定 トークン 日本語 本文"),
            _unit("u_accent", text="naïve café configuration"),
        ]
    )
    profile = {"text": 1.0}
    # Unicode word tokens match their units.
    assert [uid for uid, _ in index.search("日本語", profile, limit=10)] == ["u_cjk"]
    assert [uid for uid, _ in index.search("café", profile, limit=10)] == ["u_accent"]
    # FTS operators and stray punctuation are treated as literal terms, never
    # parsed as syntax: these must not raise. The first still matches via "café";
    # the rest have no indexed token and return empty.
    assert index.search('café AND "missing (', profile, limit=10)
    assert index.search("NEAR(unbalanced", profile, limit=10) == []
    assert index.search("***", profile, limit=10) == []


def test_reindexing_a_unit_is_idempotent() -> None:
    index = FtsIndex()
    index.index_units([_unit("u1", text="alpha beta")])
    index.index_units([_unit("u1", text="alpha beta")])
    result = index.search("alpha", {"text": 1.0}, limit=10)
    assert [uid for uid, _ in result] == ["u1"]


def test_zero_or_negative_limit_returns_empty() -> None:
    index = FtsIndex()
    index.index_units([_unit("u1", text="alpha")])
    # The limit guard must reject non-positive limits outright; a narrowed
    # ``== 0`` check would let ``scored[:-1]`` silently drop the last result.
    assert index.search("alpha", {"text": 1.0}, limit=0) == []
    assert index.search("alpha", {"text": 1.0}, limit=-1) == []
