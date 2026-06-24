"""Benchmark baselines: the demoted full-dump upper bound.

A baseline turns a corpus and a query into a *ranked list of passage texts*, the
shape the runner grades uniformly (count its tokens, check which gold labels it
covers) regardless of how the passages were produced. This module defines the
full-document dump, the naive paste-everything baseline.

The full dump is the *upper bound*, not the headline. It reaches full recall
trivially -- it returns the entire corpus -- so it only bounds how wasteful token
spend can be; it never demonstrates a competitive win. The realistic headline
baseline (chunk-and-embed) lands in this module alongside it. The two-number
honesty framing in :mod:`omnex.bench.report` keeps the upper bound demoted and
the realistic baseline as the headline.

Benchmark-only. Nothing under ``omnex.kernel`` or ``omnex.adapters`` imports this
package.
"""

from __future__ import annotations

from collections.abc import Sequence


def full_dump_baseline(documents: Sequence[str]) -> list[str]:
    """Return the whole corpus as a single ranked passage: paste everything.

    Joining every document into one passage models the naive prompt that dumps
    the full corpus into the context window. Graded against any gold set it
    reaches recall ``1.0`` (the corpus contains everything), which is exactly why
    it is the upper bound and not a competitor: it bounds token waste, it does not
    win. Returned as a one-element list so the runner grades it through the same
    path as a chunked retrieval.
    """
    return ["\n".join(documents)]
