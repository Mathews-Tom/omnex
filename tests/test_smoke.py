"""Smoke test: the package imports and exposes a usable version string."""

from __future__ import annotations

import omnex


def test_package_imports() -> None:
    assert omnex.__name__ == "omnex"


def test_version_is_non_empty_str() -> None:
    assert isinstance(omnex.__version__, str)
    assert omnex.__version__
