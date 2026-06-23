"""omnex: universal, structure-aware retrieval at a fraction of the tokens.

The top-level package intentionally exposes only the version string. Importing
``omnex`` must stay cheap and side-effect free: no model load, no network
access, and no file-system read on this import path.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
