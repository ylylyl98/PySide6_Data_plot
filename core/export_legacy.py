"""Compatibility wrapper for the active export API.

The Qt app now imports from :mod:`core.export`. This module remains so older
scripts that imported ``core.export_legacy`` keep working.
"""

from __future__ import annotations

from core.export import *  # noqa: F401,F403



_SHG_TYPE_NAMES = frozenset({
    "ShgProcessResult", "ShgSettings", "ShgSweepData",
    "ShgAngularFitResult", "ShgTwistFitResult",
})

# The old wildcard wrapper exposed every public name from core.export,
# including these formerly eager aliases. Constructing the names alone keeps
# ordinary imports lazy; only an explicit star import resolves the aliases.
__all__ = [name for name in globals() if not name.startswith("_")] + sorted(_SHG_TYPE_NAMES)


def __getattr__(name):
    """Retain the former SHG type aliases without eagerly importing SHG."""
    if name in _SHG_TYPE_NAMES:
        from core import export
        return getattr(export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
