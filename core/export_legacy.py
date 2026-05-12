"""Compatibility wrapper for the active export API.

The Qt app now imports from :mod:`core.export`. This module remains so older
scripts that imported ``core.export_legacy`` keep working.
"""

from __future__ import annotations

from core.export import *  # noqa: F401,F403

