"""Recognize spectral-axis values in CSV column names.

The full parser (``core.processing_run``) and the header-only inspectors
(``core.drr_sources``) must agree on which columns are spectrum columns, so
both use this single recognizer.
"""
from __future__ import annotations

import math
import re
from typing import Optional

# Accepted column names:
#   "703.177", "1.0486e+00", "703.", ".5"      any finite form float() reads
#   "703.177 nm", "1.742 eV"                   optional unit suffix
#   "nm_1182.5375", "eV_1.742", "nm 703"       optional unit prefix, as written
#                                              by the multi-detector acquisition
#   "703.177.1", "nm_703.177.1"                pandas duplicate-column suffix
_SPEC_COL_RE = re.compile(
    r"^\s*(?:(?:nm|ev)[_\s]*)?"
    r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?:\.\d+)?"
    r"\s*(?:nm|ev)?\s*$",
    re.IGNORECASE,
)


def parse_spectral_column_name(col) -> Optional[float]:
    """Return the spectral value named by a column header, or ``None``."""
    match = _SPEC_COL_RE.match(str(col).strip())
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None
