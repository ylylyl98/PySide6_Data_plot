"""Compact, locale-aware display text without quantizing model values."""
from __future__ import annotations

import math

from PySide6.QtCore import QLocale


def display_text(value: float, key: str, locale: QLocale | None = None) -> str:
    locale = locale or QLocale()
    value = float(value)
    if not math.isfinite(value):
        return locale.toString(value)
    if value == 0:
        return "0"
    decimals = (4 if key.startswith(("axis.x", "boundary.")) else
                3 if key.startswith("axis.y") or key == "gate" else None)
    magnitude = abs(value)
    if magnitude < 1e-3 or magnitude >= 1e6:
        text = locale.toString(value, "e", 4)
    elif decimals is not None:
        text = locale.toString(value, "f", decimals)
    else:
        text = locale.toString(value, "g", 5)
    # QLocale can use a localized exponent marker and decimal separator.
    marker = locale.exponential()
    parts = text.casefold().split(marker.casefold(), 1)
    if locale.decimalPoint() in parts[0]:
        parts[0] = parts[0].rstrip(locale.zeroDigit()).removesuffix(locale.decimalPoint())
    return marker.casefold().join(parts)
