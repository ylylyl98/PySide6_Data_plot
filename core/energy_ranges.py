"""Lightweight parsing for energy-range controls shared by DRR and MCD."""
from __future__ import annotations

import math


def parse_energy_ranges(text):
    regions = []
    for part in text.split(";"):
        if not part.strip():
            continue
        pair = tuple(float(value) for value in part.replace(",", " ").split())
        if len(pair) != 2 or not all(math.isfinite(value) for value in pair) or pair[0] >= pair[1]:
            raise ValueError("Use increasing finite energy pairs: 1.56 1.64; 1.69 1.77")
        regions.append(pair)
    return tuple(regions)
