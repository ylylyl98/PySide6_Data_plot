"""Metadata-only presentation for DRR picker rows; never changes grid identity."""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.drr_sources import DrrSource, DrrSourceGroup


_GATE_LABEL = re.compile(r"^(vbg|vtg|vbias)[ _-]*(setpoint|set|measured|meas)$", re.I)
_GATE_NAMES = {"vbg": "Vbg", "vtg": "Vtg", "vbias": "Vbias"}
_DIRECTION = {"increasing": "↑", "decreasing": "↓", "mixed": "↕", "constant": ""}


def _gate_identity(label: str) -> tuple[str, str] | None:
    match = _GATE_LABEL.fullmatch(label.strip())
    if match is None:
        return None
    base, role = match.groups()
    return base.lower(), "set" if role.lower() in {"set", "setpoint"} else "measured"


def _equal_columns(source: DrrSource, first: int, second: int) -> bool:
    """Only a complete ordered trace can certify redundant channels."""
    if not source.grid_complete or not source.gate_grid:
        return False
    if source.frame_count is None or len(source.gate_grid) != source.frame_count:
        return False
    width = len(source.gate_labels)
    return all(
        len(row) == width
        and math.isfinite(row[first])
        and math.isfinite(row[second])
        and row[first] == row[second]
        for row in source.gate_grid
    )


def _matching_pair(source: DrrSource, base: str) -> tuple[int, int] | None:
    roles: dict[str, list[int]] = {"set": [], "measured": []}
    for index, label in enumerate(source.gate_labels):
        identity = _gate_identity(label)
        if identity is not None and identity[0] == base:
            roles[identity[1]].append(index)
    if len(roles["set"]) == len(roles["measured"]) == 1:
        return roles["set"][0], roles["measured"][0]
    return None


def _column_direction(labels: tuple[str, ...], direction: str, index: int) -> str:
    # Direction strings come from the cached inspection, indexed by their
    # original labels. Keep arbitrary legacy direction text as a fallback.
    expected = labels[index].strip().casefold()
    for component in direction.split(","):
        label, separator, value = component.strip().rpartition(" ")
        if separator and label.casefold() == expected:
            return _DIRECTION.get(value.casefold(), "")
    return ""


def _gate_summary(
    labels: tuple[str, ...],
    ranges: tuple[tuple[float, float], ...],
    direction: str,
    sources: tuple[DrrSource, ...],
) -> str:
    merged: dict[int, str] = {}
    hidden: set[int] = set()
    bases = dict.fromkeys(identity[0] for label in labels if (identity := _gate_identity(label)) is not None)
    for base in bases:
        positions = [index for index, other in enumerate(labels) if _gate_identity(other) in {
            (base, "set"), (base, "measured")
        }]
        roles = {_gate_identity(labels[index])[1] for index in positions}
        if len(positions) != 2 or roles != {"set", "measured"}:
            continue
        if sources and all(
            (pair := _matching_pair(source, base)) is not None
            and _equal_columns(source, *pair)
            for source in sources
        ):
            merged[positions[0]] = _GATE_NAMES[base]
            hidden.add(positions[1])

    parts = []
    for index, original in enumerate(labels):
        if index in hidden:
            continue
        identity = _gate_identity(original)
        label = merged.get(index)
        if label is None:
            label = f"{_GATE_NAMES[identity[0]]} {identity[1]}" if identity else original.strip()
        if index < len(ranges):
            low, high = ranges[index]
            # Fixed gates should read as one value; preserve precision when
            # real set/measured differences need separate presentation.
            value = f"{low:.12g}" if low == high else f"{low:.12g}–{high:.12g}"
            label += f": {value}"
        arrow = _column_direction(labels, direction, index)
        parts.append(f"{label} {arrow}" if arrow else label)
    return "; ".join(parts) if parts else direction.strip()


def source_gate_summary(source: DrrSource) -> str:
    """Show each gate once, merging set/measured only with equal cached traces."""
    return _gate_summary(source.gate_labels, source.gate_ranges, source.gate_direction, (source,))


def group_gate_summary(group: DrrSourceGroup) -> str:
    """A group may merge a pair only when every member certifies equality."""
    return _gate_summary(group.gate_labels, group.gate_ranges, group.gate_direction, group.files)


def format_frame_count(value: int | tuple[int, int] | None) -> str:
    """Format a count or count range, including equal endpoints and zero."""
    if value is None:
        return "? frames"
    if isinstance(value, tuple):
        low, high = value
        if low != high:
            return f"{low}–{high} frames"
        value = low
    return f"{value} {'frame' if value == 1 else 'frames'}"


def source_catalog_detail(source: DrrSource) -> str:
    """Compact file detail without separate repeated direction/range lines."""
    description = " · ".join(value for value in (source.session_date, source.classification_reason) if value)
    center = "?" if source.wavelength_center_nm is None else f"{source.wavelength_center_nm:g}"
    metadata = " · ".join(value for value in (
        source_gate_summary(source), format_frame_count(source.frame_count), f"{center} nm"
    ) if value)
    return "\n".join(value for value in (description, metadata) if value)


def group_catalog_detail(group: DrrSourceGroup) -> str:
    """Compact group detail using the same gate presentation as file rows."""
    return " · ".join(value for value in (
        group_gate_summary(group), format_frame_count(group.frame_count_range)
    ) if value)
