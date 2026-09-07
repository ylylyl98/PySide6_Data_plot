"""Conservative cross-channel K/K' splitting for one matched resonance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.mcd_peak_shift import BOUNDARY_UNRELIABLE, PeakShiftResult, PeakTrack


@dataclass(frozen=True)
class ValleySplitPoint:
    field_t: float
    splitting_ev: float | None
    status: str


@dataclass(frozen=True)
class ValleySplitResult:
    method: str
    branch: str
    selected_channel: str
    counterpart_channel: str
    selected_peak_id: int | None
    counterpart_peak_id: int | None
    points: tuple[ValleySplitPoint, ...]
    status: str
    k_channel: str | None = None
    kp_channel: str | None = None


def _match_energy(track: PeakTrack) -> float | None:
    points = [point for point in track.points if point.energy_ev is not None and point.status == "tracked"]
    if not points:
        return None
    return float(min(points, key=lambda point: abs(point.field_t)).energy_ev)


def _find_track(
    analysis: PeakShiftResult,
    branch: str,
    target_energy: float,
    tolerance_ev: float,
    feature_kind: str,
    *,
    match_reference_energy: bool = True,
    allow_energy_fallback: bool = True,
) -> tuple[PeakTrack | None, str]:
    """Find one track within a branch using its true E0 when available.

    Tracks from different methods can have different field grids and their
    median measured energies are not interchangeable E0 references.  The
    fallback remains available for legacy/no-E0 analyses, but stays scoped to
    the requested branch and method analysis.
    """
    candidates = []
    for track in analysis.tracks:
        if track.branch != branch or track.quality == BOUNDARY_UNRELIABLE or track.feature_kind != feature_kind:
            continue
        if match_reference_energy:
            if track.reference_energy_ev is None and not allow_energy_fallback:
                continue
            energy = track.reference_energy_ev if track.reference_energy_ev is not None else _match_energy(track)
        else:
            energy = _match_energy(track)
        if energy is not None and abs(energy - target_energy) <= tolerance_ev:
            candidates.append((abs(energy - target_energy), track))
    if not candidates:
        return None, "unmatched"
    candidates.sort(key=lambda item: (item[0], item[1].peak_id))
    if len(candidates) > 1 and abs(candidates[1][0] - candidates[0][0]) <= 1e-4:
        return None, "ambiguous"
    return candidates[0][1], "ok"


def _sorted_track_points(track: PeakTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = sorted(track.points, key=lambda point: point.field_t)
    fields = np.asarray([point.field_t for point in points], float)
    energies = np.asarray([np.nan if point.energy_ev is None else point.energy_ev for point in points], float)
    valid = np.asarray([point.status == "tracked" and point.energy_ev is not None for point in points], bool)
    return fields, energies, valid


def _interpolate_without_gaps(fields: np.ndarray, energies: np.ndarray, valid: np.ndarray, target: float) -> float | None:
    exact = np.flatnonzero(valid & np.isclose(fields, target, rtol=0.0, atol=1e-10))
    if exact.size:
        return float(energies[int(exact[0])])
    for index in range(fields.size - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        if fields[index] < target < fields[index + 1]:
            ratio = (target - fields[index]) / (fields[index + 1] - fields[index])
            return float(energies[index] + ratio * (energies[index + 1] - energies[index]))
    return None


def compute_valley_splitting(
    analyses: dict[str, dict[str, PeakShiftResult]],
    *,
    method: str,
    selected_channel: str,
    branch: str,
    target_energy_ev: float,
    tolerance_ev: float = 0.005,
    fixed_k_channel: str | None = None,
    feature_kind: str = "peak",
    allow_energy_fallback: bool = True,
) -> ValleySplitResult:
    method_analyses = analyses.get(method, {})
    counterpart_channel = "neg" if selected_channel == "pos" else "pos"
    selected_analysis = method_analyses.get(selected_channel)
    counterpart_analysis = method_analyses.get(counterpart_channel)
    if selected_analysis is None or counterpart_analysis is None:
        return ValleySplitResult(method, branch, selected_channel, counterpart_channel, None, None, (), "unmatched")
    selected, selected_status = _find_track(selected_analysis, branch, target_energy_ev, tolerance_ev, feature_kind, allow_energy_fallback=allow_energy_fallback)
    counterpart, counterpart_status = _find_track(counterpart_analysis, branch, target_energy_ev, tolerance_ev, feature_kind, allow_energy_fallback=allow_energy_fallback)
    if selected is None or counterpart is None:
        status = selected_status if selected is None else counterpart_status
        return ValleySplitResult(method, branch, selected_channel, counterpart_channel, selected.peak_id if selected else None, counterpart.peak_id if counterpart else None, (), status)
    sf, se, sv = _sorted_track_points(selected)
    cf, ce, cv = _sorted_track_points(counterpart)
    low = max(float(np.nanmin(sf)), float(np.nanmin(cf)))
    high = min(float(np.nanmax(sf)), float(np.nanmax(cf)))
    target_fields = sorted(set(float(value) for value in np.concatenate((sf, cf)) if low <= value <= high))
    pairs: list[tuple[float, float, float]] = []
    points: list[ValleySplitPoint] = []
    for field in target_fields:
        selected_energy = _interpolate_without_gaps(sf, se, sv, field)
        counterpart_energy = _interpolate_without_gaps(cf, ce, cv, field)
        if selected_energy is None or counterpart_energy is None:
            points.append(ValleySplitPoint(field, None, "missing"))
        else:
            pairs.append((field, selected_energy, counterpart_energy))
    if fixed_k_channel in {"pos", "neg"}:
        k_channel = fixed_k_channel
    else:
        positive = [(field, selected_energy, counterpart_energy) for field, selected_energy, counterpart_energy in pairs if field > 0.0]
        if not positive:
            return ValleySplitResult(method, branch, selected_channel, counterpart_channel, selected.peak_id, counterpart.peak_id, tuple(points), "undefined positive-field ordering")
        ordering = float(np.median([selected_energy - counterpart_energy for _, selected_energy, counterpart_energy in positive]))
        if abs(ordering) <= 1e-6:
            return ValleySplitResult(method, branch, selected_channel, counterpart_channel, selected.peak_id, counterpart.peak_id, tuple(points), "ambiguous positive-field ordering")
        k_channel = selected_channel if ordering < 0.0 else counterpart_channel
    kp_channel = counterpart_channel if k_channel == selected_channel else selected_channel
    merged: list[ValleySplitPoint] = []
    for field in target_fields:
        selected_energy = _interpolate_without_gaps(sf, se, sv, field)
        counterpart_energy = _interpolate_without_gaps(cf, ce, cv, field)
        if selected_energy is None or counterpart_energy is None:
            merged.append(ValleySplitPoint(field, None, "missing"))
        else:
            selected_is_k = k_channel == selected_channel
            splitting = counterpart_energy - selected_energy if selected_is_k else selected_energy - counterpart_energy
            merged.append(ValleySplitPoint(field, splitting, "tracked"))
    return ValleySplitResult(method, branch, selected_channel, counterpart_channel, selected.peak_id, counterpart.peak_id, tuple(merged), "ok", k_channel, kp_channel)
