"""Deterministic reflection-peak tracking for MCD B sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, savgol_filter


@dataclass(frozen=True)
class PeakCandidate:
    energy_ev: float
    prominence: float
    width_points: float


@dataclass(frozen=True)
class PeakPoint:
    field_t: float
    branch: str
    energy_ev: float | None
    delta_energy_ev: float | None
    status: str


@dataclass(frozen=True)
class PeakTrack:
    peak_id: int
    branch: str
    points: tuple[PeakPoint, ...]
    reference_energy_ev: float | None
    reference_field_t: float | None
    reference_method: str


@dataclass(frozen=True)
class PeakShiftResult:
    fields_t: np.ndarray
    branches: np.ndarray
    candidates: tuple[tuple[PeakCandidate, ...], ...]
    tracks: tuple[PeakTrack, ...]
    source: str


SOURCE_LABELS = {
    "corrected average": ("pair_corrected_pos", "pair_corrected_neg"),
    "corrected pos": ("pair_corrected_pos",), "corrected neg": ("pair_corrected_neg",),
    "raw average": ("pair_raw_pos", "pair_raw_neg"), "raw pos": ("pair_raw_pos",), "raw neg": ("pair_raw_neg",),
}


def source_spectra(result: Any, source: str = "corrected average") -> np.ndarray:
    names = SOURCE_LABELS.get(source.casefold().strip())
    if names is None: raise ValueError(f"Unknown reflection source: {source}")
    arrays = [np.asarray(getattr(result, name), dtype=float) for name in names]
    if any(a.ndim != 2 for a in arrays) or any(a.shape != arrays[0].shape for a in arrays[1:]):
        raise ValueError("MCD reflection source arrays must have matching two-dimensional shapes.")
    return np.nanmean(np.stack(arrays), axis=0)


def _refine_quadratic(x: np.ndarray, y: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(x) - 1: return float(x[index])
    try:
        a, b, _ = np.polyfit(x[index - 1:index + 2], y[index - 1:index + 2], 2)
        vertex = -b / (2.0 * a)
        return float(vertex) if np.isfinite(vertex) and x[index - 1] <= vertex <= x[index + 1] else float(x[index])
    except (ValueError, np.linalg.LinAlgError): return float(x[index])


def detect_reflection_peaks(energy_ev: np.ndarray, spectrum: np.ndarray, *, prominence_fraction: float = 0.03, min_distance_points: int = 5, smoothing_points: int = 7, max_peaks: int = 8) -> tuple[PeakCandidate, ...]:
    x0, y0 = np.asarray(energy_ev, float).ravel(), np.asarray(spectrum, float).ravel(); valid = np.isfinite(x0) & np.isfinite(y0)
    if valid.sum() < 5: return ()
    order = np.argsort(x0[valid]); x, y = x0[valid][order], y0[valid][order]
    if np.any(np.diff(x) <= 0): return ()
    window = min(int(smoothing_points) | 1, len(y) if len(y) % 2 else len(y) - 1)
    smooth = savgol_filter(y, window, min(2, window - 1), mode="interp") if window >= 5 else y
    signal = smooth - np.linspace(smooth[0], smooth[-1], len(smooth)); span = float(np.nanpercentile(signal, 98) - np.nanpercentile(signal, 2))
    indices, props = find_peaks(signal, prominence=max(0.0, prominence_fraction) * max(span, np.finfo(float).eps), distance=max(1, int(min_distance_points)))
    ranked = sorted(indices.tolist(), key=lambda i: (-float(props["prominences"][np.where(indices == i)[0][0]]), i))[:max(1, int(max_peaks))]; ranked.sort()
    return tuple(PeakCandidate(_refine_quadratic(x, signal, i), float(props["prominences"][np.where(indices == i)[0][0]]), float(props.get("widths", np.full(len(indices), np.nan))[np.where(indices == i)[0][0]])) for i in ranked)


def _reference(energies: np.ndarray, fields: np.ndarray, tolerance: float) -> tuple[float | None, float | None, str]:
    finite = np.flatnonzero(np.isfinite(energies))
    if finite.size == 0: return None, None, "unavailable"
    exact = finite[np.abs(fields[finite]) <= tolerance]
    if exact.size:
        i = int(exact[np.argmin(np.abs(fields[exact]))]); return float(energies[i]), float(fields[i]), "exact 0 T"
    negative, positive = finite[fields[finite] < 0], finite[fields[finite] > 0]
    if negative.size and positive.size:
        lo, hi = int(negative[np.argmax(fields[negative])]), int(positive[np.argmin(fields[positive])])
        e = energies[lo] + (energies[hi] - energies[lo]) * (-fields[lo]) / (fields[hi] - fields[lo]); return float(e), 0.0, "interpolated near-zero"
    i = int(finite[np.argmin(np.abs(fields[finite]))]); return float(energies[i]), float(fields[i]), "nearest near-zero"


def _assign(previous: list[float], candidates: tuple[PeakCandidate, ...], max_jump: float) -> list[tuple[int, PeakCandidate | None, str]]:
    if not candidates: return [(i, None, "missing") for i in range(len(previous))]
    cost = np.asarray([[abs(p - c.energy_ev) if abs(p - c.energy_ev) <= max_jump else 1e6 for c in candidates] for p in previous])
    rows, cols = linear_sum_assignment(cost); assigned = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < 1e6}; output = []
    for i, prior in enumerate(previous):
        if i not in assigned: output.append((i, None, "missing")); continue
        j = assigned[i]; nearby = sorted(abs(prior - c.energy_ev) for c in candidates); status = "ambiguous" if len(nearby) > 1 and nearby[1] - nearby[0] < max(1e-12, max_jump * 0.05) else "tracked"
        output.append((i, candidates[j], status))
    return output


def analyze_peak_shift(result: Any, *, source: str = "corrected average", prominence_fraction: float = 0.03, min_distance_points: int = 5, smoothing_points: int = 7, max_jump_ev: float = 0.04, max_peaks: int = 6, zero_tolerance_t: float = 1e-9) -> PeakShiftResult:
    fields = np.asarray(result.pair_b, float).ravel(); branches = np.asarray(result.pair_labels, dtype=str).ravel(); spectra = source_spectra(result, source)
    if spectra.shape[0] != fields.size or branches.size != fields.size: raise ValueError("MCD fields, branches, and reflection spectra do not match.")
    candidates = tuple(detect_reflection_peaks(result.energy_ev, row, prominence_fraction=prominence_fraction, min_distance_points=min_distance_points, smoothing_points=smoothing_points, max_peaks=max_peaks) for row in spectra); tracks = []
    for branch in dict.fromkeys(branches.tolist()):
        indices = np.flatnonzero(branches == branch); anchor = int(indices[np.argmin(np.abs(fields[indices]))]); starts = list(candidates[anchor])[:max(1, int(max_peaks))]
        point_map = [[None] * fields.size for _ in starts]
        for track_index, candidate in enumerate(starts): point_map[track_index][anchor] = PeakPoint(float(fields[anchor]), branch, candidate.energy_ev, None, "tracked")
        for direction in (-1, 1):
            sequence = indices[indices < anchor][::-1] if direction < 0 else indices[indices > anchor]; previous = [candidate.energy_ev for candidate in starts]
            for row_index in sequence:
                for track_index, candidate, status in _assign(previous, candidates[row_index], max_jump_ev):
                    point_map[track_index][row_index] = PeakPoint(float(fields[row_index]), branch, None if candidate is None else candidate.energy_ev, None, status)
                    if candidate is not None and status != "ambiguous": previous[track_index] = candidate.energy_ev
        for track_index, points in enumerate(point_map, start=1):
            branch_points = tuple(p for p in points if p is not None); e = np.asarray([np.nan if p.energy_ev is None else p.energy_ev for p in branch_points]); b = np.asarray([p.field_t for p in branch_points]); ref_e, ref_b, method = _reference(e, b, zero_tolerance_t)
            normalized = tuple(PeakPoint(p.field_t, p.branch, p.energy_ev, p.energy_ev - ref_e if p.energy_ev is not None and ref_e is not None else None, p.status) for p in branch_points)
            tracks.append(PeakTrack(track_index, branch, normalized, ref_e, ref_b, method))
    return PeakShiftResult(fields, branches, candidates, tuple(tracks), source)


def valley_quantities(analysis: PeakShiftResult, selected_track_ids: tuple[int, int] = (1, 2)) -> tuple[dict[str, float | str | None], ...]:
    ids = tuple(int(value) for value in selected_track_ids)
    if len(ids) != 2 or ids[0] == ids[1]: return tuple()
    by_branch = {branch: {track.peak_id: track for track in analysis.tracks if track.branch == branch} for branch in dict.fromkeys(analysis.branches.tolist())}; rows = []
    for index, field in enumerate(analysis.fields_t):
        branch = str(analysis.branches[index]); tracks = [by_branch.get(branch, {}).get(track_id) for track_id in ids]
        points = [next((point for point in track.points if abs(point.field_t - field) <= 1e-12), None) if track else None for track in tracks]
        if any(point is None or point.energy_ev is None for point in points) or abs(float(field)) <= 1e-12:
            rows.append({"B_T": float(field), "branch": branch, "E_K": None, "E_Kp": None, "delta_E_K": None, "delta_E_Kp": None, "splitting_E_Kp_minus_E_K": None, "average_E": None, "status": "ambiguous" if abs(float(field)) <= 1e-12 else "missing"}); continue
        ordered = sorted(zip(points, tracks), key=lambda item: float(item[0].energy_ev)); low, high = ordered[0], ordered[1]
        k, kp = (low, high) if field > 0 else (high, low)
        rows.append({"B_T": float(field), "branch": branch, "E_K": float(k[0].energy_ev), "E_Kp": float(kp[0].energy_ev), "delta_E_K": float(k[0].energy_ev) - float(k[1].reference_energy_ev), "delta_E_Kp": float(kp[0].energy_ev) - float(kp[1].reference_energy_ev), "splitting_E_Kp_minus_E_K": float(kp[0].energy_ev) - float(k[0].energy_ev), "average_E": 0.5 * (float(k[0].energy_ev) + float(kp[0].energy_ev)), "status": "tracked"})
    lookup = {(round(float(row["B_T"]), 9), str(row["branch"])): row for row in rows}
    for row in rows:
        opposite = lookup.get((round(-float(row["B_T"]), 9), str(row["branch"])))
        if opposite and row["status"] == "tracked" and opposite["status"] == "tracked":
            row["even_average_E"] = 0.5 * (
                float(row["average_E"]) + float(opposite["average_E"])
            )
            row["odd_average_E"] = 0.5 * (
                float(row["average_E"]) - float(opposite["average_E"])
            )
            row["even_splitting"] = 0.5 * (
                float(row["splitting_E_Kp_minus_E_K"])
                + float(opposite["splitting_E_Kp_minus_E_K"])
            )
            row["odd_splitting"] = 0.5 * (
                float(row["splitting_E_Kp_minus_E_K"])
                - float(opposite["splitting_E_Kp_minus_E_K"])
            )
    return tuple(rows)
