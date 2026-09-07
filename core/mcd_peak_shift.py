"""Deterministic reflection-peak tracking for MCD B sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_widths, savgol_filter

from core.mcd_peak_display import second_derivative_map
from core.mcd_local_fit import LocalFitResult, fit_local_resonance


BOUNDARY_UNRELIABLE = "Boundary unreliable"


def format_mcd_angle(angle_deg: float, tolerance_deg: float = 0.01) -> str:
    """Format measured MCD angles for labels without changing source values."""
    value = float(angle_deg)
    if np.isfinite(value) and abs(value - round(value)) <= float(tolerance_deg):
        return str(int(round(value)))
    return f"{value:.6g}"


@dataclass(frozen=True)
class PeakCandidate:
    energy_ev: float
    prominence: float
    width_points: float
    quality: str = "OK"
    feature_kind: str = "peak"


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
    quality: str = "OK"
    feature_kind: str = "peak"
    locator_energy_ev: float | None = None
    model_name: str | None = None


@dataclass(frozen=True)
class PeakShiftResult:
    fields_t: np.ndarray
    branches: np.ndarray
    candidates: tuple[tuple[PeakCandidate, ...], ...]
    tracks: tuple[PeakTrack, ...]
    source: str
    tracking_method: str = "Raw spectrum"
    local_fits: tuple[LocalFitResult | None, ...] = ()
    locator_energy_ev: float | None = None
    fit_window_ev: tuple[float, float] | None = None
    model_name: str | None = None


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
    order = spectrum_energy_order(result)
    return np.nanmean(np.stack(arrays), axis=0)[:, order]


def spectrum_energy_order(result: Any) -> np.ndarray:
    """Return the shared spectral column order used by peak tracking/display."""
    wavelength = getattr(result, "wavelength_nm", None)
    if wavelength is not None:
        wavelength = np.asarray(wavelength, dtype=float).ravel()
        if wavelength.size == np.asarray(result.energy_ev).size and np.all(np.isfinite(wavelength)):
            return np.argsort(1239.841984 / wavelength, kind="stable")
    return np.argsort(np.asarray(result.energy_ev, dtype=float).ravel(), kind="stable")


def _refine_quadratic(x: np.ndarray, y: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(x) - 1: return float(x[index])
    try:
        a, b, _ = np.polyfit(x[index - 1:index + 2], y[index - 1:index + 2], 2)
        vertex = -b / (2.0 * a)
        return float(vertex) if np.isfinite(vertex) and x[index - 1] <= vertex <= x[index + 1] else float(x[index])
    except (ValueError, np.linalg.LinAlgError): return float(x[index])


def detect_reflection_peaks(energy_ev: np.ndarray, spectrum: np.ndarray, *, prominence_fraction: float = 0.03, min_distance_points: int = 5, smoothing_points: int = 7, max_peaks: int = 8, detrend: bool = True, positive_only: bool = False, feature_kind: str = "peak") -> tuple[PeakCandidate, ...]:
    x0, y0 = np.asarray(energy_ev, float).ravel(), np.asarray(spectrum, float).ravel(); valid = np.isfinite(x0) & np.isfinite(y0)
    if valid.sum() < 5: return ()
    order = np.argsort(x0[valid]); x, y = x0[valid][order], y0[valid][order]
    if np.any(np.diff(x) <= 0): return ()
    window = min(int(smoothing_points) | 1, len(y) if len(y) % 2 else len(y) - 1)
    smooth = savgol_filter(y, window, min(2, window - 1), mode="interp") if window >= 5 else y
    signal = smooth - np.linspace(smooth[0], smooth[-1], len(smooth)) if detrend else smooth
    span = float(np.nanpercentile(signal, 98) - np.nanpercentile(signal, 2))
    indices, props = find_peaks(signal, prominence=max(0.0, prominence_fraction) * max(span, np.finfo(float).eps), distance=max(1, int(min_distance_points)))
    ranked = sorted(indices.tolist(), key=lambda i: (-float(props["prominences"][np.where(indices == i)[0][0]]), i))[:max(1, int(max_peaks))]; ranked.sort()
    if positive_only:
        ranked = [index for index in ranked if signal[index] > 0.0]
    widths, _, left_ips, right_ips = peak_widths(signal, np.asarray(ranked, dtype=int), rel_height=0.5) if ranked else (np.array([]), np.array([]), np.array([]), np.array([]))
    candidates = []
    for index, width, left_ip, right_ip in zip(ranked, widths, left_ips, right_ips):
        # Require one full half-prominence width of measured support on each
        # side. Edge shoulders remain visible for inspection without becoming
        # fit inputs.
        peak_support = max(2.0, float(width))
        quality = "OK" if index >= peak_support and len(x) - 1 - index >= peak_support else BOUNDARY_UNRELIABLE
        candidates.append(PeakCandidate(
            _refine_quadratic(x, signal, index),
            float(props["prominences"][np.where(indices == index)[0][0]]),
            float(width),
            quality,
            feature_kind,
        ))
    return tuple(candidates)


def _reference(energies: np.ndarray, fields: np.ndarray, tolerance: float, eligible: np.ndarray | None = None) -> tuple[float | None, float | None, str]:
    finite_mask = np.isfinite(energies)
    if eligible is not None:
        finite_mask &= np.asarray(eligible, dtype=bool)
    finite = np.flatnonzero(finite_mask)
    if finite.size == 0: return None, None, "unavailable"
    exact = finite[np.abs(fields[finite]) <= tolerance]
    if exact.size:
        i = int(exact[np.argmin(np.abs(fields[exact]))]); return float(energies[i]), float(fields[i]), "exact 0 T"
    # Interpolate only across an adjacent valid negative/positive pair.  A
    # missing or ambiguous point between the two sides is an invalid bridge.
    order = np.argsort(fields, kind="stable")
    for left, right in zip(order[:-1], order[1:]):
        if left not in finite or right not in finite:
            continue
        if fields[left] < 0.0 < fields[right]:
            ratio = -fields[left] / (fields[right] - fields[left])
            e = energies[left] + (energies[right] - energies[left]) * ratio
            return float(e), 0.0, "interpolated near-zero"
    # A nearest non-zero measurement is useful context, but it is not E0 and
    # must not be used to manufacture a shift reference.
    return None, None, "unavailable"


def _assign(previous: list[float], candidates: tuple[PeakCandidate, ...], max_jump: float) -> list[tuple[int, PeakCandidate | None, str]]:
    if not previous:
        return []
    if not candidates: return [(i, None, "missing") for i in range(len(previous))]
    cost = np.asarray([[abs(p - c.energy_ev) if abs(p - c.energy_ev) <= max_jump else 1e6 for c in candidates] for p in previous])
    rows, cols = linear_sum_assignment(cost); assigned = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < 1e6}; output = []
    for i, prior in enumerate(previous):
        if i not in assigned: output.append((i, None, "missing")); continue
        j = assigned[i]; nearby = sorted(abs(prior - c.energy_ev) for c in candidates); status = "ambiguous" if len(nearby) > 1 and nearby[1] - nearby[0] < max(1e-12, max_jump * 0.05) else "tracked"
        output.append((i, candidates[j], status))
    return output


def _tracking_spectra(energy_ev: np.ndarray, spectra: np.ndarray, method: str, derivative_window_points: int) -> tuple[np.ndarray, np.ndarray]:
    normalized = method.casefold().strip()
    if normalized in {"raw", "raw spectrum"}:
        return np.asarray(energy_ev, float), np.asarray(spectra, float)
    if normalized not in {"second derivative", "d2", "d²r/de²"}:
        raise ValueError(f"Unknown peak tracking method: {method}")
    grid, derivative, _ = second_derivative_map(energy_ev, spectra, derivative_window_points)
    return grid, -derivative


def _associate_curvature_candidates(
    energy_ev: np.ndarray,
    raw_spectrum: np.ndarray,
    curvature_candidates: tuple[PeakCandidate, ...],
    *,
    prominence_fraction: float,
    min_distance_points: int,
    smoothing_points: int,
    max_peaks: int,
    feature_kind: str = "peak",
) -> tuple[PeakCandidate, ...]:
    """Keep -d²R/dE² maxima that coincide with a measured convex R maximum."""
    raw_candidates = detect_reflection_peaks(
        energy_ev,
        raw_spectrum if feature_kind == "peak" else -raw_spectrum,
        prominence_fraction=prominence_fraction,
        min_distance_points=min_distance_points,
        smoothing_points=smoothing_points,
        max_peaks=max_peaks,
        feature_kind=feature_kind,
    )
    if not raw_candidates:
        return ()
    step = float(np.nanmedian(np.diff(np.sort(np.asarray(energy_ev, float)))))
    tolerance = max(0.005, 3.0 * step)
    associated = []
    used_curvature: set[int] = set()
    for raw in raw_candidates:
        matches = [(index, candidate) for index, candidate in enumerate(curvature_candidates) if index not in used_curvature and abs(candidate.energy_ev - raw.energy_ev) <= tolerance]
        if matches:
            index, candidate = max(matches, key=lambda item: (item[1].prominence, -abs(item[1].energy_ev - raw.energy_ev), -item[1].energy_ev))
            used_curvature.add(index)
            # Multiple curvature maxima inside one raw resonance are not
            # safely distinguishable as a shifted center.
            competing = len(matches) > 1
            if raw.quality == BOUNDARY_UNRELIABLE or candidate.quality == BOUNDARY_UNRELIABLE:
                quality = BOUNDARY_UNRELIABLE
            else:
                quality = "ambiguous" if competing else candidate.quality
            associated.append(PeakCandidate(candidate.energy_ev, candidate.prominence, candidate.width_points, quality, candidate.feature_kind))
    return tuple(associated)


def analyze_peak_shift(result: Any, *, source: str = "corrected average", prominence_fraction: float = 0.03, min_distance_points: int = 5, smoothing_points: int = 7, max_jump_ev: float = 0.04, max_peaks: int = 6, zero_tolerance_t: float = 1e-9, tracking_method: str = "Raw spectrum", derivative_window_points: int = 35, seed_energy_ev: float | None = None, seed_half_width_ev: float = 0.005, seed_field_t: float = 0.0) -> PeakShiftResult:
    fields = np.asarray(result.pair_b, float).ravel(); branches = np.asarray(result.pair_labels, dtype=str).ravel(); spectra = source_spectra(result, source)
    if spectra.shape[0] != fields.size or branches.size != fields.size: raise ValueError("MCD fields, branches, and reflection spectra do not match.")
    seeded = seed_energy_ev is not None
    if seeded:
        seed_energy = float(seed_energy_ev)
        seed_half_width = float(seed_half_width_ev)
        seed_field = float(seed_field_t)
        if not np.isfinite(seed_energy):
            raise ValueError("seed_energy_ev must be finite when provided.")
        if not np.isfinite(seed_half_width) or seed_half_width <= 0.0:
            raise ValueError("seed_half_width_ev must be finite and positive.")
        if not np.isfinite(seed_field):
            raise ValueError("seed_field_t must be finite when provided.")
    # ``source_spectra`` puts every spectral row on ascending energy. Keep the
    # coordinate array in that same order, including wavelength-driven CSVs.
    # The result energy axis is the canonical display/tracking coordinate and
    # may already be sorted independently of the source CSV wavelength order.
    # ``source_spectra`` handles the raw-column reorder; never apply that
    # column permutation a second time to the coordinate array.
    ordered_energy = np.sort(np.asarray(result.energy_ev, dtype=float).ravel())
    tracking_energy, tracking_spectra = _tracking_spectra(ordered_energy, spectra, tracking_method, derivative_window_points)
    use_detrend = tracking_method.casefold().strip() in {"raw", "raw spectrum"}
    # A manual seed is allowed to select a lower-ranked real resonance, while
    # preserving the configured prominence and boundary-quality checks.
    detection_max_peaks = max(1, int(max_peaks)) if not seeded else max(1, int(tracking_energy.size))
    all_candidates: list[list[PeakCandidate]] = [[] for _ in range(fields.size)]
    tracks: list[PeakTrack] = []
    for kind, kind_spectra in (("peak", tracking_spectra), ("dip", -tracking_spectra)):
        candidates = []
        for raw_row, tracking_row in zip(spectra, kind_spectra):
            detected = detect_reflection_peaks(tracking_energy, tracking_row, prominence_fraction=prominence_fraction, min_distance_points=min_distance_points, smoothing_points=smoothing_points, max_peaks=detection_max_peaks, detrend=use_detrend, positive_only=not use_detrend, feature_kind=kind)
            if not use_detrend:
                detected = _associate_curvature_candidates(ordered_energy, raw_row, detected, prominence_fraction=prominence_fraction, min_distance_points=min_distance_points, smoothing_points=smoothing_points, max_peaks=detection_max_peaks, feature_kind=kind)
            candidates.append(detected)
        for row_index, row_candidates in enumerate(candidates):
            all_candidates[row_index].extend(row_candidates)
        for branch in dict.fromkeys(branches.tolist()):
            indices = np.flatnonzero(branches == branch)
            anchor = int(indices[np.argmin(np.abs(fields[indices] - seed_field))]) if seeded else int(indices[np.argmin(np.abs(fields[indices]))])
            if seeded:
                nearby = [candidate for candidate in candidates[anchor] if candidate.quality == "OK" and abs(candidate.energy_ev - seed_energy) <= seed_half_width]
                starts = [min(nearby, key=lambda candidate: (abs(candidate.energy_ev - seed_energy), candidate.energy_ev))] if nearby else []
            else:
                starts = list(candidates[anchor])[:max(1, int(max_peaks))]
            point_map = [[None] * fields.size for _ in starts]
            for track_index, candidate in enumerate(starts):
                point_map[track_index][anchor] = PeakPoint(float(fields[anchor]), branch, candidate.energy_ev, None, "tracked" if candidate.quality == "OK" else candidate.quality)
            for direction in (-1, 1):
                sequence = indices[indices < anchor][::-1] if direction < 0 else indices[indices > anchor]; previous = [candidate.energy_ev for candidate in starts]
                for row_index in sequence:
                    for track_index, candidate, status in _assign(previous, candidates[row_index], max_jump_ev):
                        point_status = status if candidate is None or candidate.quality == "OK" else candidate.quality
                        point_map[track_index][row_index] = PeakPoint(float(fields[row_index]), branch, None if candidate is None else candidate.energy_ev, None, point_status)
                        if candidate is not None and status != "ambiguous" and candidate.quality == "OK": previous[track_index] = candidate.energy_ev
            for track_index, points in enumerate(point_map, start=1):
                branch_points = tuple(p for p in points if p is not None); e = np.asarray([np.nan if p.energy_ev is None else p.energy_ev for p in branch_points]); b = np.asarray([p.field_t for p in branch_points]); eligible = np.asarray([p.status == "tracked" for p in branch_points]); ref_e, ref_b, ref_method = _reference(e, b, zero_tolerance_t, eligible)
                normalized = tuple(PeakPoint(p.field_t, p.branch, p.energy_ev, p.energy_ev - ref_e if p.energy_ev is not None and ref_e is not None and p.status == "tracked" else None, p.status) for p in branch_points)
                quality = BOUNDARY_UNRELIABLE if any(p.status == BOUNDARY_UNRELIABLE for p in normalized) else "OK"
                tracks.append(PeakTrack(track_index, branch, normalized, ref_e, ref_b, ref_method, quality, kind))
    return PeakShiftResult(fields, branches, tuple(tuple(row) for row in all_candidates), tuple(tracks), source, tracking_method)


def analyze_local_peak_shift(
    result: Any,
    *,
    source: str = "raw pos",
    seed_energy_ev: float,
    locator_energy_ev: float | None = None,
    feature_kind: str = "peak",
    window_ev: tuple[float, float] | None = None,
    window_before_ev: float = 0.0136,
    window_after_ev: float = 0.0164,
    background_model: str = "linear",
    zero_tolerance_t: float = 1e-9,
    max_starts: int = 150,
    peak_id: int = 1,
    cancel_check: Any | None = None,
) -> PeakShiftResult:
    """Fit one selected resonance locally for every measured row.

    ``locator_energy_ev`` records the clicked/detected identity.  The energy
    stored in each point is the local model centre, so a fit cannot silently
    turn a new model centre into a new physical selection.
    """
    fields = np.asarray(result.pair_b, dtype=float).ravel()
    branches = np.asarray(result.pair_labels, dtype=str).ravel()
    spectra = source_spectra(result, source)
    energy = np.sort(np.asarray(result.energy_ev, dtype=float).ravel())
    if spectra.shape[0] != fields.size or branches.size != fields.size:
        raise ValueError("MCD fields, branches, and reflection spectra do not match.")
    if spectra.shape[1] != energy.size:
        raise ValueError("MCD reflection spectra do not match the energy axis.")
    fits: list[LocalFitResult | None] = []
    for row in spectra:
        if cancel_check is not None and bool(cancel_check()):
            break
        fits.append(
            fit_local_resonance(
                energy,
                row,
                seed_energy_ev=float(seed_energy_ev),
                window_ev=window_ev,
                window_before_ev=window_before_ev,
                window_after_ev=window_after_ev,
                background_model=background_model,
                locator_energy_ev=locator_energy_ev,
                max_starts=max_starts,
            )
        )
    if len(fits) != fields.size:
        bounds = window_ev or (float(seed_energy_ev) - float(window_before_ev), float(seed_energy_ev) + float(window_after_ev))
        return PeakShiftResult(fields, branches, tuple(() for _ in fields), tuple(), source, "Local mixed fit", tuple(fits), locator_energy_ev, tuple(float(v) for v in bounds), f"Local mixed fit ({background_model.casefold()} background)")
    tracks: list[PeakTrack] = []
    for branch in dict.fromkeys(branches.tolist()):
        indices = np.flatnonzero(branches == branch)
        branch_fits = [fits[int(index)] for index in indices]
        raw_energies = np.asarray([np.nan if fit is None or fit.status != "ok" or fit.center_ev is None else fit.center_ev for fit in branch_fits], dtype=float)
        branch_fields = fields[indices]
        eligible = np.isfinite(raw_energies)
        ref_e, ref_b, ref_method = _reference(raw_energies, branch_fields, zero_tolerance_t, eligible)
        points: list[PeakPoint] = []
        for field, fit, center in zip(branch_fields, branch_fits, raw_energies):
            status = "tracked" if np.isfinite(center) else (fit.status if fit is not None else "missing")
            points.append(PeakPoint(float(field), branch, float(center) if np.isfinite(center) else None, float(center - ref_e) if np.isfinite(center) and ref_e is not None else None, status))
        quality = "OK" if all(point.status == "tracked" for point in points) else next((point.status for point in points if point.status != "tracked"), "unavailable")
        tracks.append(PeakTrack(int(peak_id), branch, tuple(points), ref_e, ref_b, ref_method, quality, feature_kind, locator_energy_ev, f"Local mixed fit ({background_model.casefold()} background)"))
    bounds = window_ev or (float(seed_energy_ev) - float(window_before_ev), float(seed_energy_ev) + float(window_after_ev))
    return PeakShiftResult(fields, branches, tuple(() for _ in fields), tuple(tracks), source, "Local mixed fit", tuple(fits), locator_energy_ev, tuple(float(v) for v in bounds), f"Local mixed fit ({background_model.casefold()} background)")


def valley_quantities(analysis: PeakShiftResult, selected_track_ids: tuple[int, int] = (1, 2)) -> tuple[dict[str, float | str | None], ...]:
    ids = tuple(int(value) for value in selected_track_ids)
    if len(ids) != 2 or ids[0] == ids[1]: return tuple()
    by_branch = {branch: {track.peak_id: track for track in analysis.tracks if track.branch == branch and track.feature_kind == "peak"} for branch in dict.fromkeys(analysis.branches.tolist())}; rows = []
    for index, field in enumerate(analysis.fields_t):
        branch = str(analysis.branches[index]); tracks = [by_branch.get(branch, {}).get(track_id) for track_id in ids]
        points = [next((point for point in track.points if abs(point.field_t - field) <= 1e-12), None) if track else None for track in tracks]
        if any(point is None or point.energy_ev is None for point in points) or abs(float(field)) <= 1e-12:
            field_points = [point for track in tracks if track is not None for point in track.points if abs(point.field_t - field) <= 1e-12]
            status = "ambiguous" if abs(float(field)) <= 1e-12 else "missing"
            if any(point.status == BOUNDARY_UNRELIABLE for point in field_points):
                status = BOUNDARY_UNRELIABLE
            rows.append({"B_T": float(field), "branch": branch, "E_K": None, "E_Kp": None, "delta_E_K": None, "delta_E_Kp": None, "splitting_E_Kp_minus_E_K": None, "average_E": None, "status": status}); continue
        ordered = sorted(zip(points, tracks), key=lambda item: float(item[0].energy_ev)); low, high = ordered[0], ordered[1]
        k, kp = (low, high) if field > 0 else (high, low)
        selected_points = (k[0], kp[0])
        if any(point.status == BOUNDARY_UNRELIABLE for point in selected_points):
            quality = BOUNDARY_UNRELIABLE
        elif any(point.status != "tracked" for point in selected_points):
            quality = next(point.status for point in selected_points if point.status != "tracked")
        elif any(track.reference_energy_ev is None for track in (k[1], kp[1])):
            quality = "unavailable"
        else:
            quality = "tracked"
        rows.append({"B_T": float(field), "branch": branch, "E_K": float(k[0].energy_ev), "E_Kp": float(kp[0].energy_ev), "delta_E_K": float(k[0].energy_ev) - float(k[1].reference_energy_ev) if quality == "tracked" else None, "delta_E_Kp": float(kp[0].energy_ev) - float(kp[1].reference_energy_ev) if quality == "tracked" else None, "splitting_E_Kp_minus_E_K": float(kp[0].energy_ev) - float(k[0].energy_ev) if quality == "tracked" else None, "average_E": 0.5 * (float(k[0].energy_ev) + float(kp[0].energy_ev)) if quality == "tracked" else None, "status": quality})
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
