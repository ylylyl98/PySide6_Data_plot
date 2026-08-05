"""Magnetic circular dichroism (MCD) B-sweep processing.

The input format is a single CSV whose first two columns are B field (T) and
waveplate/analyser angle (deg), followed by wavelength columns (nm).  The
normalisation is deliberately explicit: each angle is dark-subtracted and
normalised by its own near-zero-field spectrum before the two angles are
compared.  This removes wavelength-dependent waveplate throughput while
retaining a record of the reference used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.signal import find_peaks, peak_widths, savgol_filter

from core.loader import DataCube

EV_NM = 1239.841984


@dataclass(frozen=True)
class McdSettings:
    pos_angle: float | None = None
    neg_angle: float | None = None
    max_sequence_gap: int = 3
    max_delta_b: float = 0.1
    pair_b_alignment: Literal["direct", "interpolate"] = "direct"
    zero_window_t: float = 0.02
    reference_mode: Literal["nearest", "window"] = "nearest"
    bin_decimals: int = 3
    gain_mode: Literal["per_wavelength", "smoothed", "scalar"] = "per_wavelength"
    gain_smooth_window: int = 15
    gain_smooth_poly: int = 2
    correction_mode: Literal["global", "pair_scale", "pair_affine", "pair_spectral"] = "pair_spectral"
    spectral_order: Literal[1, 2] = 2
    background_ranges_ev: tuple[tuple[float, float], ...] = ()
    background_selection: Literal["auto", "manual", "suggested"] = "auto"
    suggestion_protected_ranges_ev: tuple[tuple[float, float], ...] = ()
    manual_protected_ranges_ev: tuple[tuple[float, float], ...] = ()
    suggestion_linear_validation_rms: float | None = None
    suggestion_quadratic_validation_rms: float | None = None
    suggestion_algorithm: str | None = None
    dark_pos_file: str | None = None
    dark_neg_file: str | None = None


@dataclass
class McdResult:
    source_file: str
    wavelength_nm: np.ndarray
    energy_ev: np.ndarray
    pos_angle: float
    neg_angle: float
    pair_b: np.ndarray
    pair_b_pos: np.ndarray
    pair_b_neg: np.ndarray
    pair_delta_b: np.ndarray
    pair_sequence_gap: np.ndarray
    pair_interpolated_pos: np.ndarray
    pair_interpolated_neg: np.ndarray
    pair_labels: np.ndarray
    pair_raw_pos: np.ndarray
    pair_raw_neg: np.ndarray
    pair_corrected_pos: np.ndarray
    pair_corrected_neg: np.ndarray
    pair_mcd_raw: np.ndarray
    pair_mcd_corrected: np.ndarray
    pair_scale: np.ndarray
    pair_offset: np.ndarray
    pair_spectral_slope: np.ndarray
    pair_spectral_curvature: np.ndarray
    pair_correction_min: np.ndarray
    pair_correction_max: np.ndarray
    pair_background_rms_before: np.ndarray
    pair_background_rms: np.ndarray
    reference_b: float
    gain: np.ndarray
    reference_pos: np.ndarray
    reference_neg: np.ndarray
    dark_pos: np.ndarray
    dark_neg: np.ndarray
    maps: dict[str, DataCube]
    summary: dict[str, object]

    def cube(self, name: str) -> DataCube:
        try:
            return self.maps[name]
        except KeyError as exc:
            raise ValueError(f"MCD map {name!r} is not available.") from exc


@dataclass(frozen=True)
class McdReflectionFeature:
    """A reviewable reflection resonance detected from the full sweep."""

    center_ev: float
    start_ev: float
    stop_ev: float
    kind: Literal["peak", "dip", "peak/dip"]
    prominence_log: float
    snr: float
    width_ev: float
    confidence: float
    recommended: bool


@dataclass(frozen=True)
class McdBackgroundSuggestion:
    """Review-only recommendation for spectral-background fitting regions.

    ``ranges`` are never applied implicitly.  They are shown to the user so
    they can be accepted, edited, or discarded before another MCD processing
    run begins.
    """

    energy_ev: np.ndarray
    median_reflectance: np.ndarray
    suitability: np.ndarray
    feature_baseline: np.ndarray
    feature_residual: np.ndarray
    feature_detection_score: np.ndarray
    ranges: tuple[tuple[float, float], ...]
    requested_protected_ranges: tuple[tuple[float, float], ...]
    detected_feature_ranges: tuple[tuple[float, float], ...]
    detected_feature_kinds: tuple[str, ...]
    protected_ranges: tuple[tuple[float, float], ...]
    suggested_order: Literal[1, 2]
    linear_validation_rms: float
    quadratic_validation_rms: float
    coverage_fraction: float
    span_fraction: float
    notes: tuple[str, ...]
    detected_features: tuple[McdReflectionFeature, ...] = ()
    manual_protected_ranges: tuple[tuple[float, float], ...] = ()


def _numeric_spectrum_columns(frame: pd.DataFrame) -> list[object]:
    columns: list[object] = []
    for column in frame.columns:
        try:
            float(column)
        except (TypeError, ValueError):
            continue
        columns.append(column)
    if not columns:
        raise ValueError("MCD CSV needs numeric wavelength columns (nm).")
    return columns


def _load_reference_file(path: str, wavelength_nm: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(path)
    columns = _numeric_spectrum_columns(frame)
    axis = np.asarray([float(column) for column in columns], float)
    order = np.argsort(axis)
    target_order = np.argsort(wavelength_nm)
    if axis.size != wavelength_nm.size or not np.allclose(axis[order], wavelength_nm[target_order], atol=1e-6, rtol=0):
        raise ValueError(f"Reference wavelength axis does not match the MCD data: {Path(path).name}")
    values_sorted = np.nanmean(frame[columns].to_numpy(float), axis=0)[order]
    return values_sorted[np.argsort(target_order)]


def load_b_sweep_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if frame.shape[1] < 4:
        raise ValueError("MCD CSV must contain B_T, angle_deg, and at least two wavelength columns.")
    try:
        b = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(float)
        angle = pd.to_numeric(frame.iloc[:, 1], errors="raise").to_numpy(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("The first two MCD CSV columns must be numeric B field and angle.") from exc
    columns = list(frame.columns[2:])
    try:
        wavelength = np.asarray([float(column) for column in columns], float)
    except (TypeError, ValueError) as exc:
        raise ValueError("MCD wavelength column headers must be numeric nm values.") from exc
    spectra = frame.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(angle)):
        raise ValueError("MCD B field and angle columns cannot contain missing values.")
    if np.unique(wavelength).size != wavelength.size:
        raise ValueError("MCD wavelength columns must be unique.")
    return b, angle, wavelength, spectra


def detect_angles(path: str) -> tuple[float, ...]:
    _b, angles, _wavelength, _spectra = load_b_sweep_csv(path)
    return tuple(float(value) for value in sorted(np.unique(angles)))


def _pair_angles(
    b: np.ndarray,
    angle: np.ndarray,
    spectra: np.ndarray,
    settings: McdSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angles = np.unique(angle)
    if angles.size < 2:
        raise ValueError("MCD data needs spectra from two waveplate/analyser angles.")
    pos = float(settings.pos_angle) if settings.pos_angle is not None else float(np.max(angles))
    neg = float(settings.neg_angle) if settings.neg_angle is not None else float(np.min(angles))
    if np.isclose(pos, neg):
        raise ValueError("The two selected MCD angles must be different.")
    if not np.any(np.isclose(angle, pos)) or not np.any(np.isclose(angle, neg)):
        raise ValueError("A selected MCD angle is not present in the CSV.")

    pending: dict[float, tuple[int, float, np.ndarray] | None] = {pos: None, neg: None}
    pair_b: list[float] = []
    pair_b_pos: list[float] = []
    pair_b_neg: list[float] = []
    pair_gap: list[int] = []
    pair_pos_index: list[int] = []
    pair_neg_index: list[int] = []
    i_pos: list[np.ndarray] = []
    i_neg: list[np.ndarray] = []
    for index, (field, value_angle, spectrum) in enumerate(zip(b, angle, spectra)):
        matched_angle = pos if np.isclose(value_angle, pos) else (neg if np.isclose(value_angle, neg) else None)
        if matched_angle is None:
            continue
        record = (index, float(field), spectrum)
        other = neg if matched_angle == pos else pos
        pending_other = pending[other]
        if pending_other is not None:
            other_index, other_field, other_spectrum = pending_other
            if index - other_index <= settings.max_sequence_gap and abs(float(field) - other_field) <= settings.max_delta_b:
                pair_b.append(0.5 * (other_field + float(field)))
                if matched_angle == pos:
                    i_pos.append(spectrum)
                    i_neg.append(other_spectrum)
                    pair_b_pos.append(float(field))
                    pair_b_neg.append(other_field)
                    pair_pos_index.append(index)
                    pair_neg_index.append(other_index)
                else:
                    i_pos.append(other_spectrum)
                    i_neg.append(spectrum)
                    pair_b_pos.append(other_field)
                    pair_b_neg.append(float(field))
                    pair_pos_index.append(other_index)
                    pair_neg_index.append(index)
                pair_gap.append(index - other_index)
                pending[other] = None
                continue
        pending[matched_angle] = record
    if not pair_b:
        raise ValueError("No opposite-angle MCD pairs found; relax pairing tolerances or verify acquisition order.")
    return (
        np.asarray(pair_b, float), np.asarray(pair_b_pos, float), np.asarray(pair_b_neg, float),
        np.asarray(pair_gap, int), np.asarray(pair_pos_index, int), np.asarray(pair_neg_index, int), np.vstack(i_pos), np.vstack(i_neg),
    )


def _sweep_segments(b: np.ndarray) -> np.ndarray:
    """Split a time-ordered sweep at field reversals, avoiding cross-branch interpolation."""
    field = np.asarray(b, float)
    segments = np.zeros(field.size, dtype=int)
    direction = 0.0
    segment = 0
    for index, step in enumerate(np.diff(field), start=1):
        new_direction = float(np.sign(step))
        if new_direction and direction and new_direction != direction:
            segment += 1
        if new_direction:
            direction = new_direction
        segments[index] = segment
    return segments


def _pair_sweep_labels(pair_b: np.ndarray) -> np.ndarray:
    """Label acquired MCD pairs by the physical direction of the B sweep.

    The first pair inherits the first nonzero measured direction.  This avoids
    incorrectly calling the first pair ``up`` when an acquisition begins at
    positive field and immediately sweeps towards negative field.
    """
    fields = np.asarray(pair_b, float).ravel()
    if fields.size == 0:
        return np.asarray([], dtype=str)
    steps = np.diff(fields)
    nonzero = steps[np.isfinite(steps) & (np.abs(steps) > 1e-12)]
    direction = 1.0 if nonzero.size == 0 else float(np.sign(nonzero[0]))
    labels: list[str] = ["B increasing" if direction > 0 else "B decreasing"]
    for step in steps:
        if np.isfinite(step) and abs(float(step)) > 1e-12:
            direction = float(np.sign(step))
        labels.append("B increasing" if direction > 0 else "B decreasing")
    return np.asarray(labels, dtype=str)


def _interpolate_pair_spectra(
    b: np.ndarray,
    angle: np.ndarray,
    spectra: np.ndarray,
    source_indices: np.ndarray,
    target_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly align one angle channel to Bpair without crossing a sweep reversal."""
    segments = _sweep_segments(b)
    aligned = np.asarray(spectra[source_indices], float).copy()
    used = np.zeros(source_indices.size, dtype=bool)
    for pair_index, (source_index, target) in enumerate(zip(source_indices, target_b)):
        candidate = np.flatnonzero((segments == segments[source_index]) & np.isclose(angle, angle[source_index]))
        order = candidate[np.argsort(b[candidate])]
        fields = b[order]
        right = int(np.searchsorted(fields, target, side="left"))
        if right == 0 or right >= fields.size:
            continue
        left = right - 1
        x0, x1 = float(fields[left]), float(fields[right])
        if not (x0 < target < x1):
            continue
        fraction = (float(target) - x0) / (x1 - x0)
        aligned[pair_index] = (1.0 - fraction) * spectra[order[left]] + fraction * spectra[order[right]]
        used[pair_index] = True
    return aligned, used


def _background_mask(energy_ev: np.ndarray, ranges: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Return safe fitting points away from the selected optical feature(s)."""
    energy = np.asarray(energy_ev, float)
    mask = np.zeros(energy.size, dtype=bool)
    for lo, hi in background_fit_regions(energy, ranges):
        mask |= (energy >= lo) & (energy <= hi)
    if ranges and np.count_nonzero(mask) < 3:
        raise ValueError("Correction background ranges must contain at least three energy samples.")
    return mask


def background_fit_regions(
    energy_ev: np.ndarray,
    ranges: tuple[tuple[float, float], ...] = (),
) -> tuple[tuple[float, float], ...]:
    """Return the exact energy intervals used for a per-pair drift fit.

    With no manual ranges, this is the lower 15% *and* upper 15% of the
    acquired energy span.  It therefore uses both ends of the spectrum.
    """
    energy = np.asarray(energy_ev, float)
    if ranges:
        return tuple(tuple(sorted((float(start), float(stop)))) for start, stop in ranges)
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        return ()
    low_cut, high_cut = np.nanpercentile(finite, (15.0, 85.0))
    return ((float(np.nanmin(finite)), float(low_cut)), (float(high_cut), float(np.nanmax(finite))))


def _robust_log_ratio_coefficients(
    energy_ev: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    valid: np.ndarray,
    *,
    order: int,
    center_ev: float,
) -> np.ndarray | None:
    """Fit log(pos/neg) with conservative Huber reweighting."""
    energy = np.asarray(energy_ev, float)
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    use = np.asarray(valid, bool) & np.isfinite(energy) & np.isfinite(pos) & np.isfinite(neg) & (pos > 1e-30) & (neg > 1e-30)
    polynomial_order = int(np.clip(order, 1, 2))
    if np.count_nonzero(use) < max(6, polynomial_order + 3):
        return None
    x = energy[use] - float(center_ev)
    values = np.log(pos[use] / neg[use])
    design = np.column_stack([x ** power for power in range(polynomial_order + 1)])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    for _iteration in range(8):
        fit_residual = values - design @ coefficients
        centered = fit_residual - float(np.nanmedian(fit_residual))
        robust_sigma = 1.4826 * float(np.nanmedian(np.abs(centered)))
        if not np.isfinite(robust_sigma) or robust_sigma <= 1e-12:
            break
        scaled = np.abs(centered) / (1.5 * robust_sigma)
        weights = np.ones_like(scaled)
        outliers = scaled > 1.0
        weights[outliers] = 1.0 / scaled[outliers]
        root_weights = np.sqrt(weights)
        updated = np.linalg.lstsq(design * root_weights[:, None], values * root_weights, rcond=None)[0]
        if np.allclose(updated, coefficients, rtol=1e-8, atol=1e-10):
            coefficients = updated
            break
        coefficients = updated
    return np.asarray(coefficients, float)


def _ranges_from_mask(energy_ev: np.ndarray, mask: np.ndarray, *, min_width_ev: float) -> tuple[tuple[float, float], ...]:
    """Turn a sorted boolean energy mask into contiguous, usable ranges."""
    energy = np.asarray(energy_ev, float)
    use = np.asarray(mask, bool) & np.isfinite(energy)
    ranges: list[tuple[float, float]] = []
    start: int | None = None
    for index, enabled in enumerate(use):
        if enabled and start is None:
            start = index
        if start is not None and (not enabled or index == use.size - 1):
            stop = index if enabled and index == use.size - 1 else index - 1
            if stop > start and float(energy[stop] - energy[start]) >= float(min_width_ev):
                ranges.append((float(energy[start]), float(energy[stop])))
            start = None
    return tuple(ranges)


def _bridge_short_mask_gaps(energy_ev: np.ndarray, mask: np.ndarray, *, max_gap_ev: float) -> np.ndarray:
    """Fill sub-resolution gaps in a candidate background mask."""
    energy = np.asarray(energy_ev, float)
    cleaned = np.asarray(mask, bool).copy()
    index = 0
    while index < cleaned.size:
        if cleaned[index]:
            index += 1
            continue
        start = index
        while index < cleaned.size and not cleaned[index]:
            index += 1
        stop = index - 1
        if start == 0 or index >= cleaned.size:
            continue
        if cleaned[start - 1] and cleaned[index] and float(energy[stop] - energy[start]) <= float(max_gap_ev):
            cleaned[start:index] = True
    return cleaned


def _mask_from_ranges(energy_ev: np.ndarray, ranges: tuple[tuple[float, float], ...]) -> np.ndarray:
    mask = np.zeros(np.asarray(energy_ev).size, dtype=bool)
    for start, stop in ranges:
        lo, hi = sorted((float(start), float(stop)))
        mask |= (energy_ev >= lo) & (energy_ev <= hi)
    return mask


def _merge_energy_ranges(ranges: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    """Normalise overlapping energy intervals into a compact set."""
    ordered = sorted((tuple(sorted((float(start), float(stop)))) for start, stop in ranges), key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for start, stop in ordered:
        if not merged or start > merged[-1][1] + 1e-12:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return tuple(merged)


def _odd_window_for_energy(energy_ev: np.ndarray, span_ev: float, *, minimum: int = 5) -> int | None:
    """Return a safe odd Savitzky-Golay window for a physical energy span."""
    energy = np.asarray(energy_ev, float)
    if energy.size < minimum:
        return None
    steps = np.diff(energy)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return None
    spacing = float(np.nanmedian(steps))
    maximum = energy.size if energy.size % 2 else energy.size - 1
    desired = max(int(minimum), int(round(float(span_ev) / max(spacing, 1e-12))))
    if desired % 2 == 0:
        desired += 1
    desired = min(desired, maximum)
    return desired if desired >= minimum else None


def _robust_sigma(values: np.ndarray, *, floor: float = 5e-4) -> float:
    """Robust Gaussian-equivalent scatter with a small physical floor."""
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(floor)
    median = float(np.nanmedian(finite))
    return max(1.4826 * float(np.nanmedian(np.abs(finite - median))), float(floor))


def _detect_reflection_features(
    energy_ev: np.ndarray,
    log_reflectance: np.ndarray,
    finite: np.ndarray,
    *,
    guard_ev: float,
) -> tuple[tuple[McdReflectionFeature, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Find physical reflection features from a detrended, noise-normalised trace.

    Direct extrema on a strongly sloped reflection trace produce a false peak
    on the shoulder of a real dip.  This routine works on a broad-baseline
    residual, measures the local noise and FWHM in eV, then clusters nearby
    peak/dip components into one reviewable resonance.
    """
    energy = np.asarray(energy_ev, float)
    signal = np.asarray(log_reflectance, float)
    valid = np.asarray(finite, bool) & np.isfinite(signal)
    empty = np.full(energy.size, np.nan, float)
    if np.count_nonzero(valid) < 9:
        return (), empty, empty, empty

    # Interpolate to a uniform grid: Savitzky-Golay windows and peak widths
    # below are specified in eV, not in a number of acquired wavelength pixels.
    grid = np.linspace(float(energy[0]), float(energy[-1]), energy.size)
    uniform = np.interp(grid, energy[valid], signal[valid])
    fine_window = _odd_window_for_energy(grid, 0.004)
    baseline_window = _odd_window_for_energy(grid, 0.040)
    if fine_window is None or baseline_window is None:
        return (), empty, empty, empty
    denoised = savgol_filter(uniform, fine_window, min(3, fine_window - 1), mode="interp")
    baseline = savgol_filter(denoised, baseline_window, min(3, baseline_window - 1), mode="interp")
    residual = denoised - baseline

    # Estimate detector noise from the part removed by the fine filter.  The
    # rolling MAD keeps a noisy spectral section from making the entire sweep
    # insensitive, while the global floor prevents implausibly large SNR in an
    # almost noiseless trace.
    high_frequency = uniform - denoised
    global_sigma = _robust_sigma(high_frequency)
    local_window = _odd_window_for_energy(grid, 0.030)
    if local_window is None:
        local_sigma = np.full(grid.size, global_sigma, float)
    else:
        series = pd.Series(high_frequency)
        min_periods = max(3, local_window // 3)
        local_median = series.rolling(local_window, center=True, min_periods=min_periods).median().to_numpy(float)
        local_mad = pd.Series(np.abs(high_frequency - local_median)).rolling(
            local_window, center=True, min_periods=min_periods
        ).median().to_numpy(float)
        local_sigma = np.maximum(1.4826 * np.nan_to_num(local_mad, nan=global_sigma), 0.50 * global_sigma)
    local_sigma = np.maximum(local_sigma, 5e-4)
    detection_score = np.abs(residual) / local_sigma

    spacing = float(grid[1] - grid[0])
    min_distance = max(2, int(round(0.003 / max(spacing, 1e-12))))
    seed_prominence = max(2.0 * global_sigma, 0.003)
    raw: list[dict[str, float | str | bool]] = []
    for extrema, info, kind, signed in (
        (find_peaks(residual, prominence=seed_prominence, distance=min_distance)[0], None, "peak", residual),
        (find_peaks(-residual, prominence=seed_prominence, distance=min_distance)[0], None, "dip", -residual),
    ):
        # Re-run only for the retained extrema so SciPy provides the exact
        # bases/prominence data used for the physical half-prominence width.
        if extrema.size == 0:
            continue
        extrema, info = find_peaks(signed, prominence=seed_prominence, distance=min_distance)
        widths, _heights, left_ips, right_ips = peak_widths(
            signed, extrema, rel_height=0.5,
            prominence_data=(info["prominences"], info["left_bases"], info["right_bases"]),
        )
        for peak, prominence, left_ip, right_ip, width_samples in zip(
            extrema, info["prominences"], left_ips, right_ips, widths
        ):
            if peak < min_distance or peak >= grid.size - min_distance:
                continue
            left_ev = float(np.interp(left_ip, np.arange(grid.size), grid))
            right_ev = float(np.interp(right_ip, np.arange(grid.size), grid))
            width_ev = max(0.0, right_ev - left_ev)
            if not 0.0015 <= width_ev <= 0.030:
                continue
            local = float(local_sigma[int(peak)])
            snr = float(prominence) / max(local, 1e-12)
            if snr < 4.0 or float(prominence) < 0.005:
                continue
            recommended = snr >= 6.0 and float(prominence) >= 0.010
            confidence = float(np.clip(0.5 * (snr / 6.0) + 0.5 * (float(prominence) / 0.010), 0.0, 1.0))
            raw.append({
                "center": float(grid[int(peak)]), "start": max(float(grid[0]), left_ev - guard_ev),
                "stop": min(float(grid[-1]), right_ev + guard_ev), "kind": kind,
                "prominence": float(prominence), "snr": snr, "width": width_ev,
                "confidence": confidence, "recommended": recommended,
            })

    # A resonance can give a weak opposite-sign shoulder after detrending.
    # Cluster it with its dominant component instead of protecting two tiny,
    # visually confusing intervals.  Comparisons are always to the dominant
    # anchor, preventing a long chain of unrelated ripples from being merged.
    clusters: list[list[dict[str, float | str | bool]]] = []
    for candidate in sorted(raw, key=lambda item: float(item["prominence"]), reverse=True):
        attached = False
        for cluster in clusters:
            anchor = cluster[0]
            proximity = max(0.010, float(anchor["width"]) + float(candidate["width"]) + guard_ev)
            overlaps = float(candidate["start"]) <= float(anchor["stop"]) + guard_ev and float(candidate["stop"]) >= float(anchor["start"]) - guard_ev
            if overlaps or abs(float(candidate["center"]) - float(anchor["center"])) <= proximity:
                cluster.append(candidate)
                attached = True
                break
        if not attached:
            clusters.append([candidate])

    detected: list[McdReflectionFeature] = []
    for cluster in clusters:
        anchor = cluster[0]
        # Preserve the dominant component's physical sign; nearby opposite
        # shoulders are diagnostic context, not a separate resonance type.
        kind: Literal["peak", "dip", "peak/dip"] = str(anchor["kind"])  # type: ignore[assignment]
        # A nearby shoulder is classified with the resonance, but must not
        # inflate its protection band unless its measured half-prominence
        # support actually overlaps the dominant component.
        support_members = [
            item for item in cluster
            if float(item["start"]) <= float(anchor["stop"]) and float(item["stop"]) >= float(anchor["start"])
        ]
        detected.append(McdReflectionFeature(
            center_ev=float(anchor["center"]),
            start_ev=min(float(item["start"]) for item in support_members),
            stop_ev=max(float(item["stop"]) for item in support_members),
            kind=kind,
            prominence_log=float(anchor["prominence"]),
            snr=float(anchor["snr"]),
            width_ev=float(anchor["width"]),
            confidence=float(anchor["confidence"]),
            recommended=bool(anchor["recommended"]),
        ))
    detected.sort(key=lambda item: item.center_ev)
    return tuple(detected), np.interp(energy, grid, baseline), np.interp(energy, grid, residual), np.interp(energy, grid, detection_score)


def _cross_validated_spectral_rms(
    pos: np.ndarray,
    neg: np.ndarray,
    energy_ev: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    order: int,
) -> float:
    """Hold out every third candidate point to score a spectral model.

    This deliberately validates on points not used by the fit.  It is not a
    physical-MCD minimisation: only the user-reviewable candidate background
    mask participates.
    """
    mask = np.asarray(candidate_mask, bool)
    indices = np.flatnonzero(mask)
    if indices.size < 12:
        return float("nan")
    validation = np.zeros(mask.size, dtype=bool)
    validation[indices[::3]] = True
    train = mask & ~validation
    center = float(np.nanmedian(np.asarray(energy_ev, float)))
    values: list[float] = []
    for pos_row, neg_row in zip(np.asarray(pos, float), np.asarray(neg, float)):
        coefficients = _robust_log_ratio_coefficients(energy_ev, pos_row, neg_row, train, order=order, center_ev=center)
        if coefficients is None:
            continue
        x = np.asarray(energy_ev, float) - center
        log_gain = sum(float(value) * x ** power for power, value in enumerate(coefficients))
        corrected = neg_row * np.exp(np.clip(log_gain, np.log(0.2), np.log(5.0)))
        valid = validation & np.isfinite(pos_row) & np.isfinite(corrected)
        if np.count_nonzero(valid) < 3:
            continue
        normalizer = max(float(np.nanmedian(np.abs(pos_row[valid]))), 1e-30)
        values.append(float(np.sqrt(np.nanmean((pos_row[valid] - corrected[valid]) ** 2)) / normalizer))
    return float(np.nanmedian(values)) if values else float("nan")


def suggest_mcd_background_ranges(
    result: McdResult,
    *,
    protected_ranges_ev: tuple[tuple[float, float], ...] = (),
    min_band_width_mev: float = 5.0,
    max_bands: int = 3,
    auto_detect_features: bool = True,
    use_all_unprotected_bands: bool = False,
) -> McdBackgroundSuggestion:
    """Suggest conservative, editable drift-fit regions from the full sweep.

    Feature detection uses the full-sweep median reflectance and local
    roughness, not the corrected MCD amplitude.  This avoids selecting bands
    merely because a correction could force their MCD to zero.
    """
    raw_energy = EV_NM / np.asarray(result.wavelength_nm, float)
    order = np.argsort(raw_energy)
    energy = raw_energy[order]
    pos = np.asarray(result.pair_corrected_pos, float)[:, order]
    global_neg = (np.asarray(result.pair_raw_neg, float) - np.asarray(result.dark_neg, float)) * np.asarray(result.gain, float)
    neg = global_neg[:, order]
    average = np.nanmedian(0.5 * (pos + neg), axis=0)
    finite = np.isfinite(energy) & np.isfinite(average) & (average > 0)
    if np.count_nonzero(finite) < 12:
        raise ValueError("The MCD sweep does not contain enough finite reflection points to suggest background ranges.")
    count = int(np.count_nonzero(finite))
    smooth_window = min(31, count if count % 2 else count - 1)
    smooth_window = max(5, smooth_window if smooth_window % 2 else smooth_window - 1)
    log_average = np.log(np.maximum(average, np.nanmin(average[finite]) * 1e-12))
    if not np.all(finite):
        log_average = np.asarray(log_average, float).copy()
        log_average[~finite] = np.interp(energy[~finite], energy[finite], log_average[finite])
    smoothed = savgol_filter(log_average, smooth_window, min(3, smooth_window - 1), mode="interp")
    first = np.gradient(smoothed, energy)
    curvature = np.abs(np.gradient(first, energy))
    # Local angle-ratio roughness flags detector spikes and unstable regions,
    # while preserving smooth B-dependent throughput that the correction can fit.
    ratio = np.log(np.where((pos > 1e-30) & (neg > 1e-30), pos / neg, np.nan))
    row_smooth = np.vstack([
        savgol_filter(row, smooth_window, min(3, smooth_window - 1), mode="interp")
        if np.count_nonzero(np.isfinite(row)) >= smooth_window else row
        for row in np.where(np.isfinite(ratio), ratio, np.nanmedian(ratio, axis=1, keepdims=True))
    ])
    roughness = np.nanmedian(np.abs(ratio - row_smooth), axis=0)
    curvature_scale = max(float(np.nanmedian(curvature[finite])), 1e-30)
    roughness_scale = max(float(np.nanmedian(roughness[finite])), 1e-30)
    suitability = curvature / curvature_scale + roughness / roughness_scale
    requested_protected = _merge_energy_ranges(tuple(protected_ranges_ev))
    # Reflection features are found on a broad-baseline residual.  Only
    # high-confidence candidates participate in the initial suggested fit;
    # weaker candidates remain visible in the review dialog for manual use.
    if auto_detect_features:
        detected_features, feature_baseline, feature_residual, feature_score = _detect_reflection_features(
            energy, log_average, finite,
            guard_ev=max(0.0025, 1.5 * float(np.nanmedian(np.diff(energy)))),
        )
    else:
        detected_features = ()
        feature_baseline = np.asarray(log_average, float)
        feature_residual = np.zeros_like(feature_baseline)
        feature_score = np.zeros_like(feature_baseline)
    automatic_protected = tuple(
        (feature.start_ev, feature.stop_ev)
        for feature in detected_features if feature.recommended
    )
    automatic_kinds = tuple(feature.kind for feature in detected_features if feature.recommended)
    protected = _merge_energy_ranges(requested_protected + automatic_protected)
    protected_mask = _mask_from_ranges(energy, protected)
    min_width_ev = max(float(min_band_width_mev) * 1e-3, 3.0 * float(np.nanmedian(np.diff(energy))))
    feature_center = float(np.nanmedian([value for pair in requested_protected for value in pair])) if requested_protected else float(np.nanmedian(energy))
    selected: list[tuple[float, float]] = []
    left: list[object] = []
    right: list[object] = []
    if use_all_unprotected_bands and protected:
        # In the user-directed workflow, protection windows are authoritative.
        # Every sufficiently wide, finite interval outside those windows is a
        # background band, including gaps between separated resonances.  The
        # suitability trace remains diagnostic and does not silently remove an
        # edge or a clean middle interval selected by the user.
        selected.extend(_ranges_from_mask(
            energy, finite & ~protected_mask, min_width_ev=min_width_ev,
        ))
        if not selected:
            raise ValueError("No usable background bands remain outside the selected protection windows.")
        left = [region for region in selected if region[1] < feature_center]
        right = [region for region in selected if region[0] > feature_center]
    else:
        signal_floor = float(np.nanpercentile(average[finite], 5.0))
        candidate = finite & ~protected_mask & (average >= signal_floor)
        # Keep the smoother half of the full spectrum.  A later segment filter
        # ensures that isolated low-score pixels cannot become fit regions.
        threshold = float(np.nanpercentile(suitability[candidate], 55.0)) if np.any(candidate) else float("inf")
        candidate &= suitability <= threshold
        candidate = _bridge_short_mask_gaps(energy, candidate, max_gap_ev=0.002)
        candidate &= ~protected_mask
        ranges_all = _ranges_from_mask(energy, candidate, min_width_ev=min_width_ev)
        if not ranges_all:
            raise ValueError("No continuous low-feature background bands were found. Choose background ranges manually.")
        range_scores = []
        for region in ranges_all:
            in_region = _mask_from_ranges(energy, (region,))
            range_scores.append((float(np.nanmean(suitability[in_region])), -(region[1] - region[0]), region))
        left = [item for item in range_scores if item[2][1] < feature_center]
        right = [item for item in range_scores if item[2][0] > feature_center]
        if left:
            selected.append(min(left)[2])
        if right:
            selected.append(min(right)[2])
        for _score, _negative_width, region in sorted(range_scores):
            if region not in selected and len(selected) < max(2, int(max_bands)):
                selected.append(region)
    selected = sorted(selected)
    selected_mask = _mask_from_ranges(energy, tuple(selected))
    full_span = max(float(energy[-1] - energy[0]), 1e-30)
    coverage = float(np.count_nonzero(selected_mask) / np.count_nonzero(finite))
    selected_span = float(energy[selected_mask].max() - energy[selected_mask].min()) if np.any(selected_mask) else 0.0
    linear_rms = _cross_validated_spectral_rms(pos, neg, energy, selected_mask, order=1)
    quadratic_rms = _cross_validated_spectral_rms(pos, neg, energy, selected_mask, order=2)
    suggested_order: Literal[1, 2] = 1
    if np.isfinite(quadratic_rms) and np.isfinite(linear_rms) and quadratic_rms < 0.85 * linear_rms:
        suggested_order = 2
    notes: list[str] = []
    if use_all_unprotected_bands:
        notes.append("All sufficiently wide unprotected intervals are background bands; suitability is diagnostic only.")
    if not left or not right:
        notes.append("Only one side of the protected feature has a usable suggested band; review before using a spectral fit.")
    if selected_span < 0.25 * full_span:
        notes.append("Suggested bands span less than 25% of the energy axis; a spectral fit may be unreliable.")
    if not np.isfinite(linear_rms) or not np.isfinite(quadratic_rms):
        notes.append("Too few suggested points for a reliable held-out model comparison.")
    if suggested_order == 2:
        notes.append("Quadratic was selected only because it improved held-out background RMS by at least 15%.")
    else:
        notes.append("Linear is recommended because the quadratic validation improvement was not large enough.")
    if detected_features:
        recommended_count = sum(feature.recommended for feature in detected_features)
        notes.append(
            f"Found {len(detected_features)} reflection feature candidate(s); "
            f"{recommended_count} high-confidence window(s) are enabled automatically."
        )
    return McdBackgroundSuggestion(
        energy_ev=energy, median_reflectance=average, suitability=suitability,
        feature_baseline=feature_baseline, feature_residual=feature_residual,
        feature_detection_score=feature_score,
        ranges=tuple(selected), requested_protected_ranges=requested_protected,
        detected_feature_ranges=tuple((feature.start_ev, feature.stop_ev) for feature in detected_features),
        detected_feature_kinds=tuple(feature.kind for feature in detected_features),
        protected_ranges=protected, suggested_order=suggested_order,
        linear_validation_rms=linear_rms, quadratic_validation_rms=quadratic_rms,
        coverage_fraction=coverage, span_fraction=selected_span / full_span, notes=tuple(notes),
        detected_features=detected_features,
    )


def _apply_pair_correction(
    pos: np.ndarray,
    neg: np.ndarray,
    energy_ev: np.ndarray,
    settings: McdSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correct pair-to-pair drift using only declared background energy regions."""
    count = pos.shape[0]
    scale = np.ones(count, float)
    offset = np.zeros(count, float)
    slope = np.zeros(count, float)
    curvature = np.zeros(count, float)
    correction_min = np.ones(count, float)
    correction_max = np.ones(count, float)
    residual_before = np.full(count, np.nan, float)
    residual = np.full(count, np.nan, float)
    if settings.correction_mode == "global":
        return neg, scale, offset, slope, curvature, correction_min, correction_max, residual_before, residual
    mask = _background_mask(energy_ev, settings.background_ranges_ev)
    energy = np.asarray(energy_ev, float)
    finite_energy = energy[np.isfinite(energy)]
    energy_center = float(np.nanmedian(finite_energy))
    if settings.correction_mode == "pair_spectral":
        selected_energy = energy[mask & np.isfinite(energy)]
        full_span = float(np.nanmax(finite_energy) - np.nanmin(finite_energy))
        selected_span = float(np.nanmax(selected_energy) - np.nanmin(selected_energy))
        if full_span > 0 and selected_span < 0.25 * full_span:
            raise ValueError(
                "Spectral correction background regions must span at least 25% of the energy range. "
                "Select separated non-resonant regions on both sides of the feature of interest."
            )
    adjusted = np.asarray(neg, float).copy()
    for index, (pos_row, neg_row) in enumerate(zip(pos, neg)):
        valid = mask & np.isfinite(pos_row) & np.isfinite(neg_row)
        if np.count_nonzero(valid) < 3:
            continue
        x = neg_row[valid]
        y = pos_row[valid]
        normalizer = max(float(np.nanmedian(np.abs(y))), 1e-30)
        residual_before[index] = float(np.sqrt(np.nanmean((y - x) ** 2)) / normalizer)
        if settings.correction_mode == "pair_spectral":
            order = int(np.clip(settings.spectral_order, 1, 2))
            coefficients = _robust_log_ratio_coefficients(
                energy, pos_row, neg_row, valid, order=order, center_ev=energy_center,
            )
            if coefficients is None:
                continue
            full_x = energy - energy_center
            log_correction = sum(float(value) * full_x ** power for power, value in enumerate(coefficients))
            spectral_gain = np.exp(np.clip(log_correction, np.log(0.2), np.log(5.0)))
            if not np.all(np.isfinite(spectral_gain)):
                continue
            adjusted[index] = neg_row * spectral_gain
            scale[index] = float(np.exp(coefficients[0]))
            slope[index] = float(coefficients[1])
            curvature[index] = float(coefficients[2]) if order == 2 else 0.0
            correction_min[index] = float(np.nanmin(spectral_gain))
            correction_max[index] = float(np.nanmax(spectral_gain))
        elif settings.correction_mode == "pair_affine":
            coefficient, constant = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
            scale[index], offset[index] = float(coefficient), float(constant)
            adjusted[index] = scale[index] * neg_row + offset[index]
        else:
            denominator = float(np.dot(x, x))
            scale[index] = float(np.dot(x, y) / denominator) if abs(denominator) > 1e-30 else 1.0
            adjusted[index] = scale[index] * neg_row
        residual[index] = float(np.sqrt(np.nanmean((pos_row[valid] - adjusted[index, valid]) ** 2)) / normalizer)
    return adjusted, scale, offset, slope, curvature, correction_min, correction_max, residual_before, residual


def _zero_reference(
    values: np.ndarray,
    b: np.ndarray,
    window: float,
    mode: Literal["nearest", "window"],
) -> tuple[np.ndarray, float]:
    if mode == "nearest":
        index = int(np.argmin(np.abs(b)))
        return np.asarray(values[index], float), float(b[index])
    mask = np.abs(b) <= float(window)
    if not np.any(mask):
        raise ValueError("No paired spectra lie inside the near-zero reference window. Increase the reference window or use nearest-pair reference.")
    return np.nanmedian(values[mask], axis=0), float(np.nanmedian(b[mask]))


def _gain_from_references(pos: np.ndarray, neg: np.ndarray, settings: McdSettings) -> np.ndarray:
    gain = pos / np.where(np.abs(neg) > 1e-30, neg, np.nan)
    finite = np.isfinite(gain)
    if not np.any(finite):
        raise ValueError("Could not calculate an angle gain from the zero-field references.")
    replacement = float(np.nanmedian(gain[finite]))
    gain = np.where(finite, gain, replacement)
    if settings.gain_mode == "scalar":
        return np.full_like(gain, float(np.nanmedian(gain)))
    if settings.gain_mode == "smoothed" and gain.size >= 5:
        window = min(int(settings.gain_smooth_window), gain.size if gain.size % 2 else gain.size - 1)
        window = max(3, window if window % 2 else window - 1)
        poly = min(int(settings.gain_smooth_poly), window - 1)
        return savgol_filter(gain, window, poly, mode="interp")
    return gain


def _bin_map(b: np.ndarray, z: np.ndarray, wavelength_nm: np.ndarray, decimals: int, name: str) -> DataCube | None:
    if not b.size:
        return None
    rounded = np.round(np.asarray(b, float), decimals)
    unique = np.unique(rounded)
    rows = [np.nanmean(z[rounded == value], axis=0) for value in unique]
    return _cube(np.asarray(unique, float), np.vstack(rows), wavelength_nm, name)


def _cube(b: np.ndarray, spectra: np.ndarray, wavelength_nm: np.ndarray, name: str) -> DataCube:
    energy = EV_NM / np.asarray(wavelength_nm, float)
    order = np.argsort(energy)
    return DataCube(
        energy[order],
        np.asarray(b, float),
        np.asarray(spectra, float)[:, order],
        "B field (T)",
        f"MCD {name}",
        "MCD",
    )


def _odd_map(cube: DataCube, name: str, tolerance: float) -> DataCube | None:
    b = np.asarray(cube.gate, float)
    z = np.asarray(cube.Z, float)
    positives = np.flatnonzero(b > 0)
    negatives = np.flatnonzero(b < 0)
    used: set[int] = set()
    fields: list[float] = []
    rows: list[np.ndarray] = []
    for pos_index in positives[np.argsort(b[positives])]:
        candidates = negatives[np.argsort(np.abs(b[negatives] + b[pos_index]))]
        candidate = next((index for index in candidates if index not in used and abs(b[index] + b[pos_index]) <= tolerance), None)
        if candidate is None:
            continue
        used.add(candidate)
        fields.append(0.5 * (b[pos_index] - b[candidate]))
        rows.append(0.5 * (z[pos_index] - z[candidate]))
    if not rows:
        return None
    return DataCube(cube.energy.copy(), np.asarray(fields), np.vstack(rows), "B field (T)", f"MCD {name}", "MCD")


def process_mcd(path: str, settings: McdSettings | None = None) -> McdResult:
    settings = settings or McdSettings()
    b, angle, wavelength, spectra = load_b_sweep_csv(path)
    pair_b, pair_b_pos, pair_b_neg, pair_gap, pair_pos_index, pair_neg_index, i_pos, i_neg = _pair_angles(b, angle, spectra, settings)
    pair_interpolated_pos = np.zeros(pair_b.size, dtype=bool)
    pair_interpolated_neg = np.zeros(pair_b.size, dtype=bool)
    if settings.pair_b_alignment == "interpolate":
        i_pos, pair_interpolated_pos = _interpolate_pair_spectra(b, angle, spectra, pair_pos_index, pair_b)
        i_neg, pair_interpolated_neg = _interpolate_pair_spectra(b, angle, spectra, pair_neg_index, pair_b)
    pos_angle = float(settings.pos_angle) if settings.pos_angle is not None else float(np.max(np.unique(angle)))
    neg_angle = float(settings.neg_angle) if settings.neg_angle is not None else float(np.min(np.unique(angle)))
    dark_pos = _load_reference_file(settings.dark_pos_file, wavelength) if settings.dark_pos_file else np.zeros_like(wavelength)
    dark_neg = _load_reference_file(settings.dark_neg_file, wavelength) if settings.dark_neg_file else np.zeros_like(wavelength)
    corrected_pos = i_pos - dark_pos
    corrected_neg = i_neg - dark_neg
    ref_pos, reference_b = _zero_reference(corrected_pos, pair_b, settings.zero_window_t, settings.reference_mode)
    ref_neg, _reference_b_neg = _zero_reference(corrected_neg, pair_b, settings.zero_window_t, settings.reference_mode)
    gain = _gain_from_references(ref_pos, ref_neg, settings)
    scaled_neg = corrected_neg * gain
    energy_ev = EV_NM / wavelength
    (
        final_neg, pair_scale, pair_offset, pair_spectral_slope, pair_spectral_curvature,
        pair_correction_min, pair_correction_max, pair_background_rms_before, pair_background_rms,
    ) = _apply_pair_correction(corrected_pos, scaled_neg, energy_ev, settings)
    denominator = corrected_pos + final_neg
    raw_denominator = i_pos + i_neg
    mcd_raw = (i_pos - i_neg) / np.where(np.abs(raw_denominator) > 1e-30, raw_denominator, np.nan)
    mcd_pairs = (corrected_pos - final_neg) / np.where(np.abs(denominator) > 1e-30, denominator, np.nan)
    # The normalised angle responses are retained as a diagnostic map.
    norm_pos = corrected_pos / np.where(np.abs(ref_pos) > 1e-30, ref_pos, np.nan)
    scaled_ref_neg = ref_neg * gain
    norm_neg = final_neg / np.where(np.abs(scaled_ref_neg) > 1e-30, scaled_ref_neg, np.nan)
    normalised_mcd = (norm_pos - norm_neg) / np.where(np.abs(norm_pos + norm_neg) > 1e-30, norm_pos + norm_neg, np.nan)
    pair_labels = _pair_sweep_labels(pair_b)
    increasing = pair_labels == "B increasing"
    decreasing = pair_labels == "B decreasing"
    combo = _bin_map(pair_b, mcd_pairs, wavelength, settings.bin_decimals, "Combo")
    assert combo is not None
    raw_combo = _bin_map(pair_b, mcd_raw, wavelength, settings.bin_decimals, "Raw")
    maps: dict[str, DataCube] = {
        "Combo": combo,
        "Raw": raw_combo or combo,
        "Normalized": _bin_map(pair_b, normalised_mcd, wavelength, settings.bin_decimals, "Normalized") or combo,
    }
    increasing_map = _bin_map(pair_b[increasing], mcd_pairs[increasing], wavelength, settings.bin_decimals, "B increasing")
    decreasing_map = _bin_map(pair_b[decreasing], mcd_pairs[decreasing], wavelength, settings.bin_decimals, "B decreasing")
    if increasing_map is not None:
        maps["B increasing"] = increasing_map
    if decreasing_map is not None:
        maps["B decreasing"] = decreasing_map
    if increasing_map is not None and decreasing_map is not None:
        fields = sorted(set(np.asarray(increasing_map.gate, float)) & set(np.asarray(decreasing_map.gate, float)))
        if fields:
            increasing_index = {float(value): index for index, value in enumerate(increasing_map.gate)}
            decreasing_index = {float(value): index for index, value in enumerate(decreasing_map.gate)}
            maps["Sweep average"] = DataCube(combo.energy.copy(), np.asarray(fields), np.vstack([0.5 * (increasing_map.Z[increasing_index[value]] + decreasing_map.Z[decreasing_index[value]]) for value in fields]), "B field (T)", "MCD Sweep average", "MCD")
    tolerance = 2.5 * 10.0 ** (-settings.bin_decimals)
    odd = _odd_map(combo, "Odd (exact)", tolerance=1e-12)
    odd_tolerance = _odd_map(combo, "Odd (tolerance)", tolerance=tolerance)
    if odd is not None:
        maps["Odd exact"] = odd
    if odd_tolerance is not None:
        maps["Odd tolerance"] = odd_tolerance
    # Same-angle field flips are a diagnostic independent of waveplate-to-
    # waveplate throughput.  Each angle was first normalised by its own B≈0
    # reference, then its +B/-B odd component is taken.
    field_flip_pos = _bin_map(pair_b, norm_pos - 1.0, wavelength, settings.bin_decimals, "Field flip +")
    field_flip_neg = _bin_map(pair_b, norm_neg - 1.0, wavelength, settings.bin_decimals, "Field flip -")
    if field_flip_pos is not None:
        diagnostic = _odd_map(field_flip_pos, "Field flip +", tolerance=tolerance)
        if diagnostic is not None:
            maps["Field flip +"] = diagnostic
    if field_flip_neg is not None:
        diagnostic = _odd_map(field_flip_neg, "Field flip -", tolerance=tolerance)
        if diagnostic is not None:
            maps["Field flip -"] = diagnostic
    return McdResult(
        source_file=str(path), wavelength_nm=wavelength, energy_ev=combo.energy.copy(), pos_angle=pos_angle,
        neg_angle=neg_angle, pair_b=pair_b, pair_b_pos=pair_b_pos, pair_b_neg=pair_b_neg,
        pair_delta_b=pair_b_pos - pair_b_neg, pair_sequence_gap=pair_gap,
        pair_interpolated_pos=pair_interpolated_pos, pair_interpolated_neg=pair_interpolated_neg,
        pair_labels=pair_labels,
        pair_raw_pos=i_pos, pair_raw_neg=i_neg, pair_corrected_pos=corrected_pos,
        pair_corrected_neg=final_neg, pair_mcd_raw=mcd_raw, pair_mcd_corrected=mcd_pairs,
        pair_scale=pair_scale, pair_offset=pair_offset,
        pair_spectral_slope=pair_spectral_slope, pair_spectral_curvature=pair_spectral_curvature,
        pair_correction_min=pair_correction_min, pair_correction_max=pair_correction_max,
        pair_background_rms_before=pair_background_rms_before, pair_background_rms=pair_background_rms,
        reference_b=reference_b, gain=gain,
        reference_pos=ref_pos, reference_neg=ref_neg, dark_pos=dark_pos, dark_neg=dark_neg, maps=maps,
        summary={"pairs": int(pair_b.size), "reference_mode": settings.reference_mode, "reference_b_t": reference_b, "zero_pairs": int(np.count_nonzero(np.abs(pair_b) <= settings.zero_window_t)), "gain_mode": settings.gain_mode, "correction_mode": settings.correction_mode, "spectral_order": settings.spectral_order, "pair_b_alignment": settings.pair_b_alignment, "background_ranges_ev": settings.background_ranges_ev, "background_selection": settings.background_selection, "max_delta_b_t": settings.max_delta_b, "bin_decimals": settings.bin_decimals, "odd_tolerance_t": tolerance},
    )


WindowMetric = Literal["mean", "absolute_mean", "field_signed_absolute_mean", "integral"]


def _window_trace_from_cube(
    cube: DataCube,
    center_ev: float,
    width_mev: float,
    *,
    metric: WindowMetric,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one energy window to an MCD(B) trace without changing its field grid."""
    energy = np.asarray(cube.energy, float)
    half = float(width_mev) * 5e-4
    mask = np.abs(energy - float(center_ev)) <= half
    if not np.any(mask):
        mask[int(np.argmin(np.abs(energy - float(center_ev))))] = True
    values = np.asarray(cube.Z, float)[:, mask]
    if metric == "integral":
        # np.trapz was removed in NumPy 2.0; trapezoid is its supported replacement.
        reduced = np.trapezoid(values, x=energy[mask], axis=1)
    elif metric == "absolute_mean":
        reduced = np.nanmean(np.abs(values), axis=1)
    elif metric == "field_signed_absolute_mean":
        reduced = np.sign(np.asarray(cube.gate, float)) * np.nanmean(np.abs(values), axis=1)
    else:
        reduced = np.nanmean(values, axis=1)
    return np.asarray(cube.gate, float), reduced


def window_trace(
    result: McdResult,
    map_name: str,
    center_ev: float,
    width_mev: float,
    *,
    metric: WindowMetric = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    return _window_trace_from_cube(result.cube(map_name), center_ev, width_mev, metric=metric)


def window_trace_comparison(
    result: McdResult,
    map_name: str,
    center_ev: float,
    width_mev: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return raw and corrected MCD(B) traces for every supported window metric."""
    raw_cube = result.maps.get("Raw", result.cube(map_name))
    corrected_cube = result.cube(map_name)
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for source, cube in (("raw", raw_cube), ("corrected", corrected_cube)):
        for metric in ("mean", "field_signed_absolute_mean", "absolute_mean", "integral"):
            traces[f"{source}_{metric}"] = _window_trace_from_cube(
                cube, center_ev, width_mev, metric=metric
            )
    return traces


def _pair_window_metric(
    result: McdResult,
    spectra: np.ndarray,
    *,
    center_ev: float,
    width_mev: float,
    metric: WindowMetric,
) -> np.ndarray:
    """Reduce every acquired pair without binning different sweep branches."""
    energy = EV_NM / np.asarray(result.wavelength_nm, float)
    order = np.argsort(energy)
    energy = energy[order]
    values = np.asarray(spectra, float)[:, order]
    half = float(width_mev) * 5e-4
    mask = np.abs(energy - float(center_ev)) <= half
    if not np.any(mask):
        mask[int(np.argmin(np.abs(energy - float(center_ev))))] = True
    selected = values[:, mask]
    if metric == "integral":
        return np.trapezoid(selected, x=energy[mask], axis=1)
    if metric == "absolute_mean":
        return np.nanmean(np.abs(selected), axis=1)
    if metric == "field_signed_absolute_mean":
        return np.sign(np.asarray(result.pair_b, float)) * np.nanmean(np.abs(selected), axis=1)
    return np.nanmean(selected, axis=1)


def pair_window_trace_by_branch(
    result: McdResult,
    center_ev: float,
    width_mev: float,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return raw/corrected MCD(B) values for each acquired B-sweep branch.

    Unlike :func:`window_trace_comparison`, this intentionally does not use a
    binned map.  Therefore repeated B values on increasing and decreasing
    sweeps remain independent values for plotting and Origin export.
    """
    all_values: dict[str, np.ndarray] = {}
    for source, spectra in (("raw", result.pair_mcd_raw), ("corrected", result.pair_mcd_corrected)):
        for metric in ("mean", "field_signed_absolute_mean", "absolute_mean", "integral"):
            all_values[f"{source}_{metric}"] = _pair_window_metric(
                result, spectra, center_ev=center_ev, width_mev=width_mev, metric=metric
            )
    labels = np.asarray(result.pair_labels, dtype=str)
    branches: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for branch in ("B increasing", "B decreasing"):
        branch_mask = labels == branch
        branch_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, values in all_values.items():
            branch_values[name] = (
                np.asarray(result.pair_b, float)[branch_mask],
                np.asarray(values, float)[branch_mask],
            )
        branches[branch] = branch_values
    return branches


def export_mcd_tables(result: McdResult, output_dir: str, *, trace_map: str, center_ev: float, width_mev: float, metric: WindowMetric = "mean") -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source_file).stem
    paths: dict[str, Path] = {}
    for name, cube in result.maps.items():
        table = pd.DataFrame(np.asarray(cube.Z, float), columns=[f"{value:.9g}" for value in cube.energy])
        table.insert(0, "B_T", np.asarray(cube.gate, float))
        path = out / f"{stem}_MCD_{name.replace(' ', '_')}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    b, values = window_trace(result, trace_map, center_ev, width_mev, metric=metric)
    trace_path = out / f"{stem}_MCD_window_{trace_map.replace(' ', '_')}.csv"
    pd.DataFrame({"B_T": b, "MCD": values, "metric": metric}).to_csv(trace_path, index=False)
    paths["window_trace"] = trace_path
    traces = window_trace_comparison(result, trace_map, center_ev, width_mev)
    field = np.unique(np.concatenate([b_values for b_values, _values in traces.values()]))
    comparison = pd.DataFrame({"B_T": field})
    for name, (b_values, values) in traces.items():
        lookup = {float(b_value): value for b_value, value in zip(b_values, values)}
        comparison[name] = [lookup.get(float(b_value), np.nan) for b_value in field]
    comparison_path = out / f"{stem}_MCD_window_comparison_{trace_map.replace(' ', '_')}.csv"
    comparison.to_csv(comparison_path, index=False)
    paths["window_comparison"] = comparison_path
    diagnostic_path = out / f"{stem}_MCD_pair_diagnostics.csv"
    pd.DataFrame({
        "pair_index": np.arange(result.pair_b.size),
        "Bpair_T": result.pair_b,
        "B_sigma_plus_T": result.pair_b_pos,
        "B_sigma_minus_T": result.pair_b_neg,
        "delta_B_T": result.pair_delta_b,
        "sequence_gap_rows": result.pair_sequence_gap,
        "sigma_plus_interpolated": result.pair_interpolated_pos,
        "sigma_minus_interpolated": result.pair_interpolated_neg,
        "sweep": result.pair_labels,
        "pair_scale": result.pair_scale,
        "pair_offset": result.pair_offset,
        "spectral_log_slope_per_eV": result.pair_spectral_slope,
        "spectral_log_curvature_per_eV2": result.pair_spectral_curvature,
        "spectral_correction_min": result.pair_correction_min,
        "spectral_correction_max": result.pair_correction_max,
        "background_relative_rms_before": result.pair_background_rms_before,
        "background_relative_rms": result.pair_background_rms,
    }).to_csv(diagnostic_path, index=False)
    paths["pair_diagnostics"] = diagnostic_path
    return paths


def _mcd_window_export_tag(center_ev: float, width_mev: float) -> str:
    """Return a readable filename tag without changing decimal points."""
    return f"E{float(center_ev):.6f}eV_W{float(width_mev):.6g}meV"


def _mcd_trace_comparison_table(
    result: McdResult,
    trace_map: str,
    center_ev: float,
    width_mev: float,
) -> pd.DataFrame:
    del trace_map  # MCD(B) deliberately uses unbinned acquired pairs.
    traces = pair_window_trace_by_branch(result, center_ev, width_mev)
    # Two X/Y blocks make direct Origin plotting possible without filtering a
    # mixed branch column.  Pandas pads the shorter branch with blank cells.
    table = pd.DataFrame()
    for branch, suffix in (("B increasing", "increasing"), ("B decreasing", "decreasing")):
        branch_traces = traces[branch]
        b_values, _ = branch_traces["corrected_mean"]
        table[f"B_{suffix}_T"] = pd.Series(b_values)
        for metric, name in (
            ("mean", "signed_mean"),
            ("field_signed_absolute_mean", "field_signed_absolute_mean"),
            ("integral", "integral"),
        ):
            _b_values, values = branch_traces[f"corrected_{metric}"]
            table[f"corrected_{name}_{suffix}"] = pd.Series(values)
    return table


def _mcd_pair_diagnostic_table(result: McdResult) -> pd.DataFrame:
    return pd.DataFrame({
        "pair_index": np.arange(result.pair_b.size),
        "Bpair_T": result.pair_b,
        "B_sigma_plus_T": result.pair_b_pos,
        "B_sigma_minus_T": result.pair_b_neg,
        "delta_B_T": result.pair_delta_b,
        "sequence_gap_rows": result.pair_sequence_gap,
        "sigma_plus_interpolated": result.pair_interpolated_pos,
        "sigma_minus_interpolated": result.pair_interpolated_neg,
        "sweep": result.pair_labels,
        "pair_scale": result.pair_scale,
        "pair_offset": result.pair_offset,
        "spectral_log_slope_per_eV": result.pair_spectral_slope,
        "spectral_log_curvature_per_eV2": result.pair_spectral_curvature,
        "spectral_correction_min": result.pair_correction_min,
        "spectral_correction_max": result.pair_correction_max,
        "background_relative_rms_before": result.pair_background_rms_before,
        "background_relative_rms": result.pair_background_rms,
    })


def _mcd_export_title(source_file: str) -> str:
    """Use the same basic filename-to-title treatment as the PL exporter."""
    return Path(source_file).stem.replace("~", " ").replace("_", " ").strip()


def _mcd_export_metadata(source_file: str) -> str:
    """Extract the useful sample metadata without consuming the whole figure."""
    tokens = [part for part in Path(source_file).stem.split("_") if part]
    selected: list[str] = []
    for token in tokens:
        low = token.lower()
        if re.fullmatch(r"[A-Za-z]+\d+", token) and not selected:
            selected.append(token)
        elif "k" in low and any(char.isdigit() for char in token) and not any("k" in item.lower() for item in selected):
            selected.append(token)
        elif token.startswith(("Vbg=", "Vtg=", "E=", "D=")):
            selected.append(token)
        elif re.match(r"\d+(?:\.\d+)?nm", low):
            selected.append(token.replace("nmc", "nm"))
    return " | ".join(selected) if selected else _mcd_export_title(source_file)


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def export_mcd_analysis_bundle(
    result: McdResult,
    output_dir: str,
    *,
    trace_map: str,
    center_ev: float,
    width_mev: float,
    metric: WindowMetric,
    settings: McdSettings | None = None,
    show_raw: bool = False,
    show_signed_mean: bool = True,
    show_field_signed_absolute_mean: bool = False,
    show_unsigned_absolute_mean: bool = False,
    show_integral: bool = False,
    fit_near_zero: bool = False,
    fit_window_t: float = 0.2,
) -> dict[str, Path]:
    """Export the compact, publication-facing MCD analysis set.

    Intermediate maps are intentionally not written here.  They remain
    available in memory for inspection, while the output folder receives only
    the MCD(B) figure/table, pairing diagnostics, and reproducible settings.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source_file).stem
    tag = _mcd_window_export_tag(center_ev, width_mev)
    trace_base = f"{stem}_MCD_vs_B_{tag}"
    table = _mcd_trace_comparison_table(result, trace_map, center_ev, width_mev)
    csv_path = out / f"{trace_base}.csv"
    table.to_csv(csv_path, index=False)

    figure_path = out / f"{trace_base}.png"
    # Match the fixed PL/DRR export canvas.  The plot rectangle is likewise
    # fixed whether or not the integral/right axis is enabled, so PNGs from
    # different trace selections align exactly in size and inner plot area.
    fig = Figure(figsize=(8.0, 6.2), dpi=150, facecolor="white")
    FigureCanvasAgg(fig)
    fig.text(0.08, 0.985, _mcd_export_metadata(result.source_file), ha="left", va="top", fontsize=8.2, fontweight="bold", color="#242424")
    fig.text(
        0.08, 0.945,
        f"MCD(B): E = {float(center_ev):.6f} eV, window = {float(width_mev):.6g} meV",
        ha="left", va="top", fontsize=16, fontweight="bold",
    )
    # The 16 pt MCD y-label needs more room than a heatmap's short axis
    # label.  This fixed rectangle is shared by every MCD(B) PNG.
    axis = fig.add_axes([0.16, 0.12, 0.70, 0.76])

    traces = pair_window_trace_by_branch(result, center_ev, width_mev)
    trace_specs = (
        ("mean", "Signed mean", "#1666b0", show_signed_mean),
        ("field_signed_absolute_mean", "Field-signed |MCD|", "#c94c00", show_field_signed_absolute_mean),
        ("absolute_mean", "Unsigned |MCD|", "#777777", show_unsigned_absolute_mean),
        ("integral", "Signed integral", "#6a3d9a", show_integral),
    )
    integral_axis = axis.twinx()
    for metric_name, label, color, visible in trace_specs:
        if not visible:
            continue
        target = integral_axis if metric_name == "integral" else axis
        for branch, branch_style, marker_fill in (("B increasing", "-", color), ("B decreasing", "--", "white")):
            for source, alpha in (("corrected", 1.0), ("raw", 0.70)):
                if source == "raw" and not show_raw:
                    continue
                b_values, values = traces[branch][f"{source}_{metric_name}"]
                target.plot(
                    b_values, values, f"o{branch_style}", ms=3.1, lw=1.25,
                    color=color, alpha=alpha, markerfacecolor=marker_fill,
                    markeredgecolor=color, markeredgewidth=0.9, label="_nolegend_",
                )
        if fit_near_zero and metric_name == "mean":
            for branch, branch_style in (("B increasing", "-"), ("B decreasing", "--")):
                b_values, values = traces[branch]["corrected_mean"]
                mask = np.isfinite(b_values) & np.isfinite(values) & (np.abs(b_values) <= float(fit_window_t))
                if np.count_nonzero(mask) >= 2:
                    slope, intercept = np.polyfit(b_values[mask], values[mask], 1)
                    axis.plot(b_values, slope * b_values + intercept, ":", color=color, lw=1.1, label="_nolegend_")

    axis.axhline(0.0, color="#555", lw=0.7)
    axis.set_xlabel("B field (T)", fontsize=16)
    axis.set_ylabel("MCD (mean / absolute mean)", fontsize=16)
    axis.tick_params(labelsize=14)
    axis.grid(alpha=0.25)
    if show_integral:
        integral_axis.set_ylabel("Integrated MCD (eV)", fontsize=16, labelpad=8)
        integral_axis.tick_params(labelsize=14)
    else:
        # Keep the same right-side gutter and inner plot dimensions as an
        # integral-enabled export, but hide the unused right axis itself.
        integral_axis.set_yticks([])
        integral_axis.set_ylabel("")
        integral_axis.spines["right"].set_visible(False)
    branch_legend = axis.legend(
        [
            Line2D([0], [0], color="#333", marker="o", markerfacecolor="#333", lw=1.25),
            Line2D([0], [0], color="#333", marker="o", markerfacecolor="white", lw=1.25, ls="--"),
        ],
        ["B increasing", "B decreasing"],
        fontsize=7.2, frameon=True, framealpha=0.82, loc="upper left", borderpad=0.35, labelspacing=0.28, handlelength=2.4,
    )
    axis.add_artist(branch_legend)
    metric_handles: list[Line2D] = []
    metric_labels: list[str] = []
    for metric_name, label, color, visible in trace_specs:
        if not visible:
            continue
        metric_handles.append(Line2D([0], [0], color=color, lw=1.6))
        metric_labels.append(f"{label} (right axis)" if metric_name == "integral" else label)
    if metric_handles:
        axis.legend(metric_handles, metric_labels, fontsize=7.2, frameon=True, framealpha=0.82, loc="lower right", borderpad=0.35, labelspacing=0.28, handlelength=2.4)
    fig.savefig(figure_path, dpi=fig.dpi, facecolor="white", edgecolor="none", pad_inches=0)

    diagnostic_path = out / f"{stem}_MCD_pair_diagnostics.csv"
    _mcd_pair_diagnostic_table(result).to_csv(diagnostic_path, index=False)
    settings_path = out / f"{stem}_MCD_settings.json"
    setting_values = asdict(settings) if settings is not None else {}
    for key in ("dark_pos_file", "dark_neg_file"):
        if setting_values.get(key):
            setting_values[key] = Path(str(setting_values[key])).name
    payload = {
        "source_file": Path(result.source_file).name,
        "map": trace_map,
        "mcd_b": {
            "center_ev": float(center_ev),
            "width_mev": float(width_mev),
            "primary_metric": metric,
            "show_raw": bool(show_raw),
            "show_signed_mean": bool(show_signed_mean),
            "show_field_signed_absolute_mean": bool(show_field_signed_absolute_mean),
            "show_unsigned_absolute_mean": bool(show_unsigned_absolute_mean),
            "show_integral": bool(show_integral),
            "fit_near_zero": bool(fit_near_zero),
            "fit_window_t": float(fit_window_t),
        },
        "sigma_plus_angle_deg": float(result.pos_angle),
        "sigma_minus_angle_deg": float(result.neg_angle),
        "reference_b_t": float(result.reference_b),
        "processing": _json_safe(setting_values),
        "processing_summary": _json_safe(result.summary),
    }
    settings_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "mcd_vs_b_png": figure_path,
        "mcd_vs_b_csv": csv_path,
        "pair_diagnostics": diagnostic_path,
        "settings": settings_path,
    }
