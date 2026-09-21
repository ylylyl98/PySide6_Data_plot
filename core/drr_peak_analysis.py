"""Range-limited DRR extrema with conservative, adjacent-row tracking.

Energies are in eV; distance controls are in meV. Prominence is a fraction of
each selected row's finite peak-to-peak range, separately for each product.
Missing samples split detection segments and missing rows terminate tracks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np
from scipy.signal import find_peaks

from core.loader import DataCube
from core.processing import apply_sg_derivative_energy


@dataclass(frozen=True)
class PeakAnalysisSettings:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    source: str = "both"
    polarity: str = "both"
    prominence: float = 0.05
    min_distance_mev: float = 1.0
    max_peaks: int = 6
    max_shift_mev: float = 3.0
    sg_window: int = 21
    sg_polyorder: int = 2
    mode: str = "all"
    seed_energy: float = 0.0
    seed_y: float = 0.0
    noise_sigma: float = 4.0
    min_width_mev: float = 1.0
    max_gap: int = 2


def _validate(cube: DataCube, settings: PeakAnalysisSettings):
    x, y, z = (np.asarray(a, dtype=float) for a in (cube.energy, cube.gate, cube.Z))
    if x.ndim != 1 or y.ndim != 1 or z.shape != (y.size, x.size):
        raise ValueError("Expected one-dimensional X/Y axes and a matching Y-by-X data matrix.")
    if x.size < 3 or y.size == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Axes must be finite, with at least three X points and one Y row.")
    if np.unique(y).size != y.size:
        raise ValueError("Repeated Y values: repeated sweep rows must be separated or aggregated before batch peak analysis.")
    if not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError("Energy must be strictly increasing or decreasing.")
    values = asdict(settings)
    for name in ("x_min", "x_max", "y_min", "y_max", "prominence", "min_distance_mev", "max_shift_mev"):
        try:
            values[name] = float(values[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number.") from exc
        if not np.isfinite(values[name]):
            raise ValueError(f"{name} must be a finite number.")
    for name in ("max_peaks", "sg_window", "sg_polyorder"):
        value = values[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")
        values[name] = int(value)
    if values["x_min"] >= values["x_max"] or values["y_min"] > values["y_max"]:
        raise ValueError("X range must increase and Y range must be ordered.")
    if settings.source not in ("raw", "second", "both") or settings.polarity not in ("peaks", "dips", "both"):
        raise ValueError("Invalid source or polarity.")
    if settings.source in ("second", "both") and x.size < 5:
        raise ValueError("Second derivative needs at least five energy samples.")
    if not 0 <= values["prominence"] <= 1:
        raise ValueError("Prominence must be a fraction between zero and one.")
    if values["min_distance_mev"] < 0 or values["max_shift_mev"] < 0 or values["max_peaks"] < 1:
        raise ValueError("Distances must be nonnegative and max_peaks must be positive.")
    if settings.source in ("second", "both") and (
            values["sg_polyorder"] < 2 or values["sg_window"] <= values["sg_polyorder"]):
        raise ValueError("SG polynomial order must be at least two and smaller than its window.")
    normalized = PeakAnalysisSettings(**values)
    if settings.mode not in ('all', 'seed'):
        raise ValueError('Invalid analysis mode.')
    if settings.mode == 'seed':
        if settings.source == 'both' or settings.polarity == 'both':
            raise ValueError('Target branch requires one source and one polarity.')
        for name in ('seed_energy', 'seed_y', 'noise_sigma', 'min_width_mev'):
            if not np.isfinite(float(getattr(settings, name))):
                raise ValueError(f'{name} must be finite.')
        if not settings.x_min <= settings.seed_energy <= settings.x_max or not settings.y_min <= settings.seed_y <= settings.y_max:
            raise ValueError('The seed must lie inside the analysis range.')
        if settings.noise_sigma <= 0 or settings.min_width_mev <= 0 or settings.max_shift_mev <= 0:
            raise ValueError('Target noise, width and displacement limits must be positive.')
        if isinstance(settings.max_gap, bool) or not isinstance(settings.max_gap, int) or not 0 <= settings.max_gap <= 10:
            raise ValueError('max_gap must be an integer between zero and ten.')
    # The regridding SG helper requires increasing energy (CubicSpline).
    # Normalize local views only; original row indices remain untouched.
    if x[0] > x[-1]:
        x, z = x[::-1], z[:, ::-1]
    ix = np.flatnonzero((x >= normalized.x_min) & (x <= normalized.x_max))
    iy = np.flatnonzero((y >= normalized.y_min) & (y <= normalized.y_max))
    if ix.size < 3 or not iy.size:
        raise ValueError("Selected range needs at least three X samples and one Y row.")
    return x, y, z, ix, iy, normalized


def _detect(x: np.ndarray, row: np.ndarray, settings: PeakAnalysisSettings, retain_candidates=False) -> list[dict]:
    finite = np.isfinite(row)
    if np.count_nonzero(finite) < 3:
        return []
    span = float(np.ptp(row[finite]))
    if not np.isfinite(span) or span <= 0:
        return []
    threshold = 0 if retain_candidates else settings.prominence * span
    edges = np.diff(np.r_[False, finite, False].astype(int))
    candidates = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        for sign, polarity in ((1, "peak"), (-1, "dip")):
            if settings.polarity != "both" and settings.polarity != polarity + "s":
                continue
            indices, properties = find_peaks(sign * row[start:end], prominence=threshold,width=(None,None))
            for local,index in enumerate(indices + start):
                prominence=properties['prominences'][local]
                left=np.interp(properties['left_ips'][local],np.arange(end-start),x[start:end])
                right=np.interp(properties['right_ips'][local],np.arange(end-start),x[start:end])
                candidates.append({"energy": float(x[index]), "amplitude": float(row[index]),
                                   "prominence": float(prominence), "prominence_fraction":float(prominence/span),
                                   "width_mev":float(abs(right-left)*1000), "polarity": polarity})
    if retain_candidates:return sorted(candidates,key=lambda p:(p['energy'],p['polarity']))
    # Enforce distance in energy, not index count, including uneven grids.
    chosen = []
    for point in sorted(candidates, key=lambda p: (-p["prominence"], p["energy"], p["polarity"])):
        if all(p["polarity"] != point["polarity"] or
               abs(p["energy"] - point["energy"]) >= settings.min_distance_mev / 1000
               for p in chosen):
            chosen.append(point)
            if len(chosen) >= settings.max_peaks:
                break
    return sorted(chosen, key=lambda p: (p["energy"], p["polarity"]))


def _link(previous: list[dict], current: list[dict], next_id: int, max_shift: float) -> int:
    """Connect only isolated one-to-one matches; never guess at split/merge events."""
    edges = [(i, j) for i, old in enumerate(previous) for j, new in enumerate(current)
             if old["status"] == "accepted" and old["polarity"] == new["polarity"]
             and abs(old["energy"] - new["energy"]) <= max_shift + 1e-12]
    old_neighbors = {i: [] for i in range(len(previous))}
    new_neighbors = {j: [] for j in range(len(current))}
    for i,j in edges:old_neighbors[i].append(j);new_neighbors[j].append(i)
    # Mark every point participating in any ambiguous connected component.
    ambiguous_old = {i for i, js in old_neighbors.items() if len(js) > 1}
    ambiguous_new = {j for j, inds in new_neighbors.items() if len(inds) > 1}
    while True:
        old_count, new_count = len(ambiguous_old), len(ambiguous_new)
        ambiguous_new.update(j for i, j in edges if i in ambiguous_old)
        ambiguous_old.update(i for i, j in edges if j in ambiguous_new)
        if (old_count, new_count) == (len(ambiguous_old), len(ambiguous_new)):
            break
    for i in ambiguous_old:
        previous[i]["status"] = "uncertain"
        # A point whose identity became ambiguous must not remain attached to
        # its incoming track either; isolated markers keep exports honest.
        previous[i]["track_id"] = next_id
        next_id += 1
    for j, point in enumerate(current):
        matches = new_neighbors[j]
        point["status"] = "uncertain" if j in ambiguous_new else "accepted"
        if matches and j not in ambiguous_new:
            point["track_id"] = previous[matches[0]]["track_id"]
        else:
            point["track_id"] = next_id
            next_id += 1
    return next_id


def compute_second_derivative_row(cube: DataCube, row_index: int,
                                  settings: PeakAnalysisSettings) -> tuple[np.ndarray, int | None]:
    """Shared full-axis derivative policy for analysis and comparison exports.

    Return the derivative in the input energy order and the effective SG window.
    Constant or undersampled rows return NaNs and no effective window, avoiding
    spurious peaks caused by differentiating roundoff on constant spectra.
    """
    x = np.asarray(cube.energy, float)
    original = np.asarray(cube.Z[row_index], float)
    finite = np.isfinite(original)
    if np.count_nonzero(finite) < 5 or np.ptp(original[finite]) == 0:
        return np.full(original.shape, np.nan), None
    descending = x[0] > x[-1]
    row = original[::-1] if descending else original
    full_row = replace(cube, energy=x[::-1] if descending else x,
                       gate=np.asarray(cube.gate[row_index:row_index+1], float),
                       Z=row[np.newaxis, :])
    derivative, used_window = apply_sg_derivative_energy(
        full_row, derivative=2, window_length=settings.sg_window, polyorder=settings.sg_polyorder)
    result = np.asarray(derivative.Z[0], float).copy()
    if descending:
        result = result[::-1].copy()
    result[~finite] = np.nan
    return result, int(used_window)


def analyze_drr_peaks(cube: DataCube, settings: PeakAnalysisSettings,
                      cancelled: Callable[[], bool] | None = None,
                      progress: Callable[[int], None] | None = None, *, retain_candidates=False, auto_candidates=False) -> dict:
    """Return JSON-safe points without modifying the source cube.

    Cancellation raises RuntimeError and never returns a partial result. SG is
    applied to each selected row's complete energy axis before X selection.
    The existing SG implementation regrids nonuniform spectra; original missing
    samples are restored as gaps so detection cannot span missing measurements.
    """
    def check_cancelled():
        if cancelled is not None and cancelled():
            raise RuntimeError("Peak analysis cancelled.")

    check_cancelled()
    if progress is not None: progress(0)
    x, y, z, ix, iy, settings = _validate(cube, settings)
    if auto_candidates:retain_candidates=True
    normalized_cube = replace(cube, energy=x, gate=y, Z=z)
    sources = ("raw", "second") if settings.source == "both" else (settings.source,)
    if settings.mode == 'seed':
        from core.drr_seed_tracking import track_seed
        return track_seed(normalized_cube, settings, ix, iy, check_cancelled, progress)
    products = {source: {"points": []} for source in sources}
    completed = 0
    for source in sources:
        previous = []
        previous_row = None
        next_id = 1
        for row_index in iy:
            check_cancelled()
            row = z[row_index]
            if source == "second":
                row, used_window = compute_second_derivative_row(normalized_cube, int(row_index), settings)
                if used_window is not None:
                    products[source]["sg_window_used"] = used_window
            check_cancelled()
            points = _detect(x[ix], row[ix], settings,retain_candidates)
            if auto_candidates:
                from core.drr_auto_candidates import assess_row,derivative_noise_floor
                noise_floor=(lambda start,end: derivative_noise_floor(x,z[row_index][ix[start:end]],used_window,settings.sg_polyorder)) if source=='second' else 0.
                assess_row(x[ix],row[ix],points,settings,noise_floor)
            for point in points:
                point.update(row_index=int(row_index), y=float(y[row_index]))
            if previous_row is None or row_index != previous_row + 1:
                previous = []
            if retain_candidates:
                for point in points:
                    point.update(status='accepted',track_id=next_id);next_id+=1
            else:next_id = _link(previous, points, next_id, settings.max_shift_mev / 1000)
            products[source]["points"].extend(points)
            previous, previous_row = points, int(row_index)
            completed += 1
            if progress is not None: progress(min(99, int(100 * completed / (len(sources) * len(iy)))))
        products[source]["processing"] = ("Raw DRR; no smoothing" if source == "raw" else
                                            "Full-energy-axis Savitzky-Golay second derivative before range crop")
        if auto_candidates:
            from core.drr_auto_candidates import assess_neighbors
            assess_neighbors(products[source]['points'],check_cancelled)
    check_cancelled()
    if progress is not None: progress(100)
    return {"schema_version": 1, "candidate_pool":bool(retain_candidates), 'auto_candidates':bool(auto_candidates), "settings": asdict(settings), "y_label": str(cube.gate_label),
            'metadata':{'width_definition':'Full width at half prominence, interpolated on the energy axis (meV)',
                        'auto_policy':{'version':1,'windows_samples':[5,9,15],'noise_estimator':'MAD of local smoothing residual / 0.67448975, separately per finite segment and product',
                                       'derivative_noise_floor':'Per finite segment: raw first-difference MAD / (0.67448975 * sqrt(2)), propagated by the L2 norm of the SG second-derivative coefficients; assumes independent sample noise',
                                       'strong_ratio':5,'weak_ratio':2.5,'required_scales':2,'weak_requires_both_adjacent_rows':True,'quality_is_probability':False} if auto_candidates else None,
                        'candidate_policy':'All local extrema in selected range/polarity; prominence, spacing and count applied after detection' if retain_candidates else 'Detection thresholds applied'},
            "y_values": [float(y[i]) for i in iy], "products": products}
