"""Simultaneous full-spectrum peak decomposition with stable peak identities."""

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import find_peaks, peak_widths, savgol_filter

from core.power_peaks import PeakFit, peak_model


@dataclass(frozen=True)
class PeakSeed:
    center_ev: float
    fwhm_ev: float = .01
    height: float | None = None
    center_min_ev: float | None = None
    center_max_ev: float | None = None


@dataclass(frozen=True)
class MultiPeakSettings:
    peaks: tuple[PeakSeed, ...]
    model: str = "Lorentzian"


@dataclass
class SpectrumFit:
    power_uw: float
    status: str
    components: tuple[PeakFit, ...]
    settings: MultiPeakSettings
    manual: bool = False
    rms: float = np.nan
    parameters: tuple[float, ...] = ()
    x: np.ndarray | None = field(default=None, repr=False)
    fitted: np.ndarray | None = field(default=None, repr=False)
    residual: np.ndarray | None = field(default=None, repr=False)
    baseline: np.ndarray | None = field(default=None, repr=False)
    component_curves: tuple[np.ndarray, ...] = field(default=(), repr=False)


def detect_peak_seeds(cube, max_peaks=8) -> tuple[PeakSeed, ...]:
    """Suggest persistent local maxima without letting bright rows mask weak ones.

    Each usable spectrum is normalized separately, then candidates are pooled
    across power. Detection runs on a uniform energy grid; fitting still uses
    the original measured samples. Suggestions require review by the user.
    """
    x = np.asarray(cube.energy, dtype=float)
    z = np.asarray(cube.Z, dtype=float)
    if x.ndim != 1 or len(x) < 8 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("Peak detection needs at least eight finite, increasing energy samples.")
    if z.ndim != 2 or z.shape[1] != len(x):
        raise ValueError("Power rows and spectra do not match the energy axis.")
    if not isinstance(max_peaks, (int, np.integer)) or not 1 <= max_peaks <= 8:
        raise ValueError("Choose between one and eight peaks.")
    grid = np.linspace(x[0], x[-1], len(x))
    dx = float(grid[1] - grid[0])
    candidates = []
    usable = 0
    # Bound runtime while sampling the entire power sweep.
    rows = np.unique(np.linspace(0, max(len(z) - 1, 0), min(len(z), 32), dtype=int))
    for row_index in rows:
        row = z[row_index]
        finite = np.isfinite(row)
        if finite.sum() < 8:
            continue
        y = np.interp(grid, x[finite], row[finite])
        smooth = savgol_filter(y, min(9, len(y) // 2 * 2 - 1), 2)
        edge = max(2, len(y) // 40)
        baseline = np.linspace(np.median(smooth[:edge]), np.median(smooth[-edge:]), len(y))
        signal = smooth - baseline
        scale = float(np.ptp(signal))
        if scale <= 100 * np.finfo(float).eps * max(1., float(np.max(np.abs(y)))):
            continue
        # The high-frequency residual estimates noise in the original signal.
        residual = y - smooth
        noise = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        threshold = max(.045 * scale, 5. * noise)
        indices, props = find_peaks(signal, prominence=threshold, distance=3)
        usable += 1
        if len(indices):
            widths = peak_widths(signal, indices, rel_height=.5)[0]
            for index, prominence, width in zip(indices, props["prominences"], widths):
                candidates.append((float(grid[index]), float(prominence / scale),
                                   max(float(width) * dx, 2 * dx), int(row_index)))
    if not usable or not candidates:
        raise ValueError("No reliable peaks detected; add a peak manually or inspect the spectra.")
    # Group nearby observations while preserving distinct resolved maxima.
    groups = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        if (groups and candidate[3] not in {item[3] for item in groups[-1]}
                and candidate[0] - np.median([item[0] for item in groups[-1]])
                <= max(3 * dx, min(.2 * np.median([item[2] for item in groups[-1]]), .003))):
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    required = max(1, int(np.ceil(usable * .35)))
    ranked = []
    for group in groups:
        support = len({item[3] for item in group})
        if support >= required or (support >= 2 and max(item[1] for item in group) >= .12):
            ranked.append((support, max(item[1] for item in group), group))
    if not ranked:
        raise ValueError("No consistent peaks detected; add a peak manually or inspect the spectra.")
    chosen = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)[:max_peaks]
    seeds = [PeakSeed(float(np.median([item[0] for item in group])),
                      float(np.median([item[2] for item in group])))
             for _, _, group in chosen]
    return tuple(sorted(seeds, key=lambda seed: seed.center_ev))


def detect_power_peak_seeds(cubes, max_peaks=8) -> tuple[PeakSeed, ...]:
    """Detect on all channels, resampling to their common measured energy range."""
    from types import SimpleNamespace

    channels = list(cubes.values()) if hasattr(cubes, "values") else list(cubes)
    if not channels:
        raise ValueError("Load at least one Power intensity channel.")
    axes = [np.asarray(cube.energy, dtype=float) for cube in channels]
    for axis, cube in zip(axes, channels):
        if (axis.ndim != 1 or len(axis) < 8 or np.any(~np.isfinite(axis))
                or np.any(np.diff(axis) <= 0)):
            raise ValueError("Peak detection needs at least eight finite, increasing energy samples.")
        if np.asarray(cube.Z).ndim != 2 or np.asarray(cube.Z).shape[1] != len(axis):
            raise ValueError("Power rows and spectra do not match the energy axis.")
    lo, hi = max(axis[0] for axis in axes), min(axis[-1] for axis in axes)
    if lo >= hi:
        raise ValueError("Power channels have no common energy range.")
    # Detect separately so a short channel is not overwhelmed by a long one.
    if len(channels) == 1:
        return detect_peak_seeds(channels[0], max_peaks=max_peaks)
    suggestions = []
    for channel_index, (axis, cube) in enumerate(zip(axes, channels)):
        mask = (axis >= lo) & (axis <= hi)
        if np.count_nonzero(mask) < 8:
            raise ValueError("Power channels need at least eight shared energy samples.")
        try:
            seeds = detect_peak_seeds(SimpleNamespace(energy=axis[mask],
                Z=np.asarray(cube.Z, dtype=float)[:, mask]), max_peaks=max_peaks)
        except ValueError as exc:
            if not str(exc).startswith(("No reliable peaks", "No consistent peaks")):
                raise
            continue
        suggestions.extend((channel_index, seed) for seed in seeds)
    if not suggestions:
        raise ValueError("No reliable peaks detected; add a peak manually or inspect the spectra.")
    # Coalesce the same feature across channels; a tighter tolerance than the
    # estimated width keeps nearby resolved peaks separate.
    dx = max(float(np.median(np.diff(axis))) for axis in axes)
    groups = []
    for channel, seed in sorted(suggestions, key=lambda item: item[1].center_ev):
        matches = []
        for group in groups:
            if channel in {item[0] for item in group}:
                continue
            center = float(np.median([item[1].center_ev for item in group]))
            width = float(np.median([item[1].fwhm_ev for item in group]))
            distance = abs(seed.center_ev - center)
            if distance <= max(3 * dx, min(.2 * width, .003)):
                matches.append((distance, group))
        if matches:
            min(matches, key=lambda item: item[0])[1].append((channel, seed))
        else:
            groups.append([(channel, seed)])
    # Prefer features seen in more channels when the requested limit is reached.
    groups = sorted(groups, key=len, reverse=True)[:max_peaks]
    return tuple(sorted((PeakSeed(float(np.median([item[1].center_ev for item in group])),
                                 float(np.median([item[1].fwhm_ev for item in group])))
                         for group in groups), key=lambda item: item.center_ev))



def center_bounds(settings, energy):
    centers = np.asarray([seed.center_ev for seed in settings.peaks])
    if not len(centers) or len(centers) > 8 or np.any(~np.isfinite(centers)) or np.any(np.diff(centers) <= 0):
        raise ValueError("Use 1–8 peaks with centers in increasing energy order (Peak 1 is lowest energy).")
    edges = np.r_[energy[0], .5 * (centers[1:] + centers[:-1]), energy[-1]]
    bounds = []
    for index, seed in enumerate(settings.peaks):
        lo = edges[index] if seed.center_min_ev is None else max(energy[0], seed.center_min_ev)
        hi = edges[index + 1] if seed.center_max_ev is None else min(energy[-1], seed.center_max_ev)
        if not np.isfinite(lo) or not np.isfinite(hi) or not lo < seed.center_ev < hi:
            raise ValueError(f"Peak {index + 1}: its center must lie inside its center bounds and the measured spectrum.")
        if bounds and lo < bounds[-1][1]:
            raise ValueError("Peak center bounds must not overlap, to keep peak labels consistent across power.")
        if not np.isfinite(seed.fwhm_ev) or seed.fwhm_ev <= 0:
            raise ValueError("Peak width guesses must be positive.")
        if seed.height is not None and (not np.isfinite(seed.height) or seed.height <= 0):
            raise ValueError("Peak height guesses must be positive, or automatic.")
        bounds.append((float(lo), float(hi)))
    return bounds


def multi_peak_model(x, parameters, *, model, origin):
    out = parameters[0] + parameters[1] * (x - origin)
    for offset in range(2, len(parameters), 3):
        height, center, width = parameters[offset:offset + 3]
        out = out + peak_model(x, 0., 0., height, center, width, model=model, origin=origin)
    return out


def multi_peak_jacobian(x, parameters, *, model, origin):
    columns = [np.ones_like(x), x - origin]
    for offset in range(2, len(parameters), 3):
        height, center, width = parameters[offset:offset + 3]
        dx = x - center
        if model == "Lorentzian":
            t = 2 * dx / width
            shape = 1 / (1 + t * t)
            columns.extend([shape, height * 4 * t / width * shape ** 2,
                            height * 2 * t * t / width * shape ** 2])
        else:
            k = 4 * np.log(2.)
            shape = np.exp(-k * (dx / width) ** 2)
            columns.extend([shape, height * shape * 2 * k * dx / width ** 2,
                            height * shape * 2 * k * dx * dx / width ** 3])
    return np.column_stack(columns)


def fit_multi_spectrum(energy, spectrum, power, settings, *, manual=False) -> SpectrumFit:
    if settings.model not in ("Lorentzian", "Gaussian"):
        raise ValueError("Choose Lorentzian or Gaussian.")
    energy, spectrum = np.asarray(energy, float), np.asarray(spectrum, float)
    if energy.shape != spectrum.shape or energy.ndim != 1 or np.any(~np.isfinite(energy)) or np.any(np.diff(energy) <= 0):
        raise ValueError("Spectrum needs a matching, increasing, finite energy axis.")
    bounds = center_bounds(settings, energy)
    components = tuple(PeakFit(float(power), "insufficient finite points") for _ in settings.peaks)
    result = SpectrumFit(float(power), "insufficient finite points", components, settings, manual)
    valid = np.isfinite(spectrum)
    x, y = energy[valid], spectrum[valid]
    if len(x) < max(8, 3 * len(settings.peaks) + 6):
        return result
    scale = float(np.ptp(y))
    if scale <= np.finfo(float).eps * max(1., np.max(np.abs(y))):
        result.status = "flat spectrum"
        for component in components:
            component.status = result.status
        return result
    origin, span = float(np.mean(x)), float(x[-1] - x[0])
    dx = float(np.median(np.diff(x)))
    edge = max(2, len(x) // 20)
    slope = (np.median(y[-edge:]) - np.median(y[:edge])) / span
    baseline = float(np.median(y[:edge]) + slope * (origin - np.mean(x[:edge])))
    reduced = y - baseline - slope * (x - origin)
    p0 = [baseline / scale, slope / scale]
    lower, upper = [-np.inf, -np.inf], [np.inf, np.inf]
    for seed, (lo, hi) in zip(settings.peaks, bounds):
        height = seed.height if seed.height is not None else max(float(np.interp(seed.center_ev, x, reduced)), scale * .02)
        p0.extend([height / scale, seed.center_ev, np.clip(seed.fwhm_ev, dx * .51, span * .99)])
        lower.extend([0., lo, dx * .5])
        upper.extend([np.inf, hi, span])
    def model(xx, *parameters):
        return multi_peak_model(xx, parameters, model=settings.model, origin=origin)
    def jacobian(xx, *parameters):
        return multi_peak_jacobian(xx, parameters, model=settings.model, origin=origin)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            parameters, covariance = curve_fit(model, x, y / scale, p0=p0, bounds=(lower, upper),
                                                jac=jacobian, maxfev=3000, x_scale="jac")
        result.x = x
        result.fitted = model(x, *parameters) * scale
        result.residual = y - result.fitted
        result.rms = float(np.sqrt(np.mean(result.residual ** 2)))
        result.baseline = (parameters[0] + parameters[1] * (x - origin)) * scale
        actual = parameters.copy()
        actual[:2] *= scale
        actual[2::3] *= scale
        result.parameters = tuple(float(value) for value in actual)
        curves = []
        for index, component in enumerate(components):
            offset = 2 + 3 * index
            height, center, width = actual[offset:offset + 3]
            error = float(np.sqrt(covariance[offset + 2, offset + 2]))
            curve = peak_model(x, 0., 0., height, center, width, model=settings.model, origin=origin)
            curves.append(curve)
            component.center_ev, component.rms = float(center), result.rms
            lo, hi = bounds[index]
            if width <= dx or width >= span * .95:
                component.status = "width unresolved or reaches spectrum limit"
            elif min(center - lo, hi - center) < dx * .25:
                component.status = "center reaches tracking limit"
            elif not np.isfinite(error) or error > width:
                component.status = "width uncertainty too large"
            elif height <= 3 * result.rms:
                component.status = "peak below residual noise"
            else:
                component.status = "ok"
                component.fwhm_mev = float(width * 1000)
                component.fwhm_error_mev = error * 1000
                component.height = float(height)
                component.area = float(np.trapezoid(curve, x))
        result.component_curves = tuple(curves)
        count = sum(component.status == "ok" for component in components)
        result.status = "ok" if count == len(components) else f"{count}/{len(components)} peaks valid"
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        result.status = f"fit failed: {exc}"
        for component in components:
            component.status = result.status
    return result


def fit_power_multi_peaks(cube, settings, *, cancelled=None):
    if np.asarray(cube.Z).shape != (len(cube.gate), len(cube.energy)):
        raise ValueError("Power rows and spectra do not match their axes.")
    results = []
    for power, spectrum in zip(cube.gate, cube.Z):
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        results.append(fit_multi_spectrum(cube.energy, spectrum, power, settings))
    return tuple(results)
