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


def detect_peak_seeds(cube, max_peaks=4) -> tuple[PeakSeed, ...]:
    x = np.asarray(cube.energy)
    if len(x) < 8:
        raise ValueError("At least eight spectral samples are needed to detect peaks.")
    z = np.asarray(cube.Z)
    count = np.isfinite(z).sum(axis=0)
    y = np.divide(np.where(np.isfinite(z), z, 0).sum(axis=0), count,
                  out=np.zeros(z.shape[1]), where=count > 0)
    y = savgol_filter(y, min(21, len(y) // 2 * 2 - 1), 2)
    y -= np.linspace(np.median(y[:5]), np.median(y[-5:]), len(y))
    indexes, properties = find_peaks(y, prominence=max(float(np.ptp(y)) * .045, 1e-12),
                                      distance=max(2, len(y) // 25))
    if not len(indexes):
        indexes = np.asarray([int(np.clip(np.argmax(y), 1, len(y) - 2))])
        widths = np.asarray([max(2., len(y) / 10)])
    else:
        chosen = np.argsort(properties["prominences"])[-max_peaks:]
        indexes = np.sort(indexes[chosen])
        widths = peak_widths(y, indexes, rel_height=.5)[0]
    dx = float(np.median(np.diff(x)))
    return tuple(PeakSeed(float(x[i]), max(float(width) * dx, dx * 2)) for i, width in zip(indexes, widths))


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
