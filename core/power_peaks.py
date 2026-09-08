"""Single-peak analysis of power sweeps, in photon energy coordinates."""

from dataclasses import dataclass, asdict, field
import csv
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from core.loader import DataCube
from core.plotting import plain_log_ticks


@dataclass(frozen=True)
class PeakSettings:
    center_ev: float
    half_window_ev: float
    model: str = "Lorentzian"


@dataclass
class PeakFit:
    power_uw: float
    status: str
    center_ev: float = np.nan
    fwhm_mev: float = np.nan
    fwhm_error_mev: float = np.nan
    area: float = np.nan
    height: float = np.nan
    rms: float = np.nan
    x: np.ndarray | None = field(default=None, repr=False)
    fitted: np.ndarray | None = field(default=None, repr=False)
    residual: np.ndarray | None = field(default=None, repr=False)


def peak_model(x, baseline, slope, height, center, fwhm, *, model, origin):
    dx = np.asarray(x) - center
    if model == "Lorentzian":
        peak = height / (1.0 + (2.0 * dx / fwhm) ** 2)
    elif model == "Gaussian":
        peak = height * np.exp(-4.0 * np.log(2.0) * (dx / fwhm) ** 2)
    else:
        raise ValueError("Choose Lorentzian or Gaussian.")
    return baseline + slope * (np.asarray(x) - origin) + peak


def fit_power_peaks(cube: DataCube, settings: PeakSettings, *, cancelled=None) -> tuple[PeakFit, ...]:
    """Fit each measured row independently in a fixed window about one peak.

    Area is measured intensity minus the fitted linear baseline, integrated over
    the finite sampled points in the requested window (a.u. eV). Height is the
    fitted peak amplitude. Unreliable fits retain diagnostics but not trend values.
    """
    center, half = settings.center_ev, settings.half_window_ev
    if not np.isfinite(center) or not np.isfinite(half) or half <= 0:
        raise ValueError("Choose a finite peak center and positive fitting half-window.")
    if settings.model not in {"Lorentzian", "Gaussian"}:
        raise ValueError("Choose Lorentzian or Gaussian.")
    energy = np.asarray(cube.energy, float)
    if np.asarray(cube.Z).shape != (len(cube.gate), len(energy)):
        raise ValueError("Power rows and spectra must match the energy and power axes.")
    mask = np.isfinite(energy) & (energy >= center - half) & (energy <= center + half)
    results = []
    for power, spectrum in zip(cube.gate, cube.Z):
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        fit = PeakFit(float(power), "insufficient points (need at least 8)")
        results.append(fit)
        valid = mask & np.isfinite(spectrum)
        x, y = energy[valid], np.asarray(spectrum, float)[valid]
        if x.size < 8:
            continue
        order = np.argsort(x)
        x, y = x[order], y[order]
        if np.any(np.diff(x) <= 0):
            fit.status = "duplicate energy points"
            continue
        span = x[-1] - x[0]
        dx = float(np.median(np.diff(x)))
        if x[0] >= center or x[-1] <= center:
            fit.status = "peak center outside measured window"
            continue
        scale = float(np.ptp(y))
        if scale <= np.finfo(float).eps * max(1.0, float(np.max(np.abs(y)))):
            fit.status = "flat spectrum"
            continue
        edge = max(2, x.size // 10)
        slope = (np.median(y[-edge:]) - np.median(y[:edge])) / span
        baseline = float(np.median(y[:edge]) + slope * (center - np.mean(x[:edge])))
        reduced = y - baseline - slope * (x - center)
        # Scaling makes fitting weak and strong PL signals equally well conditioned.
        yn = y / scale
        lower_center = max(x[0], center - half * 0.5)
        upper_center = min(x[-1], center + half * 0.5)
        candidates = (x >= lower_center) & (x <= upper_center)
        if not np.any(candidates):
            fit.status = "no samples near selected peak center"
            continue
        guess_center = float(x[candidates][np.argmax(reduced[candidates])])
        min_width = dx * 0.5
        def model(xx, *p):
            return peak_model(xx, *p, model=settings.model, origin=center)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                p, covariance = curve_fit(
                    model, x, yn,
                    p0=[baseline / scale, slope / scale, max(float(np.max(reduced)) / scale, 0.01),
                        np.clip(guess_center, lower_center + 1e-12, upper_center - 1e-12),
                        max(dx, span / 6)],
                    bounds=([-np.inf, -np.inf, 0., lower_center, min_width],
                            [np.inf, np.inf, np.inf, upper_center, span]),
                    maxfev=6000, x_scale="jac",
                )
            fit.x = x
            fit.fitted = model(x, *p) * scale
            fit.residual = y - fit.fitted
            fit.rms = float(np.sqrt(np.mean(fit.residual ** 2)))
            width_error = float(np.sqrt(covariance[4, 4])) if covariance[4, 4] >= 0 else np.nan
            height = float(p[2] * scale)
            fit.center_ev = float(p[3])
            if p[4] <= dx or p[4] >= span * 0.95:
                fit.status = "width unresolved or reaches window limit"
            elif min(p[3] - lower_center, upper_center - p[3]) < dx * 0.25:
                fit.status = "center reaches tracking limit"
            elif not np.isfinite(width_error) or width_error > p[4]:
                fit.status = "width uncertainty too large"
            elif height <= 3.0 * fit.rms:
                fit.status = "peak below residual noise"
            else:
                fit.status = "ok"
                fit.fwhm_mev = float(p[4] * 1000)
                fit.fwhm_error_mev = width_error * 1000
                fit.height = height
                local_baseline = (p[0] + p[1] * (x - center)) * scale
                fit.area = float(np.trapezoid(y - local_baseline, x))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            fit.status = f"fit failed: {exc}"
    return tuple(results)


def draw_peak_trends(axes, results, *, log_power: bool, metric: str, power_limits=None,
                     log_intensity=True, log_linewidth=True, records=None):
    """Shared drawing path for the UI and exported trend figures."""
    for axis, log_y in zip(axes, (log_intensity, log_linewidth)):
        axis.clear()
        axis.set_yscale("log" if log_y else "linear")
        if log_y:
            plain_log_ticks(axis.yaxis)
        axis.set_xscale("log" if log_power else "linear")
        if log_power:
            plain_log_ticks(axis.xaxis)
        axis.set_xlabel("Power (uW)")
        axis.grid(alpha=0.25)
        if power_limits is not None:
            lo, hi = power_limits
            if lo < hi and (not log_power or lo > 0):
                axis.set_xlim(lo, hi)
    axes[0].set_ylabel("Peak height (a.u.)" if metric == "Peak height" else "Integrated PL (a.u. eV)")
    axes[1].set_ylabel("FWHM (meV)")
    series = {}
    series_records = {}
    for label, fits in results.items():
        count = max((len(getattr(fit, "components", (fit,))) for fit in fits), default=1)
        for index in range(count):
            name = f"Peak {index + 1}" if label == "Power" else f"{label} Peak {index + 1}"
            series[name] = tuple(getattr(fit, "components", (fit,))[index] for fit in fits)
            series_records[name] = (records or {}).get(label, ())
    origins = {}
    for label, fits in series.items():
        power = np.array([fit.power_uw for fit in fits])
        shown = np.isfinite(power) & ((power > 0) if log_power else True)
        # NaNs deliberately break curves at failed fits.
        intensity = np.array([fit.height if metric == "Peak height" else fit.area for fit in fits], dtype=float)
        width = np.array([fit.fwhm_mev for fit in fits], dtype=float)
        errors = np.array([fit.fwhm_error_mev for fit in fits])
        for axis, values, log_y in ((axes[0], intensity, log_intensity), (axes[1], width, log_linewidth)):
            if log_y and np.any(np.isfinite(values[shown]) & (values[shown] <= 0)):
                axis.text(.02, .98, "Nonpositive values hidden on log axis", transform=axis.transAxes,
                          va="top", fontsize=7)
            if log_y:
                values[values <= 0] = np.nan
        if log_linewidth:
            errors = np.where(errors < width, errors, np.nan)
        axes[0].plot(power[shown], intensity[shown], "o-", ms=4, label=label)
        axes[1].errorbar(power[shown], width[shown], yerr=errors[shown], fmt="o-", ms=4,
                         capsize=2, label=label)
        rows = series_records[label]
        if len(rows) == len(power):
            from core.power_workflow import record_sources
            sources = [tuple(record_sources(row)) for row in rows]
            if len(set(sources)) > 1:
                color = axes[0].lines[-1].get_color()
                for source in dict.fromkeys(sources):
                    marker = origins.setdefault(source, ("o", "s", "^", "D", "v", "P", "X")[len(origins) % 7])
                    selected = shown & np.array([s == source for s in sources])
                    for axis, values in zip(axes, (intensity, width)):
                        axis.scatter(power[selected], values[selected], marker=marker, color=color, s=28, zorder=4)
    for axis in axes:
        if results:
            axis.legend(fontsize=8)
    if origins:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker=marker, ls="", color="gray", label=f"Source {i + 1}")
                   for i, marker in enumerate(origins.values())]
        legend = axes[0].legend(handles=handles, fontsize=6, loc="upper left")
        axes[0].add_artist(legend)
        axes[0].legend(fontsize=8, loc="best")


def export_peak_analysis(folder, payload):
    """Export exactly the analyzed rows, including failed fits and provenance."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    folder = Path(folder)
    base = folder / "Power_peak_analysis"
    suffix = 1
    while any(base.with_suffix(ext).exists() for ext in (".csv", ".png", ".json")):
        base = folder / f"Power_peak_analysis_{suffix:02d}"
        suffix += 1
    settings = payload["settings"]
    fields = ["channel", "peak", "power_uW", "intensity_area_au_eV", "peak_height_au", "FWHM_meV",
              "FWHM_error_meV", "center_eV", "residual_rms", "status", "model", "window_min_eV",
              "window_max_eV", "source_file", "source_row", "source_provenance", "manual_refit"]
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, fits in payload["results"].items():
            records = payload["records"][label]
            for index, fit in enumerate(fits):
                record = records[index] if index < len(records) else None
                fit_settings = getattr(fit, "settings", settings)
                if hasattr(fit_settings, "peaks"):
                    lo = float(fit.x[0]) if fit.x is not None else ""
                    hi = float(fit.x[-1]) if fit.x is not None else ""
                else:
                    lo, hi = settings.center_ev - settings.half_window_ev, settings.center_ev + settings.half_window_ev
                for peak_index, peak in enumerate(getattr(fit, "components", (fit,))):
                    writer.writerow(dict(zip(fields, [label, f"Peak {peak_index + 1}", fit.power_uw, peak.area, peak.height,
                        peak.fwhm_mev, peak.fwhm_error_mev, peak.center_ev, peak.rms, peak.status,
                        fit_settings.model, lo, hi,
                        getattr(record, "file_name", ""), getattr(record, "row_index", ""),
                        getattr(record, "source_provenance", ""), getattr(fit, "manual", False)])))
    metadata = {"settings": asdict(settings), "power_axis": "log" if payload["log_power"] else "linear",
                "intensity_axis": "log" if payload.get("log_intensity", True) else "linear",
                "linewidth_axis": "log" if payload.get("log_linewidth", True) else "linear",
                "intensity_metric": payload["metric"], "background_constant": payload["background"],
                "baseline": "independent fitted linear baseline per spectrum",
                "area": ("fitted component integrated over the full measured spectrum" if hasattr(settings, "peaks")
                         else "measured spectrum minus fitted baseline, integrated over sampled fit window"),
                "uncertainty": "one standard error from least-squares covariance; no measurement noise model",
                "power_limits": payload["power_limits"]}
    metadata['power_law_fits'] = payload.get('power_laws', [])
    metadata['power_law_ranges'] = payload.get('power_law_ranges', [])
    metadata['power_law_uncertainty'] = 'OLS log-log standard error; excludes gain calibration uncertainty. Suggested intervals are exploratory.'
    if payload.get('power_laws'):
        law_path = base.with_name(base.name + '_power_law.csv')
        fields_law = list(payload['power_laws'][0])
        with law_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields_law)
            writer.writeheader()
            writer.writerows(payload['power_laws'])
    metadata["manual_refits"] = [
        {"channel": label, "power_uW": fit.power_uw, "settings": asdict(fit.settings),
         "parameters": list(fit.parameters), "status": fit.status}
        for label, fits in payload["results"].items() for fit in fits if getattr(fit, "manual", False)
    ]
    base.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    figure = Figure(figsize=(9, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    draw_peak_trends(axes, payload["results"], log_power=payload["log_power"],
                     metric=payload["metric"], power_limits=payload["power_limits"],
                     log_intensity=payload.get("log_intensity", True),
                     log_linewidth=payload.get("log_linewidth", True), records=payload.get("records"))
    if payload['metric'] == 'Integrated area':
        from core.power_law_analysis import draw_power_laws
        draw_power_laws(axes[0], payload.get('power_laws', []))
    figure.savefig(base.with_suffix(".png"), dpi=200)
    return base.with_suffix(".csv")
