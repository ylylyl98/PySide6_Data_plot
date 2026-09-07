"""Local mixed absorptive/dispersive fits for one MCD resonance.

The fitter intentionally works on one seeded local window.  A seed identifies
the resonance to fit; the fitted centre is a model parameter and is kept
separate from that locator identity by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares


FIT_OK = "ok"
FIT_INSUFFICIENT = "insufficient_points"
FIT_GAP = "gap"
FIT_FAILED = "fit_failed"


@dataclass(frozen=True)
class LocalFitResult:
    status: str
    center_ev: float | None = None
    gamma_mev: float | None = None
    parameters: tuple[float, ...] = ()
    rmse: float | None = None
    n_points: int = 0
    seed_ev: float | None = None
    window_ev: tuple[float, float] | None = None
    background_model: str = "linear"
    locator_energy_ev: float | None = None
    fit_x_mev: tuple[float, ...] = ()
    fit_y: tuple[float, ...] = ()
    message: str = ""
    median: float | None = None
    scale: float | None = None


def mixed_model(x_mev: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate ``b0+b1*x+(A+D*t)/(1+t²)`` with ``t=(x-c)/gamma``."""
    p = np.asarray(parameters, dtype=float)
    if p.size < 6:
        raise ValueError("mixed model needs [b0, b1, A, D, center_mev, gamma_mev]")
    b0, b1, amplitude, dispersive, center, gamma = p[:6]
    t = (np.asarray(x_mev, dtype=float) - center) / gamma
    denominator = 1.0 + t * t
    value = b0 + b1 * np.asarray(x_mev, dtype=float)
    return value + (amplitude + dispersive * t) / denominator


def quadratic_model(x_mev: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate the optional quadratic-background comparison model."""
    p = np.asarray(parameters, dtype=float)
    if p.size < 7:
        raise ValueError("quadratic model needs [b0, b1, A, D, center_mev, gamma_mev, q]")
    return mixed_model(x_mev, p[:6]) + p[6] * (np.asarray(x_mev, dtype=float) / 10.0) ** 2


def _normalise(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    median = float(np.nanmedian(y))
    q05, q95 = np.nanpercentile(y, (5.0, 95.0))
    scale = max(float(q95 - q05), float(np.nanstd(y)), 1.0)
    return (y - median) / scale, median, scale


def _starts(x: np.ndarray, y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[np.ndarray]:
    extrema = [float(x[int(np.argmin(y))]), float(x[int(np.argmax(y))]), 0.0, -3.0, 3.0]
    centers = sorted({float(np.clip(v, lower[4] + 1e-6, upper[4] - 1e-6)) for v in extrema})
    scale = max(0.1, float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5)))
    widths = (0.25, 0.6, 1.2, 2.5, 5.0)
    phases = ((1.0, 0.0), (-1.0, 0.0), (0.7, 0.7), (0.7, -0.7), (-0.7, 0.7), (-0.7, -0.7))
    starts: list[np.ndarray] = []
    for center in centers:
        for width in widths:
            for phase_a, phase_d in phases:
                starts.append(np.clip(np.array([0.0, 0.0, phase_a * scale, phase_d * scale, center, width], dtype=float), lower + 1e-8, upper - 1e-8))
    return starts


def _fit_parameters(x: np.ndarray, y: np.ndarray, *, background_model: str, max_starts: int = 150) -> tuple[np.ndarray | None, float, str]:
    qlo, qhi = float(np.min(x)), float(np.max(x))
    lower = np.array([-4.0, -0.5, -6.0, -6.0, qlo + 0.25, 0.15], dtype=float)
    upper = np.array([4.0, 0.5, 6.0, 6.0, qhi - 0.25, min(10.0, max(2.0, 0.75 * (qhi - qlo)))], dtype=float)
    if qhi - qlo <= 0.5:
        return None, float("inf"), "window_too_narrow"
    starts = _starts(x, y, lower, upper)[: max(1, int(max_starts))]
    successful: list[tuple[Any, float]] = []
    for start in starts:
        try:
            fit = least_squares(lambda p: mixed_model(x, p) - y, start, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=2500)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if bool(fit.success) and np.all(np.isfinite(fit.x)) and np.all(np.isfinite(fit.fun)):
            successful.append((fit, float(np.dot(fit.fun, fit.fun))))
    if not successful:
        return None, float("inf"), "no_converged_start"
    best, rss = min(successful, key=lambda item: item[1])
    params = np.asarray(best.x, dtype=float)
    if background_model.casefold() not in {"linear", "quadratic"}:
        raise ValueError(f"Unknown local background model: {background_model}")
    fit_model = mixed_model
    all_successful = successful
    lower_check, upper_check = lower, upper
    if background_model.casefold() == "quadratic":
        qlo_bound, qhi_bound = -4.0, 4.0
        lower_check, upper_check = np.r_[lower, qlo_bound], np.r_[upper, qhi_bound]
        all_successful = []
        for base_start in starts:
            qstart = np.r_[base_start, 0.0]
            try:
                fit_q = least_squares(lambda p: quadratic_model(x, p) - y, qstart, bounds=(lower_check, upper_check), method="trf", x_scale="jac", max_nfev=2500)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if bool(fit_q.success) and np.all(np.isfinite(fit_q.x)) and np.all(np.isfinite(fit_q.fun)):
                all_successful.append((fit_q, float(np.dot(fit_q.fun, fit_q.fun))))
        if not all_successful:
            return None, float("inf"), "quadratic_fit_failed"
        best, rss = min(all_successful, key=lambda item: item[1])
        params = np.asarray(best.x, dtype=float)
        fit_model = quadratic_model
    residual = np.asarray(best.fun, dtype=float)
    jac = np.asarray(best.jac, dtype=float)
    singular = np.linalg.svd(jac, compute_uv=False) if jac.ndim == 2 else np.array([])
    rank = int(np.count_nonzero(singular > max(float(singular[0]), 1.0) * 1e-10)) if singular.size else 0
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    bound_hits = [int(i) for i, (v, lo, hi) in enumerate(zip(params, lower_check, upper_check)) if abs(float(v - lo)) < 1e-5 or abs(float(v - hi)) < 1e-5]
    near = [float(fit.x[4]) for fit, score in all_successful if score <= rss * (1.0 + 1.0e-3) + 1e-12]
    center_spread = float(max(near) - min(near)) if near else float("inf")
    expected_rank = 7 if background_model.casefold() == "quadratic" else 6
    reason = ""
    if bound_hits:
        reason = "parameter_bound_hit:" + ",".join(map(str, bound_hits))
    elif rank < expected_rank:
        reason = f"rank_deficient:{rank}/{expected_rank}"
    elif not np.isfinite(condition) or condition > 1e8:
        reason = f"ill_conditioned:{condition:.3g}"
    elif center_spread > 0.5:
        reason = f"nonunique_center_minima:{center_spread:.3g}meV"
    status = "rejected_" + reason if reason else "ok"
    return params, rss, status


def fit_mixed(x_mev: np.ndarray, y: np.ndarray, *, background_model: str = "linear", max_starts: int = 150) -> dict[str, Any]:
    """Fit a normalized local spectrum and return diagnostics-compatible values."""
    x = np.asarray(x_mev, dtype=float).ravel()
    values = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(values)
    if int(valid.sum()) < 12:
        return {"status": FIT_INSUFFICIENT, "n_points": int(valid.sum()), "params": None}
    order = np.argsort(x[valid], kind="stable")
    x, values = x[valid][order], values[valid][order]
    if np.any(np.diff(x) <= 0):
        return {"status": FIT_FAILED, "n_points": int(x.size), "params": None, "message": "x must be strictly increasing"}
    yn, median, scale = _normalise(values)
    params, rss, status = _fit_parameters(x, yn, background_model=background_model, max_starts=max_starts)
    if params is None:
        return {"status": status, "n_points": int(x.size), "params": None, "median": median, "scale": scale}
    model = quadratic_model if background_model.casefold() == "quadratic" else mixed_model
    residual = yn - model(x, params)
    return {
        "status": status,
        "params": tuple(float(v) for v in params),
        "center_mev": float(params[4]),
        "gamma_mev": float(params[5]),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "rss": float(rss),
        "n_points": int(x.size),
        "median": median,
        "scale": scale,
        "x_mev": tuple(float(v) for v in x),
        "y": tuple(float(v) for v in yn),
        "fit_y": tuple(float(v) for v in model(x, params)),
        "background_model": background_model.casefold(),
    }


def fit_local_resonance(
    energy_ev: np.ndarray,
    spectrum: np.ndarray,
    *,
    seed_energy_ev: float,
    window_ev: tuple[float, float] | None = None,
    window_before_ev: float = 0.0136,
    window_after_ev: float = 0.0164,
    background_model: str = "linear",
    locator_energy_ev: float | None = None,
    max_starts: int = 150,
) -> LocalFitResult:
    """Fit one local window around a locator seed.

    NaN/inf rows are treated as a gap.  A gap causes an explicit failed fit so
    downstream plots can leave the point unavailable rather than forcing zero.
    """
    seed = float(seed_energy_ev)
    if not np.isfinite(seed):
        raise ValueError("seed_energy_ev must be finite")
    bounds = window_ev or (seed - float(window_before_ev), seed + float(window_after_ev))
    lo, hi = map(float, bounds)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("window_ev must be a finite increasing pair")
    energy = np.asarray(energy_ev, dtype=float).ravel()
    values = np.asarray(spectrum, dtype=float).ravel()
    if energy.size != values.size:
        raise ValueError("energy and spectrum must have matching sizes")
    mask = (energy >= lo) & (energy <= hi)
    selected_e = energy[mask]
    selected_y = values[mask]
    if selected_e.size < 12:
        return LocalFitResult(FIT_INSUFFICIENT, seed_ev=seed, window_ev=(lo, hi), background_model=background_model, locator_energy_ev=locator_energy_ev, n_points=int(selected_e.size))
    if not np.all(np.isfinite(selected_e)) or not np.all(np.isfinite(selected_y)):
        return LocalFitResult(FIT_GAP, seed_ev=seed, window_ev=(lo, hi), background_model=background_model, locator_energy_ev=locator_energy_ev, n_points=int(selected_e.size), message="nonfinite point in local fit window")
    fit = fit_mixed((selected_e - seed) * 1000.0, selected_y, background_model=background_model, max_starts=max_starts)
    center_mev = fit.get("center_mev")
    status = str(fit.get("status", FIT_FAILED))
    accepted = status == FIT_OK
    return LocalFitResult(
        status,
        None if center_mev is None or not accepted else seed + float(center_mev) / 1000.0,
        None if fit.get("gamma_mev") is None or not accepted else float(fit["gamma_mev"]),
        tuple(fit.get("params") or ()),
        None if fit.get("rmse") is None else float(fit["rmse"]),
        int(fit.get("n_points", selected_e.size)), seed, (lo, hi), background_model.casefold(), locator_energy_ev,
        tuple(float(v) for v in (selected_e - seed) * 1000.0), tuple(float(v) for v in fit.get("fit_y", ())), str(fit.get("message", "")),
        None if fit.get("median") is None else float(fit["median"]), None if fit.get("scale") is None else float(fit["scale"]),
    )


def local_fit_cache_key(result: Any, *, source: str, background_model: str, window_ev: tuple[float, float], seed_energy_ev: float, settings: tuple[Any, ...] = ()) -> tuple[Any, ...]:
    """Build a stable cache key without hashing mutable result internals."""
    return (id(result), str(getattr(result, "source_file", "")), str(source).casefold(), str(background_model).casefold(), tuple(float(v) for v in window_ev), round(float(seed_energy_ev), 12), tuple(settings))
