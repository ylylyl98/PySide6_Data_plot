"""Numerical helpers for power-law (log-log) fits.

The public functions deliberately return ordinary Python values where a result
is intended to be persisted as JSON.  Input arrays may be any array-like
objects accepted by :func:`numpy.asarray`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _arrays(power: Any, intensity: Any, valid: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        p = np.asarray(power, dtype=float).ravel()
        y = np.asarray(intensity, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("power and intensity must be numeric one-dimensional arrays") from exc
    if p.size != y.size:
        raise ValueError("power and intensity must have the same length")
    mask = np.isfinite(p) & np.isfinite(y) & (p > 0) & (y > 0)
    if valid is not None:
        v = np.asarray(valid, dtype=bool).ravel()
        if v.size != p.size:
            raise ValueError("valid must have the same length as power")
        mask &= v
    return p, y, mask


def fit_power_law(power, intensity, lo, hi, valid=None) -> dict:
    """Fit ``intensity = amplitude * power ** alpha`` over ``[lo, hi]``.

    The fit is ordinary least squares in natural-log coordinates.  Invalid,
    non-positive, masked, and out-of-range observations are reported through
    ``excluded`` and do not enter the fit.
    """
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError) as exc:
        raise ValueError("lo and hi must be finite positive numbers") from exc
    if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0 and hi > lo):
        raise ValueError("require lo > 0 and hi > lo")
    p, y, mask = _arrays(power, intensity, valid)
    in_range = mask & (p >= lo) & (p <= hi)
    pp, yy = p[in_range], y[in_range]
    if pp.size < 4 or np.unique(pp).size < 3:
        raise ValueError("at least 4 valid points and 3 distinct powers are required")
    x, z = np.log(pp), np.log(yy)
    xbar, zbar = float(np.mean(x)), float(np.mean(z))
    sxx = float(np.sum((x - xbar) ** 2))
    if not math.isfinite(sxx) or sxx <= 0:
        raise ValueError("at least 3 distinct powers are required")
    alpha = float(np.sum((x - xbar) * (z - zbar)) / sxx)
    intercept = zbar - alpha * xbar
    residual = z - (intercept + alpha * x)
    n = int(pp.size)
    log_rms = float(np.sqrt(np.mean(residual**2)))
    if n > 2:
        alpha_error = float(np.sqrt(np.sum(residual**2) / (n - 2) / sxx))
    else:  # guarded above, retained for defensive clarity
        alpha_error = float("nan")
    try:
        amplitude = float(math.exp(intercept))
    except OverflowError as exc:
        raise ValueError("power-law amplitude is not finite") from exc
    if not math.isfinite(amplitude):
        raise ValueError("power-law amplitude is not finite")
    return {
        "lower": float(lo),
        "upper": float(hi),
        "observed_lower": float(np.min(pp)),
        "observed_upper": float(np.max(pp)),
        "alpha": alpha,
        "alpha_error": alpha_error,
        "amplitude": amplitude,
        "n": n,
        "excluded": int(p.size - n),
        "log_rms": log_rms,
        "method": "OLS log-log",
    }


def local_slopes(power, intensity, valid=None, window=7):
    """Return center powers and centered moving-window log-log slopes."""
    try:
        window = int(window)
    except (TypeError, ValueError) as exc:
        raise ValueError("window must be an odd integer >= 3") from exc
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    p, y, mask = _arrays(power, intensity, valid)
    order = np.argsort(p[mask], kind="mergesort")
    p, y = p[mask][order], y[mask][order]
    if p.size < window:
        return np.array([], dtype=float), np.array([], dtype=float)
    x, z = np.log(p), np.log(y)
    half = window // 2
    xs, slopes = [], []
    for i in range(half, p.size - half):
        xx, zz = x[i - half : i + half + 1], z[i - half : i + half + 1]
        if np.unique(xx).size < 2:
            continue
        slopes.append(float(np.polyfit(xx, zz, 1)[0]))
        xs.append(float(p[i]))
    return np.asarray(xs), np.asarray(slopes)


def suggest_power_range(power, intensity, valid=None, tolerance=.15, min_points=6, min_decades=.5):
    """Find the broadest reliable near-linear, approximately unit-slope range."""
    if not (math.isfinite(float(tolerance)) and tolerance > 0):
        raise ValueError("tolerance must be positive")
    min_points, min_decades = int(min_points), float(min_decades)
    if min_points < 4 or min_decades <= 0:
        raise ValueError("invalid minimum range requirements")
    p, y, mask = _arrays(power, intensity, valid)
    order = np.argsort(p[mask], kind="mergesort")
    p, y = p[mask][order], y[mask][order]
    n = p.size
    candidates = []
    if n < min_points:
        raise ValueError("No reliable near-linear range found.")
    # Sample at most 60 endpoints, while always covering both ends.  Fits use
    # every point between the endpoints, including datasets larger than 500.
    endpoint_count = min(60, n)
    endpoints = np.unique(np.rint(np.linspace(0, n - 1, endpoint_count)).astype(int))
    slope_x, all_slopes = local_slopes(p, y, window=7 if n >= 7 else 5)
    for i in endpoints:
        for j in endpoints:
            if j <= i or j - i + 1 < min_points:
                continue
            if math.log10(p[j] / p[i]) < min_decades:
                continue
            try:
                fit = fit_power_law(p[i:j + 1], y[i:j + 1], p[i], p[j])
            except ValueError:
                continue
            if abs(fit["alpha"] - 1.0) > tolerance:
                continue
            if fit["alpha_error"] > tolerance:
                continue
            if fit["log_rms"] > tolerance:
                continue
            if all_slopes.size:
                inside = (slope_x >= p[i]) & (slope_x <= p[j])
                sl = all_slopes[inside]
            else:
                sl = all_slopes
            if sl.size and (np.max(np.abs(sl - 1.0)) > tolerance):
                continue
            candidates.append((math.log10(p[j] / p[i]), j - i + 1, -fit["log_rms"], fit))
    if not candidates:
        raise ValueError("No reliable near-linear range found.")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]
