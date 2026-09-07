"""Display-only transforms for MCD peak maps."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def second_derivative_map(energy, raw, window_points: int = 35):
    """Return a uniform-energy grid and SG second derivative of each spectrum."""
    e = np.asarray(energy, dtype=float).ravel()
    z = np.asarray(raw, dtype=float)
    if z.ndim != 2 or z.shape[1] != e.size:
        raise ValueError("raw must be a 2-D array with one column per energy")
    if e.size < 7 or not np.all(np.isfinite(e)) or np.unique(e).size != e.size:
        raise ValueError("energy must contain finite unique values")
    order = np.argsort(e)
    es = e[order]
    n = es.size
    w = int(window_points)
    w = max(7, min(w, n if n % 2 else n - 1))
    if w % 2 == 0:
        w -= 1
    grid = np.linspace(es[0], es[-1], n)
    step = float(grid[1] - grid[0])
    out = np.full((z.shape[0], n), np.nan, dtype=float)
    for i, row in enumerate(z[:, order]):
        if not np.all(np.isfinite(row)):
            continue
        uniform = np.interp(grid, es, row)
        out[i] = savgol_filter(uniform, w, 3, deriv=2, delta=step, axis=0, mode="interp")
    half = w // 2
    out[:, :half] = np.nan
    out[:, -half:] = np.nan
    return grid, out, w
