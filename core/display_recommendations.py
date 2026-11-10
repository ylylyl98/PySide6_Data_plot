"""Conservative, display-only suggestions for measured heatmaps.

Recommendations never modify the input grid or analysis settings.  A weak-feature
scale deliberately saturates strong structure, so its clipped fraction is reported.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColorRecommendation:
    low: float
    high: float
    saturated_fraction: float
    background: float
    noise: float
    message: str


def _selection(cube, xlim, ylim):
    x = np.asarray(cube.energy, dtype=float).reshape(-1)
    y = np.asarray(cube.gate, dtype=float).reshape(-1)
    z = np.asarray(cube.Z, dtype=float)
    if z.shape != (len(y), len(x)):
        raise ValueError("cube.Z must have shape (len(gate), len(energy))")
    xm, ym = np.isfinite(x), np.isfinite(y)
    if xlim is not None:
        lo, hi = sorted(map(float, xlim)); xm &= (x >= lo) & (x <= hi)
    if ylim is not None:
        lo, hi = sorted(map(float, ylim)); ym &= (y >= lo) & (y <= hi)
    return x[xm], y[ym], z[np.ix_(ym, xm)]


def _quiet_statistics(z):
    """Estimate background and within-window variation in quieter windows.

    Window scores use robust spread rather than magnitude. Center each selected
    window separately before pooling its residuals: differences between window
    baselines are not measurement noise. The returned scale remains a display
    heuristic; broad structure within a window may contribute to it.
    """
    if not z.size:
        return 0., 0.
    width = max(3, min(31, z.shape[1] // 20))
    blocks = []
    for start in range(0, z.shape[1], width):
        values = z[:, start:start + width].ravel()
        values = values[np.isfinite(values)]
        if values.size < 5:
            continue
        center = float(np.median(values))
        sigma = float(1.4826 * np.median(np.abs(values - center)))
        blocks.append((sigma, center, values))
    if not blocks:
        values = z[np.isfinite(z)]
        if not values.size:
            return 0., 0.
        center = float(np.median(values))
        return center, float(1.4826 * np.median(np.abs(values - center)))
    blocks.sort(key=lambda item: item[0])
    chosen = blocks[:max(1, (len(blocks) + 2) // 3)]
    values = np.concatenate([item[2] for item in chosen])
    center = float(np.median(values))
    residuals = np.concatenate([values - background for _, background, values in chosen])
    return center, float(1.4826 * np.median(np.abs(residuals)))


def _guard_weak_bounds(finite, low, high, strength):
    """Keep a distribution envelope visible without exceeding Overall's span.

    A quiet patch can belong to an unrepresentative baseline. Including a
    central quantile envelope limits clipping on either tail, instead of
    allowing that patch to saturate most of a map. The envelope contracts with
    strength, as does the proposed interval; intersecting with fixed Overall
    limits preserves this nesting. Tied data may clip less than the budget.
    """
    budget = float(np.interp(np.log2(strength), [-1., 0., 1., 2.], [.20, .40, .60, .75]))
    tail = budget * 50.
    overall_low, guard_low, guard_high, overall_high = np.percentile(
        finite, [1., tail, 100. - tail, 99.])
    guarded_low = float(max(overall_low, min(low, guard_low)))
    guarded_high = float(min(overall_high, max(high, guard_high)))
    adjusted = guarded_low != low or guarded_high != high
    return guarded_low, guarded_high, adjusted


def recommend_color_bounds(cube, *, xlim=None, ylim=None, mode='overall', strength=1.0, log_scale=False):
    """Return a color suggestion for the selected view, or None if unusable.

    Overall uses the 1st/99th percentiles. Weak starts from quiet-window
    variation, then protects the data distribution from excessive saturation
    and caps the interval at Overall. Strength .5/1/2/4 permits up to roughly
    20/40/60/75 percent saturation, respectively, before UI rounding or Fix.
    Log suggestions only include positive samples and always have positive limits.
    """
    if mode not in ('overall', 'weak'):
        raise ValueError("mode must be 'overall' or 'weak'")
    strength = float(strength)
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('strength must be finite and positive')
    x, _, z = _selection(cube, xlim, ylim)
    z = z[:, np.argsort(x)]
    finite = z[np.isfinite(z)]
    if log_scale:
        finite = finite[finite > 0]
    if not finite.size:
        return None
    background, noise = _quiet_statistics(z)
    if mode == 'weak' and noise > 0 and not log_scale:
        # Put the quiet background slightly below the midpoint for a pale,
        # rather than saturated, background with common diverging colormaps.
        low, high = background - 3 * noise / strength, background + 5 * noise / strength
        low, high, adjusted = _guard_weak_bounds(finite, low, high, strength)
        message = 'Weak-feature contrast from within-window variation; strong signals may saturate.'
        if adjusted:
            message += ' Background coverage protected; range capped at Overall.'
    else:
        low, high = np.percentile(finite, [1, 99])
        message = ('Positive-data percentile scale; weak linear contrast is unavailable on a log scale.'
                   if log_scale and mode == 'weak' else 'Overall 1st–99th percentile scale.')
    if not high > low:
        if log_scale:
            low, high = max(float(low) / 1.01, np.finfo(float).tiny), float(high) * 1.01
        else:
            pad = max(abs(float(low)) * .01, 1e-12)
            low, high = float(low) - pad, float(high) + pad
    fraction = float(np.mean((finite < low) | (finite > high)))
    return ColorRecommendation(float(low), float(high), fraction, background, noise, message)


def recommend_x_range(cubes):
    """Suggest conservative common energy bounds, or None if evidence is weak.

    Detect structure relative to local scan-to-scan variation, then add generous
    padding. A suggestion is withheld for short grids and weak/ambiguous data.
    """
    proposals = []
    for cube in cubes:
        x, y, z = _selection(cube, None, None)
        if len(x) < 30 or len(y) < 5 or not np.all(np.isfinite(z)):
            return None
        order = np.argsort(x); x = x[order]; z = z[:, order]
        scan_order = np.argsort(y); y = y[scan_order]; z = z[scan_order]
        if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
            return None
        # Subtract a local baseline after smoothing, and estimate the actual
        # residual noise in quiet energy windows. This does not assume adjacent
        # derivative samples are independent. The broad baseline scale is kept
        # conservative: slowly varying structure may prevent useful cropping.
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(z, 2., axis=1, mode='nearest')
        baseline_width = max(8., min(25., len(x) / 65.))
        residual = smooth - gaussian_filter1d(smooth, baseline_width, axis=1, mode='nearest')
        window = max(10, len(x) // 45)
        scales = []
        for start in range(0, len(x), window):
            values = residual[:, start:start + window].ravel()
            center = np.median(values)
            scales.append(float(1.4826 * np.median(np.abs(values - center))))
        noise = float(np.quantile(scales, .25))
        if not np.isfinite(noise) or noise <= 0:
            return None
        # Three neighboring scans and two neighboring energies reject isolated
        # fluctuations without requiring a large fraction of all scan rows.
        above = np.abs(residual) > 5 * noise
        scan_runs = np.zeros_like(above, dtype=bool)
        for offset in range(3):
            scan_runs[offset:len(z) - 2 + offset] |= (
                above[:-2] & above[1:-1] & above[2:])
        support_count = np.sum(scan_runs, axis=0) >= 3
        support = np.convolve(support_count.astype(float), np.ones(3), mode='same') >= 2
        width = 5
        indices = np.flatnonzero(support)
        if len(indices) < 2:
            return None
        pad = max(width * 2, int(.05 * len(x)))
        first, last = max(0, indices[0] - pad), min(len(x) - 1, indices[-1] + pad)
        # No meaningful cropping or structure extends close to either edge.
        if first == 0 and last == len(x) - 1:
            return None
        proposals.append((float(x[first]), float(x[last])))
    if not proposals:
        return None
    return min(v[0] for v in proposals), max(v[1] for v in proposals)
