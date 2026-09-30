"""Mixed SG derivatives and conservative, data-dependent window suggestions."""
import numpy as np
from scipy.signal import find_peaks, peak_widths
from core.loader import DataCube


def validate_axis(axis):
    x = np.asarray(axis, float)
    if x.ndim != 1 or x.size < 5 or not np.all(np.isfinite(x)):
        raise ValueError('Mixed derivative needs at least 5 finite samples on each axis.')
    d = np.diff(x)
    if not (np.all(d > 0) or np.all(d < 0)):
        raise ValueError('Mixed derivative requires strictly monotonic X and Y axes; constant or repeated coordinates cannot be differentiated.')
    return x


def effective_window(requested, count, poly):
    minimum = max(5, int(poly) + 2)
    minimum += minimum % 2 == 0
    maximum = count if count % 2 else count - 1
    if minimum > maximum:
        raise ValueError(f'Mixed derivative needs at least {minimum} samples for polynomial order {poly}.')
    window = max(minimum, min(int(requested), maximum))
    return min(window + (window % 2 == 0), maximum)


def window_span(axis, window):
    x = np.asarray(axis, float)
    return float(np.median(np.abs(x[window-1:] - x[:len(x)-window+1])))


def recommend_window(axis, traces, poly=2, *, maximum=21):
    x = validate_axis(axis)
    values = np.asarray(traces, float).reshape(-1, len(x))
    # A bounded set of complete traces estimates noise without a full-grid scan.
    values = values[np.linspace(0, len(values)-1, min(17, len(values)), dtype=int)]
    values = values[np.all(np.isfinite(values), axis=1)]
    desired = 5
    if values.size:
        span = np.percentile(values, 95, axis=1) - np.percentile(values, 5, axis=1)
        noise = np.median(np.abs(np.diff(values, n=2, axis=1)), axis=1) / 1.65
        ratio = float(np.median(noise / np.maximum(span, 1e-12)))
        desired = 5 + 2 * int(np.clip(np.ceil(ratio * 60), 0, 8))
        # Cap smoothing at half of detectable feature widths, subject to the
        # minimum window required by the polynomial. This is a starting point.
        widths = []
        for row, amplitude in zip(values, span):
            for sign in (1, -1):
                peaks, _ = find_peaks(sign*row, prominence=max(float(amplitude)*.3, 1e-12))
                if peaks.size:
                    widths.extend(peak_widths(sign*row, peaks)[0])
        if widths:
            desired = min(desired, max(5, int(np.percentile(widths, 20)/2)))
    return effective_window(min(desired, maximum), len(x), poly)


def _first_derivative(values, axis, window, poly):
    x = validate_axis(axis)
    z = np.asarray(values, float)
    result = np.empty_like(z)
    for i in range(len(x)):
        start = min(max(0, i-window//2), len(x)-window)
        offsets = x[start:start+window] - x[i]
        scale = np.max(np.abs(offsets))
        design = np.polynomial.polynomial.polyvander(offsets/scale, poly)
        weights = np.linalg.pinv(design)[1]/scale
        result[..., i] = z[..., start:start+window] @ weights
    return result


def mixed_derivative(cube, window_x, window_y, poly=2):
    x, y = validate_axis(cube.energy), validate_axis(cube.gate)
    if not 1 <= int(poly) <= 6:
        raise ValueError('Mixed derivative polynomial order must be 1 to 6.')
    z = np.asarray(cube.Z, float)
    if z.shape != (len(y), len(x)):
        raise ValueError('Mixed derivative data must match the Y by X grid.')
    wx = effective_window(window_x, len(x), poly)
    wy = effective_window(window_y, len(y), poly)
    dx = _first_derivative(z, x, wx, poly)
    result = _first_derivative(dx.T, y, wy, poly).T
    label = f'd2(DR/R)/(dE d{cube.gate_label})'
    return DataCube(x.copy(), y.copy(), result, cube.gate_label,
                    f'{cube.title} (dXdY)', label, cube.gate_unit, cube.y_axis_semantic), wx
