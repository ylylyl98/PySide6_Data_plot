"""Cached noise evidence and reversible support filters for heatmap extrema.

SNR means prominence / robust detection-signal noise, not a probability.
The broad detrending window is used only to estimate noise; peak coordinates,
prominences and widths are never moved or refitted here.
"""
from bisect import bisect_left
from collections import defaultdict
from math import ceil

import numpy as np
from scipy.signal import savgol_filter


QUALITY_VERSION = 1
def prepare_quality_cache(dataset, pool, cancelled=None, progress=None):
    """Enrich legacy candidates once, without detecting peaks or mutating input."""
    if pool['dataset_key'] != dataset['key']:
        raise ValueError('Candidate cache belongs to another dataset.')
    local_pl = dataset.get('kind') == 'PL' and pool.get('settings', {}).get('pl_weak_features', False)
    version = 3 if local_pl else QUALITY_VERSION
    if (pool.get('quality', {}).get('version') == version
            and all('snr' in p and 'noise_sigma' in p for p in pool['points'])):
        return pool
    from core.peak_candidates import detection_spectrum
    rows = defaultdict(list)
    points = [dict(p) for p in pool['points']]
    for point in points:
        rows[point['row_index']].append(point)
    x = dataset['cube'].energy
    for row_number, row in enumerate(sorted(rows)):
        values = dataset['cube'].Z[row]
        if cancelled and cancelled():
            raise RuntimeError('Peak quality calculation cancelled.')
        # Always use the cached detector settings, including when new settings
        # are pending in the UI and have not yet rebuilt the candidate pool.
        signal = detection_spectrum(x, values, pool['settings'])
        noise = np.full(len(x), np.nan)
        edges = np.diff(np.r_[False, np.isfinite(signal), False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end - start < 7:
                continue
            xx = x[start:end]
            grid = np.linspace(xx[0], xx[-1], len(xx))
            uniform = np.interp(grid, xx, signal[start:end])
            window = min(51, (len(xx) - 1) // 2 * 2 + 1)
            if local_pl:
                from scipy.ndimage import median_filter
                raw = np.interp(grid, xx, values[start:end])
                residual = raw - uniform
                # With smoothing disabled, estimate high-frequency raw noise
                # using a small SG window without changing detection positions.
                if pool['settings']['detection_method'] == 'local':
                    residual = raw - savgol_filter(raw, min(11, window), 2)
                neighborhood = min(101, (len(xx) - 1) // 2 * 2 + 1)
                center = median_filter(residual, size=neighborhood, mode='reflect')
                sigma = median_filter(abs(residual - center), size=neighborhood, mode='reflect') / .67448975
                sigma = np.maximum(sigma, max(float(np.ptp(raw)) * 1e-12, np.finfo(float).tiny))
                noise[start:end] = np.interp(xx, grid, sigma)
            else:
                residual = uniform - savgol_filter(uniform, window, 2)
                sigma = float(np.median(abs(residual - np.median(residual))) / .67448975)
                noise[start:end] = max(sigma, float(np.ptp(uniform)) * 1e-12, np.finfo(float).tiny)
        # Legacy caches may contain interpolated coordinates: retain argmin's
        # lower-index tie behavior while searching all row positions at once.
        energies = np.asarray([point['energy'] for point in rows[row]])
        right = np.clip(np.searchsorted(x, energies), 0, len(x) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(abs(x[left] - energies) <= abs(x[right] - energies), left, right)
        for point, index in zip(rows[row], indices):
            sigma = float(noise[index])
            point['noise_sigma'] = sigma if np.isfinite(sigma) else None
            contrast = point.get('detection_contrast', 0.) if local_pl and point.get('candidate_origin') == 'curvature' else point['prominence']
            point['snr'] = float(contrast / sigma) if np.isfinite(sigma) else None
        if progress:
            progress(int((row_number + 1) * 100 / len(rows)))
    if cancelled and cancelled():
        raise RuntimeError('Peak quality calculation cancelled.')
    return {**pool, 'points': points, 'quality': dict(version=version,
        noise_estimator=('Local raw-minus-SG residual MAD / 0.67448975 over up to 101 samples; per finite segment; heuristic.' if local_pl else
        'MAD / 0.67448975 of detection signal minus an order-2 SG baseline (up to 51 samples), per finite energy segment; at least 7 samples.'),
        snr_definition=('Extremum prominence or provisional shoulder curvature contrast / local raw-noise estimate; heuristic, not a probability.' if local_pl else
                        'Candidate prominence / estimated detection-signal noise; heuristic, not a probability.'))}


def supported_candidates(points, gates, required, tolerance_mev, cancelled=None, *, adaptive=False):
    """Count same-polarity support among up to five sorted neighboring scans.

    Counts use the input set once, without recursive promotion. At dataset or
    scan-gap boundaries the required fraction is kept (ceil(required*n/5)).
    Tolerance is per scan row (twice as far for a second neighbor). A gap
    larger than 1.5 times the median scan step breaks a neighborhood.

    When adaptive is enabled, a failed base match may extend to one quarter of
    the smaller peak width, up to twice the base tolerance per row. Such extra
    matches must be mutual, uniquely nearest neighbors of the same polarity.
    Existing base matches are unchanged. A zero tolerance never expands.
    """
    if not 0 <= required <= 5 or tolerance_mev < 0:
        raise ValueError('Support must be 0–5 rows and match tolerance nonnegative.')
    order = np.argsort(gates)
    steps = np.diff(np.asarray(gates)[order])
    gap = 1.5 * float(np.median(steps)) if len(steps) else np.inf
    blocks = np.r_[0, np.cumsum(steps > gap)]
    windows = {}
    for n, row in enumerate(order):
        windows[int(row)] = [(int(order[j]), max(1, abs(j - n)))
                            for j in range(max(0, n - 2), min(len(order), n + 3)) if blocks[j] == blocks[n]]
    groups = defaultdict(list)
    for point in points:
        groups[(point['row_index'], point['polarity'])].append(point)
    energies = {}
    for key, group in groups.items():
        group.sort(key=lambda point: point['energy'])
        energies[key] = [point['energy'] for point in group]

    def unique_nearest(key, energy, margin):
        values = energies.get(key, [])
        index = bisect_left(values, energy)
        # The two closest elements must be among these four positions.
        candidates = sorted((abs(values[i] - energy), i)
                            for i in range(max(0, index - 2), min(len(values), index + 2)))
        if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= margin):
            return None
        return groups[key][candidates[0][1]]
    kept = []
    tolerance = tolerance_mev / 1000
    for point in points:
        if cancelled and cancelled():
            raise RuntimeError('Peak filtering cancelled.')
        adjacent = windows[point['row_index']]
        support = 0
        for row, distance in adjacent:
            values = energies.get((row, point['polarity']), [])
            reach = tolerance * distance
            index = bisect_left(values, point['energy'] - reach - 1e-12)
            base_match = index < len(values) and values[index] <= point['energy'] + reach + 1e-12
            if base_match:
                support += 1
                continue
            if not adaptive or tolerance <= 0:
                continue
            key = (row, point['polarity'])
            neighbor = unique_nearest(key, point['energy'], .25 * tolerance * distance)
            if neighbor is None:
                continue
            widths = (point.get('width_mev'), neighbor.get('width_mev'))
            if any(width is None or not np.isfinite(width) or width <= 0 for width in widths):
                continue
            expanded = min(2 * tolerance, min(widths) / 4000) * distance
            if abs(neighbor['energy'] - point['energy']) > expanded + 1e-12:
                continue
            own_key = (point['row_index'], point['polarity'])
            if unique_nearest(own_key, neighbor['energy'], .25 * tolerance * distance) is point:
                support += 1
        minimum = ceil(required * len(adjacent) / 5)
        if support >= minimum:
            kept.append({**point, 'neighbor_support': int(support),
                         'neighbor_available': len(adjacent), 'neighbor_required': minimum})
    return kept
