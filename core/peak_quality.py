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
    if (pool.get('quality', {}).get('version') == QUALITY_VERSION
            and all('snr' in p and 'noise_sigma' in p for p in pool['points'])):
        return pool
    from core.peak_candidates import detection_spectrum
    rows = defaultdict(list)
    points = [dict(p) for p in pool['points']]
    for point in points:
        rows[point['row_index']].append(point)
    x = dataset['cube'].energy
    for row, values in enumerate(dataset['cube'].Z):
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
            residual = uniform - savgol_filter(uniform, window, 2)
            sigma = float(np.median(abs(residual - np.median(residual))) / .67448975)
            noise[start:end] = max(sigma, float(np.ptp(uniform)) * 1e-12, np.finfo(float).tiny)
        for point in rows[row]:
            index = int(np.argmin(abs(x - point['energy'])))
            sigma = float(noise[index])
            point['noise_sigma'] = sigma if np.isfinite(sigma) else None
            point['snr'] = float(point['prominence'] / sigma) if np.isfinite(sigma) else None
        if progress:
            progress(int((row + 1) * 100 / len(dataset['cube'].gate)))
    if cancelled and cancelled():
        raise RuntimeError('Peak quality calculation cancelled.')
    return {**pool, 'points': points, 'quality': dict(version=QUALITY_VERSION,
        noise_estimator='MAD / 0.67448975 of detection signal minus an order-2 SG baseline (up to 51 samples), per finite energy segment; at least 7 samples.',
        snr_definition='Candidate prominence / estimated detection-signal noise; heuristic, not a probability.')}


def supported_candidates(points, gates, required, tolerance_mev, cancelled=None):
    """Count same-polarity support among up to five sorted neighboring scans.

    Counts use the input set once, without recursive promotion. At dataset or
    scan-gap boundaries the required fraction is kept (ceil(required*n/5)).
    Tolerance is per scan row (twice as far for a second neighbor). A gap
    larger than 1.5 times the median scan step breaks a neighborhood.
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
    energies = defaultdict(list)
    for point in points:
        energies[(point['row_index'], point['polarity'])].append(point['energy'])
    for group in energies.values():
        group.sort()
    kept = []
    tolerance = tolerance_mev / 1000
    for point in points:
        if cancelled and cancelled():
            raise RuntimeError('Peak filtering cancelled.')
        adjacent = windows[point['row_index']]
        support = 0
        for row, distance in adjacent:
            values = energies[(row, point['polarity'])]
            reach = tolerance * distance
            index = bisect_left(values, point['energy'] - reach - 1e-12)
            support += index < len(values) and values[index] <= point['energy'] + reach + 1e-12
        minimum = ceil(required * len(adjacent) / 5)
        if support >= minimum:
            kept.append({**point, 'neighbor_support': int(support),
                         'neighbor_available': len(adjacent), 'neighbor_required': minimum})
    return kept
