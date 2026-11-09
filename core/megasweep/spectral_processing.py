"""Processing of spectral slices on their measured energy coordinates."""
from __future__ import annotations

import numpy as np


def process_spectral_slice(cut, mode='spectrum', window=9, *, is_rc=False):
    """Return a new result; never mutate acquired spectra or their coordinates.

    The second derivative uses a local cubic least-squares polynomial (quadratic
    for a three-point window). Actual energy offsets handle unequal spacing.
    At either edge the nearest full window is used, without extrapolating data.
    A missing channel propagates NaN through the affected fitting windows.
    """
    result = dict(cut)
    result['processing'] = mode
    result['signal_label'] = 'RC (ΔI/I₀)' if is_rc else 'PL Intensity (a.u.)'
    if mode == 'spectrum':
        return result
    if mode != 'second_derivative':
        raise ValueError(f'Unknown spectral processing: {mode}')
    energy = np.asarray(cut['energy'], dtype=float)
    spectra = np.asarray(cut['spectra'], dtype=float)
    if energy.ndim != 1 or spectra.ndim != 2 or spectra.shape[1] != energy.size:
        raise ValueError('Spectra must contain one column per energy channel.')
    if not isinstance(window, (int, np.integer)) or window < 3 or window % 2 != 1 or window > energy.size:
        raise ValueError('Derivative window must be odd, at least 3, and no larger than the channel count.')
    if not np.all(np.isfinite(energy)) or len(np.unique(energy)) != energy.size:
        raise ValueError('The energy axis must contain finite, distinct values for differentiation.')
    order = np.argsort(energy)
    energy_sorted, values = energy[order], spectra[:, order]
    derivative = np.empty_like(values)
    degree = min(3, window - 1)
    for channel in range(energy.size):
        start = min(max(0, channel - window // 2), energy.size - window)
        indices = slice(start, start + window)
        offsets = energy_sorted[indices] - energy_sorted[channel]
        scale = np.max(np.abs(offsets))
        design = np.vander(offsets / scale, N=degree + 1, increasing=True)
        weights = 2 * np.linalg.pinv(design)[2] / scale**2
        derivative[:, channel] = values[:, indices] @ weights
    result['spectra'] = np.empty_like(derivative)
    result['spectra'][:, order] = derivative
    result['derivative_window'] = int(window)
    result['signal_label'] = 'd²RC/dE² (eV⁻²)' if is_rc else 'd²PL/dE² (a.u./eV²)'
    return result
