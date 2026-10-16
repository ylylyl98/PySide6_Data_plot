"""Gate-local resonance masks and conservative shared fine spectral response.

Shared texture is only identifiable under the assumption that reproducible fine
structure outside protected resonances is instrumental. Never extrapolate it
through an unsupported band, and retain the smooth-only result for comparison.
"""
import warnings
import numpy as np
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter
from core.load_cancellation import check_load_cancelled


def mask_regions(energy, protected):
    edges = np.r_[energy[0], (energy[:-1]+energy[1:])/2, energy[-1]]
    starts = np.flatnonzero(np.diff(np.r_[False, protected, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, protected, False].astype(int)) == -1)
    return tuple((float(edges[a]), float(edges[b])) for a, b in zip(starts, ends))


def gate_anchor_masks(energy, raw):
    """Detect both signs at every gate; moving peaks do not mask other gates."""
    from core.drr_background_quality import _screen_threshold
    grid = np.linspace(energy[0], energy[-1], len(energy))
    step = grid[1]-grid[0]
    values = np.array([np.interp(grid, energy, np.log(row)) for row in raw])
    protected = np.zeros(raw.shape, bool)
    for width in (.035, .070):
        check_load_cancelled()
        window = min(len(grid) if len(grid) % 2 else len(grid)-1, max(7, round(width/step) | 1))
        high = values-savgol_filter(values, window, 2, axis=1)
        threshold, _ = _screen_threshold(values, window)
        significant = abs(high) > threshold
        significant[:, :window//2] = significant[:, -(window//2):] = False
        for index, row in enumerate(significant):
            check_load_cancelled()
            starts = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)) == 1)
            ends = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)) == -1)
            for a, b in zip(starts, ends):
                if b-a >= 3:
                    protected[index] |= (energy >= grid[a]-.005) & (energy <= grid[b-1]+.005)
    return ~protected


def pooled_profile(values, anchors, *, minimum):
    supported = anchors.sum(axis=0) >= minimum
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        profile = np.nanmedian(np.where(anchors, values, np.nan), axis=0)
    return profile, supported


def _local_quadratic_trend(energy, values, width):
    """SG-style quadratic trend on measured coordinates, without resampling.

    Wavelength-calibrated detector pixels are not equally spaced in energy.
    A fixed number of pixels therefore cannot represent a fixed meV bandwidth.
    Construct a sparse local projection once and apply it to all gates.
    """
    size = len(energy)
    left = np.searchsorted(energy, energy-width/2)
    right = np.searchsorted(energy, energy+width/2, side='right')
    narrow = right-left < 7
    left[narrow] = np.clip(np.arange(size)[narrow]-3, 0, size-7)
    right[narrow] = left[narrow]+7
    offsets = np.r_[0, np.cumsum(right-left)]
    weights = np.empty(offsets[-1])
    columns = np.empty(offsets[-1], dtype=np.int32)
    for i, (a, b) in enumerate(zip(left, right)):
        if i % 64 == 0:
            check_load_cancelled()
        x = (energy[a:b]-energy[i])/(width/2)
        design = np.column_stack((np.ones(len(x)), x, x*x))
        # lstsq avoids squaring the condition number on uneven calibrations.
        weights[offsets[i]:offsets[i+1]] = np.linalg.lstsq(design.T, [1., 0., 0.], rcond=None)[0]
        columns[offsets[i]:offsets[i+1]] = np.arange(a, b)
    projection = csr_matrix((weights, columns, offsets), shape=(size, size))
    return (projection @ values.T).T


def shared_texture(energy, residual, anchors):
    """Validate fine structure in two disjoint groups of available gates.

    Return a log-multiplicative correction, its smoothly tapered coverage, and
    diagnostics. Only unprotected detrended samples supply the template. Detrend
    each measured spectrum before pooling; interpolating a pooled profile over
    missing bands would create artificial structure and false disagreement.
    """
    count, size = residual.shape
    zero = np.zeros(size)
    if count < 12:
        return zero, zero.copy(), dict(status='Insufficient gates', corrected_fraction=0.)
    if size < 7:
        return zero, zero.copy(), dict(status='Insufficient energy samples', corrected_fraction=0.)
    width = .035
    high = residual-_local_quadratic_trend(energy, residual, width)
    # Split the available gates at each energy, retaining their gate order.
    # A moving resonance can block one fixed half of a scan while leaving many
    # independent observations elsewhere. Never relax the original peak mask.
    rank = np.cumsum(anchors, axis=0)
    half = anchors.sum(axis=0)//2
    minimum = max(6, int(np.ceil(.15*count)))
    profiles, supports = [], []
    for part in (anchors & (rank <= half), anchors & (rank > half)):
        check_load_cancelled()
        profile, supported = pooled_profile(high, part, minimum=minimum)
        if supported.sum() < 7:
            return zero, zero.copy(), dict(status='Insufficient unprotected support', corrected_fraction=0.)
        profiles.append(np.where(supported, profile, 0.))
        supports.append(supported)
    a, b = profiles
    valid = supports[0] & supports[1]
    left = np.searchsorted(energy, energy-width/2)
    right = np.searchsorted(energy, energy+width/2, side='right')
    def local_sum(values):
        summed = np.r_[0., np.cumsum(values)]
        return summed[right]-summed[left]
    observations = local_sum(valid)
    common = (a+b)/2
    aa, bb, ab = (local_sum(v*valid) for v in (a*a, b*b, a*b))
    coherence = ab/np.maximum(np.sqrt(np.maximum(aa*bb, 0)), 1e-16)
    amplitude = np.sqrt(np.maximum(local_sum(common*common*valid), 0)/np.maximum(observations, 1))
    repeatable = ((coherence >= .65) & (amplitude >= .0005)
                  & (observations >= np.maximum(7, .5*(right-left))))
    # A fine-response correction is deliberately limited to weak structure;
    # large features and regions with insufficient independent support stay intact.
    accepted = valid & repeatable & (abs(common) <= .02)
    accepted[(energy < energy[0]+width/2) | (energy > energy[-1]-width/2)] = False
    # Each row was detrended before masking, so no interpolated band edges
    # need a half-window guard. Retain a 1 meV guard and 5 meV C2 taper, using
    # physical distances rather than median pixel spacing.
    before = np.maximum.accumulate(np.where(~accepted, energy, -np.inf))
    after = np.minimum.accumulate(np.where(~accepted, energy, np.inf)[::-1])[::-1]
    distance = np.minimum(energy-before, after-energy)
    t = np.clip((distance-.001)/.005, 0, 1)
    coverage = np.clip(t*t*t*(10+t*(-15+6*t)), 0, 1)
    if not np.any(coverage > .5):
        return zero, zero.copy(), dict(status='No repeatable fine structure', corrected_fraction=0.)
    profile, _ = pooled_profile(high, anchors, minimum=2*minimum)
    texture = np.nan_to_num(profile)*coverage
    return texture, coverage, dict(status='Applied',
        corrected_fraction=float(np.mean(coverage > .5)), rms_percent=float(100*np.sqrt(np.mean(texture**2))),
        uncorrected_regions_eV=mask_regions(energy, coverage == 0),
        validation='Disjoint lower/upper available-gate groups at each energy; >= 6 gates and 15% of scan per group; '
                   'local correlation >= 0.65; measured-energy 35 meV quadratic detrending; no gap interpolation')
