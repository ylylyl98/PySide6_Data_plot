"""Gate-local resonance masks and conservative shared fine spectral response.

Shared texture is only identifiable under the assumption that reproducible fine
structure outside protected resonances is instrumental. Never extrapolate it
through an unsupported band, and retain the smooth-only result for comparison.
"""
from collections import OrderedDict
from threading import RLock
import warnings
import numpy as np
from scipy.sparse import csr_matrix
from scipy.signal import savgol_filter
from core.load_cancellation import check_load_cancelled


_PROJECTION_CACHE_MAX_ENTRIES = 8
_PROJECTION_CACHE_MAX_BYTES = 16 * 1024 * 1024
_projection_cache = OrderedDict()
_projection_cache_bytes = 0
_projection_cache_lock = RLock()


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


def _build_local_quadratic_projection(energy, width):
    """Build the measured-coordinate projection without retaining spectra."""
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
    return csr_matrix((weights, columns, offsets), shape=(size, size))


def _local_quadratic_projection(energy, width):
    """Reuse a bounded set of projections by complete coordinates and width."""
    global _projection_cache_bytes
    check_load_cancelled()
    # Own the coordinate snapshot so a caller cannot change the cache key's
    # meaning while construction runs. No raw intensity data enters this cache.
    energy = np.array(energy, copy=True)
    key = (float(width), energy.dtype.str, energy.shape, energy.tobytes())
    with _projection_cache_lock:
        cached = _projection_cache.get(key)
        if cached is not None:
            check_load_cancelled()
            _projection_cache.move_to_end(key)
            return cached[0]
    # Build outside the lock: unrelated requests and cancellation stay responsive.
    projection = _build_local_quadratic_projection(energy, width)
    size = len(key[-1]) + sum(a.nbytes for a in (projection.data, projection.indices, projection.indptr))
    with _projection_cache_lock:
        check_load_cancelled()
        cached = _projection_cache.get(key)
        if cached is not None:
            _projection_cache.move_to_end(key)
            return cached[0]
        if size <= _PROJECTION_CACHE_MAX_BYTES:
            _projection_cache[key] = (projection, size)
            _projection_cache_bytes += size
            while (len(_projection_cache) > _PROJECTION_CACHE_MAX_ENTRIES
                   or _projection_cache_bytes > _PROJECTION_CACHE_MAX_BYTES):
                _, (_, removed_size) = _projection_cache.popitem(last=False)
                _projection_cache_bytes -= removed_size
    return projection


def _local_quadratic_trend(energy, values, width):
    """SG-style quadratic trend on measured coordinates, without resampling.

    Wavelength-calibrated detector pixels are not equally spaced in energy.
    A fixed number of pixels therefore cannot represent a fixed meV bandwidth.
    Reuse the sparse projection across files and apply it to all gates.
    """
    projection = _local_quadratic_projection(energy, width)
    check_load_cancelled()
    result = (projection @ values.T).T
    check_load_cancelled()
    return result


def _masked_local_quadratic_trend(energy, values, anchors, width):
    """Detrend eligible samples without allowing protected peaks into the fit.

    Keep the cached projection exactly for fully observed local windows. Only
    eligible centres whose windows include protected samples need a new fit.
    A partial window needs ten anchors and a well-conditioned quadratic design;
    otherwise its centre cannot contribute to shared-texture estimation.
    """
    projection = _local_quadratic_projection(energy, width)
    safe_values = np.where(anchors, values, 0.)
    check_load_cancelled()
    trend = (projection @ safe_values.T).T
    # CSR rows contain the same contiguous windows, including the existing
    # seven-sample fallback on coarse detector grids.
    lengths = np.diff(projection.indptr)
    left = projection.indices[projection.indptr[:-1]]
    right = left + lengths
    usable = anchors.copy()
    partial = np.zeros_like(anchors)
    row_batch = max(1, min(64, 65536 // len(energy)))
    for start in range(0, len(values), row_batch):
        check_load_cancelled()
        mask = anchors[start:start+row_batch]
        prefix = np.pad(np.cumsum(mask, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
        counts = prefix[:, right]-prefix[:, left]
        partial[start:start+row_batch] = mask & (counts != lengths)
    usable[partial] = False
    for i in np.flatnonzero(partial.any(axis=0)):
        check_load_cancelled()
        a, b = left[i], right[i]
        x = (energy[a:b]-energy[i])/(width/2)
        design = np.column_stack((np.ones(len(x)), x, x*x))
        products = (design[:, :, None]*design[:, None, :]).reshape(len(x), 9)
        rows = np.flatnonzero(partial[:, i])
        batch_size = max(1, min(64, 65536 // len(x)))
        for start in range(0, len(rows), batch_size):
            check_load_cancelled()
            selected = rows[start:start+batch_size]
            mask = anchors[selected, a:b]
            normal = (mask @ products).reshape(-1, 3, 3)
            eigenvalues = np.linalg.eigvalsh(normal)
            good = ((mask.sum(axis=1) >= 10)
                    & (eigenvalues[:, 0] > 1e-7*eigenvalues[:, -1]))
            if not good.any():
                continue
            selected = selected[good]
            rhs = safe_values[selected, a:b] @ design
            trend[selected, i] = np.linalg.solve(normal[good], rhs[..., None])[:, 0, 0]
            usable[selected, i] = True
    check_load_cancelled()
    return trend, usable


def shared_texture(energy, residual, anchors):
    """Validate fine structure in two disjoint groups of available gates.

    Return a log-multiplicative correction, its smoothly tapered coverage, and
    diagnostics. Protected values enter neither detrending nor pooling. Detrend
    each measured spectrum using its local anchors; interpolating a pooled
    profile over missing bands would create artificial structure.
    """
    count, size = residual.shape
    zero = np.zeros(size)
    if count < 12:
        return zero, zero.copy(), dict(status='Insufficient gates', corrected_fraction=0.)
    if size < 7:
        return zero, zero.copy(), dict(status='Insufficient energy samples', corrected_fraction=0.)
    width = .035
    trend, anchors = _masked_local_quadratic_trend(energy, residual, anchors, width)
    high = np.where(anchors, residual, 0.)-trend
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
    # Local trends used only anchors, so protected values cannot leak through
    # the detrending window. Retain a 1 meV guard and 5 meV C2 taper, using
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
                   'local correlation >= 0.65; anchor-only measured-energy 35 meV quadratic detrending; '
                   'partial windows need >= 10 anchors and quadratic normal condition < 1e7; no gap interpolation')
