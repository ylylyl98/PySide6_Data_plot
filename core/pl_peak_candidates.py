"""Experimental PL shoulder landmarks; evidence, not physical peak claims.

Suggestions are derived from curvature at multiple scales. Fit original linear
samples and review the outcome; shoulders need not be separate components.
"""
import numpy as np
from scipy.signal import find_peaks, savgol_coeffs, savgol_filter


def suggest_pl_candidates(x, values, base_candidates, config=None):
    """Return provisional shoulder suggestions, without altering inputs.

    ``config`` is reserved for detection-only options (currently none). Filter
    choices do not change detection or cache identity. Energies are original
    measured samples. Finite runs are processed independently, without bridging
    missing values. Returned ``width_mev`` is a provisional seed width based on
    curvature scale, not a measured peak FWHM. A shoulder has no measurable
    topographic prominence; separate curvature metrics are provided instead.
    """
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    if x.ndim != 1 or values.shape != x.shape or len(x) < 25 or not np.isfinite(x).all():
        return []
    delta = np.diff(x)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        return []
    if delta[0] < 0:
        x, values = x[::-1], values[::-1]
    finite = np.isfinite(values)
    if finite.sum() < 25 or np.ptp(values[finite]) <= 0:
        return []
    candidates = [p for p in base_candidates if p.get('polarity', 'peak') == 'peak'
                  and np.isfinite(p.get('energy', np.nan)) and np.isfinite(p.get('prominence', np.nan))]
    if not candidates:
        return []
    main = max(candidates, key=lambda p: p['prominence'])
    width = float(main.get('width_mev', 0)) / 1000
    if not np.isfinite(width) or width <= 0:
        return []
    all_span = float(np.ptp(values[finite]))
    edges = np.diff(np.r_[False, finite, False].astype(int))
    suggestions = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        xx, raw = x[start:end], values[start:end]
        if len(xx) < 25 or not xx[0] <= main['energy'] <= xx[-1]:
            continue
        grid = np.linspace(xx[0], xx[-1], len(xx)); step = float(grid[1]-grid[0])
        y = np.interp(grid, xx, raw)
        third = np.diff(y, n=3)
        sigma = max(float(np.median(abs(third-np.median(third))) / (.67448975*np.sqrt(20))), all_span*1e-10)
        groups = []
        center_scale = float(np.clip(width * .4, .005, .009))
        windows = sorted({max(7, int(round(center_scale * scale / step)) | 1) for scale in (.8, 1., 1.2)})
        if len(windows) != 3 or max(windows) > min(151, len(xx)//3):
            continue
        for window in windows:
            curve = -savgol_filter(y, window, 3, deriv=2, delta=step)
            noise = sigma*np.linalg.norm(savgol_coeffs(window, 3, deriv=2, delta=step))
            loc, props = find_peaks(curve, prominence=max(5*noise, np.max(abs(curve))*.03),
                                    distance=max(1, int(.0015/step)), width=(None, None))
            groups.append([(grid[k], float(curve[k]), float(props['prominences'][j]),
                            float(props['widths'][j]*step), float(noise), window)
                           for j,k in enumerate(loc) if window//2 <= k < len(xx)-window//2
                           and max(.003, width*.15) < abs(grid[k]-main['energy']) < width*2])
        for energy, curvature, strength, local_width, noise, window in groups[1]:
            tolerance=max(.0015, 3*step)
            matches=[min((p for p in group if abs(p[0]-energy) <= tolerance),
                         key=lambda p:abs(p[0]-energy),default=None) for group in groups]
            if any(p is None for p in matches) or sum(p[1] > 5*p[4] for p in matches) < 2:
                continue
            # A meaningful nearby measured extremum already supplies a seed.
            if any(abs(p['energy']-energy) <= max(.0015, 3*step) and p['prominence'] >= 5*sigma
                   for p in candidates):
                continue
            index=int(np.argmin(abs(xx-energy)))
            seed_width = max(local_width, window*step*.5)
            contrast = strength * (window*step)**2 / 8
            suggestions.append(dict(energy=float(xx[index]), amplitude=float(raw[index]), polarity='peak',
                prominence=0., prominence_fraction=0., width_mev=float(seed_width*1000),
                candidate_origin='curvature', provisional=True, width_definition='provisional curvature seed width; not FWHM',
                curvature_width_mev=float(local_width*1000), detection_contrast=float(contrast),
                detection_contrast_fraction=float(contrast/all_span), curvature_score=float(strength/max(noise,np.finfo(float).tiny)),
                scale_support=3, curvature_windows=windows))
    selected=[]
    for p in sorted(suggestions,key=lambda p:-p['curvature_score']):
        if all(abs(p['energy']-q['energy']) >= .0015 for q in selected):
            selected.append(p)
            if len(selected) == 4: break
    return sorted(selected,key=lambda p:p['energy'])
