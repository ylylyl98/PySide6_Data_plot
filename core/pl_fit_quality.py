"""Conservative PL-specific model selection and fit diagnostics.

The underlying Power multi-peak fitter remains unchanged. Numerical model
selection and residual diagnostics are evidence for review, not physical
identification of spectral components.
"""
import numpy as np

from core.power_multi_peaks import MultiPeakSettings, PeakSeed, fit_multi_spectrum
from core.pl_fit_models import PLSeed, fit_pl_model


def _strongest_measured_peak(x, y, points):
    """Choose the highest seed above the same edge baseline used by PL fits.

    Manual seeds need not have a detector prominence. Compare all seeds on
    the measured linear spectrum rather than mixing optional metadata sources.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    finite = np.isfinite(x) & np.isfinite(y)
    xf, yf = x[finite], y[finite]
    if len(xf) < 2 or xf[-1] <= xf[0]:
        return max(range(len(points)), key=lambda i: points[i].get('prominence') or
                   points[i].get('detection_contrast') or 0.)
    edge = max(2, len(xf) // 20)
    slope = (np.median(yf[-edge:]) - np.median(yf[:edge])) / (xf[-1] - xf[0])
    origin = float(np.mean(xf))
    baseline = float(np.median(yf[:edge]) + slope * (origin - np.mean(xf[:edge])))
    def height(point):
        energy = point['energy']
        return float(np.interp(energy, xf, yf) - baseline - slope * (energy - origin))
    return max(range(len(points)), key=lambda i: height(points[i]))


def _fit_model(x, y, gate, points, model, manual_ids, *, width_factor=1., prominence_initial=False, cancelled=None):
    step = float(np.median(np.diff(x)))
    centers = [p['energy'] for p in points]
    edges = np.r_[x[0], .5 * (np.asarray(centers[1:]) + centers[:-1]), x[-1]]
    seeds = []
    constrained = model.startswith('Asymmetric ') or any(p.get('fit_center_fixed') or p.get('fit_center_min') is not None or p.get('fit_center_max') is not None for p in points)
    for i, point in enumerate(points):
        width = max(float(point['width_mev']) / 1000, .0001)
        lo, hi = float(edges[i]), float(edges[i + 1])
        shoulder = point.get('candidate_origin') in ('curvature', 'shoulder')
        if point['branch_id'] not in manual_ids:
            radius = max(3 * step, min(width * (1. if shoulder else .5), .010))
            lo, hi = max(lo, centers[i] - radius), min(hi, centers[i] + radius)
        if point.get('fit_center_min') is not None:
            lo = max(lo, float(point['fit_center_min']))
        if point.get('fit_center_max') is not None:
            hi = min(hi, float(point['fit_center_max']))
        height = (point.get('detection_contrast') if shoulder else point.get('prominence')) if (prominence_initial or shoulder) else None
        if height is not None and (not np.isfinite(height) or height <= 0):
            height = None
        if constrained:
            seeds.append(PLSeed(centers[i], width * width_factor, height=height, center_min_ev=lo, center_max_ev=hi, center_fixed=bool(point.get('fit_center_fixed'))))
        else:
            seeds.append(PeakSeed(centers[i], width * width_factor, height=height, center_min_ev=lo, center_max_ev=hi))
    if constrained:
        primary = _strongest_measured_peak(x, y, points) if model.startswith('Asymmetric ') else None
        result, meta = fit_pl_model(x, y, gate, seeds, model, asymmetric_index=primary, cancelled=cancelled)
        result.pl_meta = meta
        return result
    return fit_multi_spectrum(x, y, gate, MultiPeakSettings(tuple(seeds), model))


def _bic(result):
    residual = result.residual
    if residual is None or not len(residual) or not np.all(np.isfinite(residual)):
        return np.inf
    n = len(residual)
    rss = max(float(np.dot(residual, residual)), np.finfo(float).tiny)
    return n * np.log(rss / n) + getattr(result, 'pl_meta', {}).get('n_vary', len(result.parameters)) * np.log(n)


def _local_bic_difference(full, reduced, x, y, point):
    """BIC(full) - BIC(reduced) in a one-linewidth window, for review.

    Both fits use the same complete raw spectrum; this local diagnostic does
    not refit a separate baseline. Correlated residuals limit its interpretation.
    """
    valid_x = np.asarray(x)[np.isfinite(y)]
    if full.residual is None or reduced.residual is None:
        return None
    width = point.get('width_mev')
    if width is None or not np.isfinite(width) or width <= 0:
        return None
    mask = np.abs(valid_x - point['energy']) <= width / 1000
    n = int(np.count_nonzero(mask))
    if n < max(12, len(full.parameters) + 3):
        return None
    def score(result):
        residual = result.residual[mask]
        rss = max(float(np.dot(residual, residual)), np.finfo(float).tiny)
        return n * np.log(rss / n) + getattr(result, 'pl_meta', {}).get('n_vary', len(result.parameters)) * np.log(n)
    return float(score(full) - score(reduced))


def fit_pl_spectrum(x, y, gate, points, model, manual_ids, cancelled=None):
    """Fit seeds and conservatively remove at most one redundant auto seed.

    Compare the full model against at most three alternatives, tried in order
    of lower measured prominence. Require a 10-point global BIC improvement
    and no strong local BIC evidence for the full model before pruning. Try a
    second initialization only for a weak companion. The last peak and all
    explicit manual seeds stay. Correlated residuals limit BIC interpretation.
    """
    def check():
        if cancelled and cancelled():
            raise RuntimeError('Peak fitting cancelled.')

    check()
    full = _fit_model(x, y, gate, points, model, manual_ids, cancelled=cancelled)
    # One additional initialization only when a weak companion is present.
    # This stays within five solves per row, including deletion trials.
    initializations_tested = 1
    strengths = [p.get('prominence') for p in points]
    finite_strengths = [float(v) for v in strengths if v is not None and np.isfinite(v) and v > 0]
    has_shoulder = any(p.get('candidate_origin') in ('curvature', 'shoulder') for p in points)
    if len(points) > 1 and (has_shoulder or (len(finite_strengths) >= 2 and min(finite_strengths) < .2 * max(finite_strengths))):
        check()
        alternate = _fit_model(x, y, gate, points, model, manual_ids,
                               width_factor=2., prominence_initial=True, cancelled=cancelled)
        initializations_tested += 1
        if _bic(alternate) < _bic(full):
            full = alternate
    best, best_score, selected = full, _bic(full), list(points)
    tested = 0
    options = []
    local_protected = []
    if len(points) > 1 and np.isfinite(best_score):
        options = [p for p in points if p['branch_id'] not in manual_ids]
        options.sort(key=lambda p: (p.get('prominence', np.inf), p['energy']))
        for removed in options[:3]:
            check()
            reduced = [p for p in points if p is not removed]
            if not reduced:
                continue
            candidate = _fit_model(x, y, gate, reduced, model, manual_ids, cancelled=cancelled)
            tested += 1
            score = _bic(candidate)
            if score + 10 < best_score:
                local_delta = _local_bic_difference(full, candidate, x, y, removed)
                if local_delta is not None and local_delta <= -10:
                    local_protected.append(removed['branch_id'])
                    continue
                best, best_score, selected = candidate, score, reduced
    check()
    residual = best.residual
    quality = dict(status='unavailable', rms_fraction=None, residual_lag1=None,
                   bic=float(best_score) if np.isfinite(best_score) else None,
                   excluded_count=len(points) - len(selected),
                   alternatives_tested=tested, alternatives_skipped=max(0, len(options) - tested),
                   initializations_tested=initializations_tested, local_protected_count=len(local_protected),
                   local_protected_ids=local_protected)
    finite_y = np.asarray(y)[np.isfinite(y)]
    if residual is not None and len(residual) > 2 and len(finite_y):
        span = float(np.ptp(finite_y))
        fraction = float(best.rms / span) if span > 0 else None
        a, b = residual[:-1], residual[1:]
        lag = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.
        if not np.isfinite(lag):
            lag = 0.
        quality.update(rms_fraction=fraction, residual_lag1=lag,
                       status='review residual structure' if fraction is not None and
                           (fraction > .03 or fraction > .005 and lag > .5) else 'no large residual flag')
    return best, selected, quality
