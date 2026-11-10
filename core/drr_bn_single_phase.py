"""Exploratory smooth phase model derived from a single measured BN texture.

This module only proposes a background. Same-scan fitting cannot distinguish a
sample feature that overlaps the measured BN response; no result is activated.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, CubicSpline
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.optimize import least_squares
from scipy.signal import hilbert, savgol_filter

from core.drr_bn_calibration import _check
from core.drr_bn_joint import production_d2
from core.drr_bn_settings import BnCalibrationSettings


def _d2_shrink_factor(before, after, tolerance):
    """Scale a failing window in proportion to its worst tolerance excess."""
    before=np.asarray(before,float);after=np.asarray(after,float);tolerance=np.asarray(tolerance,float)
    ratio=float(np.max((after-before)/np.maximum(tolerance,1e-15)))
    return min(1., .98/max(ratio,1.000000001))


def fit_phase_texture(e, z, b, fine, settings, cancellation=None):
    """Fit smooth phase/gain fields and return ``(candidate, diagnostics)``.

    Eligibility requires every unmasked raw and normalized per-gate local D2
    window and every full-axis DR/R projection check to pass. The robust mask
    only affects fitting; it never excludes observations from final checks.
    """
    _check(cancellation)
    config = settings if isinstance(settings, BnCalibrationSettings) else BnCalibrationSettings(**dict(settings))
    energy, data, background, texture = (np.asarray(v, dtype=float) for v in (e, z, b, fine))
    if (energy.ndim != 1 or data.ndim != 2 or data.shape[1] != len(energy) or not len(data)
            or background.shape != energy.shape or texture.shape != energy.shape
            or len(energy) < max(config.template_window, config.derivative_window + 2, 16)
            or not all(np.isfinite(v).all() for v in (energy, data, background, texture))
            or np.any(data <= 0) or np.any(background <= 0)
            or not (np.all(np.diff(energy) > 0) or np.all(np.diff(energy) < 0))):
        raise ValueError('Phase fitting needs positive finite spectra and a strictly monotonic energy grid.')
    if (config.template_window < 3 or config.template_window % 2 != 1
            or config.template_polyorder < 0 or config.template_polyorder >= config.template_window
            or len(config.single_smooth_gain_bounds) != 2 or not 0 < config.single_smooth_gain_bounds[0] < config.single_smooth_gain_bounds[1]):
        raise ValueError('Invalid phase fitting template or gain bounds.')
    reverse = energy[0] > energy[-1]
    if reverse:
        energy, data, background, texture = energy[::-1], data[:, ::-1], background[::-1], texture[::-1]
    pixels = np.arange(len(energy), dtype=float)
    analytic = hilbert(texture)
    amplitude, carrier = np.abs(analytic), np.unwrap(np.angle(analytic))
    k = 10
    breaks = np.linspace(0, len(energy)-1, k-2)
    knots = np.r_[[breaks[0]]*4, breaks[1:-1], [breaks[-1]]*4]
    W = BSpline.design_matrix(pixels, knots, 3).toarray()
    greville = np.array([np.mean(knots[i+1:i+4]) for i in range(k)])

    old_raw = data/background - 1
    old_raw_d2, used = production_d2(energy, old_raw, config)
    old_ratio = data/background
    old_norm = old_ratio/np.median(old_ratio, axis=1, keepdims=True)-1
    old_norm_d2, _ = production_d2(energy, old_norm, config)
    # Recompute moving-feature mask from this exact input; mask affects fit only.
    center = np.median(old_norm_d2, axis=0)
    dev = np.abs(old_norm_d2-center)
    median_dev = np.median(dev, axis=0)
    spread = 1.4826*np.median(np.abs(dev-median_dev), axis=0)
    threshold = median_dev + 4*np.maximum(spread, np.percentile(spread,25)*.1) + .1
    mask = maximum_filter1d((dev > threshold).astype(np.uint8), size=21, axis=1, mode='nearest').astype(bool)
    high = np.log(data)
    high -= savgol_filter(high, config.template_window, config.template_polyorder, axis=1)
    with np.errstate(all='ignore'):
        target = np.ma.median(np.ma.array(high, mask=mask), axis=0).filled(np.nan)
    missing = ~np.isfinite(target)
    target[missing] = np.median(high[:, missing], axis=0)
    disp = np.median(np.abs(high-np.median(high, axis=0)), axis=0)
    weight = (1-mask.mean(axis=0))/(1+(disp/max(3*np.median(disp), 1e-8))**2)
    band_weight = np.ones(len(energy))
    for ids in np.array_split(np.arange(len(energy)), 8):
        band_weight[ids] = 1/max(np.sqrt(np.mean(target[ids]**2)), .25*np.sqrt(np.mean(target**2)), 1e-5)
    weight *= (band_weight/np.median(band_weight))**2
    quad = analytic.imag
    centers, phase_seed, gain_seed = [], [], []
    for ids in np.array_split(np.arange(len(energy)), k):
        _check(cancellation)
        X = np.column_stack((texture[ids], quad[ids]))
        root = np.sqrt(weight[ids])
        coeff = np.linalg.lstsq(X*root[:, None], target[ids]*root, rcond=None)[0]
        centers.append(float(np.mean(ids)))
        phase_seed.append(float(np.arctan2(-coeff[1], coeff[0])))
        gain_seed.append(float(np.clip(np.linalg.norm(coeff), *config.single_smooth_gain_bounds)))
    seed = np.r_[np.interp(greville, centers, np.unwrap(phase_seed)), np.interp(greville, centers, gain_seed)]
    low = np.r_[np.full(k, -8*np.pi), np.full(k, config.single_smooth_gain_bounds[0])]
    high_bounds = np.r_[np.full(k, 8*np.pi), np.full(k, config.single_smooth_gain_bounds[1])]
    scale = max(float(np.median(np.abs(target-np.median(target)))*1.4826), 1e-5)
    def waveform(p):
        return (W@p[k:])*amplitude*np.cos(carrier + W@p[:k])
    def residual(p):
        _check(cancellation)
        difference = (waveform(p)-target)*np.sqrt(weight)/scale/np.sqrt(len(energy))
        return np.r_[difference, .015*np.diff(p[:k], 2), .015*np.diff(p[k:], 2), .002*p[:k], .002*(p[k:]-1)]
    fits = []
    for initial in (seed, np.r_[np.zeros(k), np.ones(k)]):
        _check(cancellation)
        fits.append(least_squares(residual, np.clip(initial, low+1e-8, high_bounds-1e-8),
            bounds=(low, high_bounds), loss='soft_l1', f_scale=.05, max_nfev=120, x_scale='jac', ftol=1e-5))
    fit = min(fits, key=lambda result: result.cost)
    delta = waveform(fit.x)-texture
    limit = min(.2, max(.02, 3*np.sqrt(np.mean(texture**2))))
    delta *= min(1., limit/max(float(np.max(np.abs(delta))), 1e-30))
    span = min(len(energy), max(int(config.local_validation_window), int(used)))
    stride = max(1, span//2)
    starts = sorted(set([*range(0, len(energy)-span+1, stride), len(energy)-span]))
    spline = CubicSpline(pixels, texture)
    from core.drr_bn_single import _full_axis_drr_check
    def evaluate(alpha):
        _check(cancellation)
        candidate = background*np.exp(alpha*delta)
        ratio = data/candidate
        raw_d2, _ = production_d2(energy, ratio-1, config)
        normalized = ratio/np.median(ratio, axis=1, keepdims=True)-1
        norm_d2, _ = production_d2(energy, normalized, config)
        failures, desired = [], np.ones(len(energy))
        for start in starts:
            stop = start+span
            for name, old, new in (('raw', old_raw_d2, raw_d2), ('normalized', old_norm_d2, norm_d2)):
                before = np.sqrt(np.mean(old[:, start:stop]**2, axis=1))
                after = np.sqrt(np.mean(new[:, start:stop]**2, axis=1))
                threshold = config.max_local_degradation_fraction*before + config.max_absolute_d2_degradation
                violation = after-before-threshold
                if np.any(violation > 0):
                    idx = int(np.argmax(violation))
                    failures.append(dict(kind=name, energy_range=[float(energy[start]), float(energy[stop-1])],
                        gate_index=idx, old_rms=float(before[idx]), new_rms=float(after[idx]), allowed=float(threshold[idx])))
                    left, right = max(0, start-12), min(len(energy), stop+12)
                    desired[left:right] = np.minimum(desired[left:right], _d2_shrink_factor(before, after, threshold))
        checks = _full_axis_drr_check(old_norm, normalized, energy, texture, spline, config)
        for item in checks:
            if not item['passed']:
                lo, hi = item['region']
                desired[(energy >= lo) & (energy <= hi)] = np.minimum(desired[(energy >= lo) & (energy <= hi)], .75)
        return candidate, failures, checks, desired
    alpha = np.ones(len(energy))
    iterations = 0
    for iterations in range(1, 42):
        candidate, failures, checks, desired = evaluate(alpha)
        if not failures and all(v['passed'] for v in checks):
            break
        updated = gaussian_filter1d(alpha*desired, sigma=20, mode='nearest')
        if np.max(np.abs(updated-alpha)) < 1e-8:
            break
        alpha = updated
    # Recompute after the last shrink; no masked result is used for acceptance.
    candidate, failures, checks, _ = evaluate(alpha)
    finite = bool(np.isfinite(candidate).all() and np.all(candidate > 0))
    raw_pass = not any(v['kind'] == 'raw' for v in failures)
    normalized_pass = not any(v['kind'] == 'normalized' for v in failures)
    drr_pass = all(v['passed'] for v in checks)
    success = bool(fit.success and finite and raw_pass and normalized_pass and drr_pass)
    info = dict(model='phase_ten_knots', independent_validation=False, success=success,
        node_count=k, basis_source='measured_bn_only', phase_limit_radians=float(8*np.pi),
        gain_bounds=list(config.single_smooth_gain_bounds), strength_min=float(alpha.min()),
        strength_max=float(alpha.max()), strength_mean=float(alpha.mean()),
        all_gate_raw_d2_pass=raw_pass, all_gate_normalized_d2_pass=normalized_pass,
        full_axis_drr_pass=drr_pass, checked_d2_windows=len(starts), failed_d2_windows=failures,
        fit_success=bool(fit.success), phase_knots=fit.x[:k].tolist(), gain_knots=fit.x[k:].tolist(),
        knot_pixels=greville.tolist(), alpha_knots=np.interp(greville, pixels, alpha).tolist(),
        alpha_min=float(alpha.min()), alpha_max=float(alpha.max()), iterations=iterations,
        fit_mask_fraction=float(mask.mean()), correction_limit=float(limit), used_window=int(used),
        d2_shrink_policy='ratio_dependent_v1',
        raw_and_normalized_d2_failed_windows=failures, full_axis_drr_reports=checks,
        full_axis_d2_check=not failures, full_axis_drr_check=all(v['passed'] for v in checks),
        finite_positive=finite, eligible=success,
        warning='Same-scan model; physical features aligned with BN texture may be changed.')
    return (candidate[::-1].copy() if reverse else candidate.copy()), info
