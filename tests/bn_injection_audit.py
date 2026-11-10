"""Synthetic feature audit with full BN calibration refitting.

The expected injection is evaluated against the frozen pre-injection applied
background. Measurements are within-scan sensitivity checks, not independent
validation or proof of preservation of real spectral features.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_single import evaluate_bn_single


def audit_injection(energy, raw, background, gates, *, center_ev, sigma_mev, amplitude,
                    moving=False, movement_ev=.005, baseline_result=None,
                    settings=None, background_repeats=None, evaluator=evaluate_bn_single,
                    audit_selected_candidate=False):
    """Refit after injecting a positive Gaussian in each measurement gate."""
    e, z, b, g = (np.asarray(x, dtype=float) for x in (energy, raw, background, gates))
    if (e.ndim != 1 or len(e) < 3 or z.ndim != 2 or g.ndim != 1 or b.shape != e.shape
            or z.shape != (len(g), len(e)) or not len(g)
            or not all(np.isfinite(x).all() for x in (e, z, b, g))
            or np.any(z <= 0) or np.any(b <= 0)
            or not (np.all(np.diff(e) > 0) or np.all(np.diff(e) < 0))):
        raise ValueError('Expected finite positive gate-by-energy spectra and a monotonic energy grid.')
    if (not np.isfinite([center_ev, sigma_mev, amplitude, movement_ev]).all()
            or sigma_mev <= 0 or amplitude <= 0 or movement_ev < 0):
        raise ValueError('Injection needs finite center, positive width and amplitude, and nonnegative movement.')
    if background_repeats is not None:
        repeats = np.asarray(background_repeats, dtype=float)
        if repeats.ndim != 2 or repeats.shape[1] != len(e) or not np.isfinite(repeats).all() or np.any(repeats <= 0):
            raise ValueError('BN repeats must be finite positive spectra on the measured energy grid.')
    config = settings or BnCalibrationSettings(single_measurement=True, single_full_range=True)
    relative = np.zeros(len(g)) if np.ptp(g) == 0 else (g-g.min())/np.ptp(g)-.5
    centers = center_ev + (movement_ev*relative if moving else np.zeros_like(g))
    sigma = sigma_mev/1000.
    if np.any(centers-4*sigma < e.min()) or np.any(centers+4*sigma > e.max()):
        raise ValueError('The full four-sigma injection support must lie on the measured grid.')
    feature = amplitude*np.exp(-.5*((e[None, :]-centers[:, None])/sigma)**2)
    start = perf_counter()
    if baseline_result is None:
        baseline_result = evaluator(e, z, b, gates=g, settings=config, background_repeats=background_repeats)
    injected = evaluator(e, z*(1+feature), b, gates=g, settings=config, background_repeats=background_repeats)
    def audited_background(result):
        if audit_selected_candidate:
            if result.selected_method == 'original' or result.selected_method not in result.candidate_backgrounds:
                raise ValueError('Selected preview candidate is unavailable for injection audit.')
            return np.asarray(result.candidate_backgrounds[result.selected_method], float)
        return np.asarray(result.applied_background if result.eligible else b, float)
    baseline_b = audited_background(baseline_result)
    injected_b = audited_background(injected)
    if (baseline_b.shape != e.shape or injected_b.shape != e.shape or
            not np.isfinite(baseline_b).all() or not np.isfinite(injected_b).all() or
            np.any(baseline_b <= 0) or np.any(injected_b <= 0)):
        raise ValueError('Evaluator returned an invalid applied background.')
    observed = z*(1+feature)/injected_b-z/baseline_b
    expected = z/baseline_b*feature
    expected_bn = z/b*feature
    integration = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    rows = []
    for i, center in enumerate(centers):
        mask = np.abs(e-center) <= 4*sigma
        idx = int(np.argmin(np.abs(e-center)))
        if np.count_nonzero(mask) < 2:
            raise ValueError('Injection support must contain at least two measured points.')
        area = float(integration(observed[i, mask], e[mask]))
        expected_area = float(integration(expected[i, mask], e[mask]))
        original_area = float(integration(expected_bn[i, mask], e[mask]))
        rows.append(dict(gate=float(g[i]), center_ev=float(center), peak_observed=float(observed[i, idx]),
            peak_expected=float(expected[i, idx]), peak_ratio=float(observed[i, idx]/expected[i, idx]),
            peak_expected_original_bn=float(expected_bn[i, idx]),
            peak_ratio_original_bn=float(observed[i, idx]/expected_bn[i, idx]),
            area_observed=area, area_expected=expected_area, area_ratio=area/expected_area,
            area_expected_original_bn=original_area, area_ratio_original_bn=area/original_area))
    def model(result):
        if result.report.get('optimization_policy') in {'shared_bn_trend_v1', 'shared_bn_trend_v2'}:
            return 'shared_trend_normalization:' + result.selected_method
        return result.report.get('single_pair_scope', {}).get('model', result.selected_method)
    delta = np.log(injected_b/baseline_b)
    return dict(center_ev=float(center_ev), sigma_mev=float(sigma_mev), amplitude=float(amplitude),
        moving=bool(moving), movement_ev=float(movement_ev) if moving else 0.,
        audit_target='selected_preview_candidate' if audit_selected_candidate else 'applied_background',
        expected_reference=('frozen pre-injection selected preview candidate' if audit_selected_candidate
                            else 'frozen pre-injection applied background'),
        baseline_model=model(baseline_result), injected_model=model(injected),
        baseline_eligible=bool(baseline_result.eligible), injected_eligible=bool(injected.eligible),
        model_switched=(model(baseline_result), baseline_result.eligible) != (model(injected), injected.eligible),
        background_log_change=delta.tolist(), background_log_change_rms=float(np.sqrt(np.mean(delta**2))),
        background_log_change_max=float(np.max(np.abs(delta))), elapsed_seconds=perf_counter()-start,
        gates=rows, caveat='Synthetic within-scan sensitivity is not independent validation.')
