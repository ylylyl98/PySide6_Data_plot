"""Conservative, offline group BN comparison with a frozen final holdout.

This module estimates an empirical reference texture, never a physical ground
truth. No caller array is modified. ``selected_method`` records the choice made
on selection repeats; a failed final check changes only ``applied_method`` back
to ``original``. Candidate parameters are never refitted using final repeats.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import hashlib
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from core.load_cancellation import LoadCancelled, check_load_cancelled
from core.drr_bn_joint import products, repeat_scores, joint_checks
from core.drr_bn_d2_objective import (fit_d2_priority, audit_training_influence, fit_regions,
                                    ranked_influence_probes, _detrender, INFLUENCE_POLICY, POLICY)


@dataclass(frozen=True)
class BnCalibrationSettings:
    core_region: tuple[float, float] = (1.900, 1.970)
    train_region: tuple[float, float] = (1.900, 1.935)
    validation_region: tuple[float, float] = (1.945, 1.970)
    taper_width: float = .010
    gate_range: tuple[float, float] | None = None
    gate_count: int = 12
    template_window: int = 101
    template_polyorder: int = 3
    max_shift_pixels: float = .75
    gain_bounds: tuple[float, float] = (.75, 1.25)
    min_repeats: int = 4
    min_band_points: int = 12
    # Operational acceptance thresholds, not statistical confidence levels.
    min_improvement_fraction: float = .10
    min_absolute_improvement: float = 1e-5  # dimensionless DR/R
    drr_noninferiority_fraction: float = .02  # operational allowance, not a noise estimate
    drr_noninferiority_absolute: float = 1e-5
    derivative_window: int = 21
    derivative_polyorder: int = 2
    min_absolute_d2_improvement: float = .1  # (DR/R) / eV^2; operational, not a noise estimate
    max_local_degradation_fraction: float = .05
    max_absolute_drr_degradation: float = 1e-5
    max_absolute_d2_degradation: float = .1  # (DR/R) / eV^2
    local_validation_window: int = 31
    protected_drr_absolute_tolerance: float = 1e-12
    protected_d2_absolute_tolerance: float = 1e-7  # (DR/R) / eV^2; no amplitude normalization
    max_temporal_shift_difference: float = .15
    max_temporal_gain_difference: float = .10
    max_bn_shift_difference: float = .05
    max_bn_gain_difference: float = .05
    trusted_regions: tuple[tuple[float, float], ...] = ()
    uncertain_regions: tuple[tuple[float, float], ...] = ((1.940, 2.020),)
    protected_regions: tuple[tuple[float, float], ...] = ((1.580, 1.860),)
    protection_guard_pixels: int = 12
    train_regions: tuple[tuple[float, float], ...] = ()


@dataclass
class BnCalibrationResult:
    original_background: np.ndarray
    candidate_backgrounds: dict[str, np.ndarray]
    selected_method: str
    applied_method: str
    applied_background: np.ndarray
    eligible: bool
    status: str
    reasons: tuple[str, ...]
    report: dict[str, Any] = field(default_factory=dict)


_FAMILIES = ('local_shift', 'local_shift_gain')
_STRENGTHS = (.25, .5, .75, 1.)
_METHOD_SPECS = {family if alpha == 1 else f'{family}_s{format(alpha*100, "g").replace(".", "p")}': (family, alpha)
                 for family in _FAMILIES for alpha in _STRENGTHS}
_METHODS = ('original', *_METHOD_SPECS)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _check(cancellation):
    check_load_cancelled()
    if cancellation is None:
        return
    if hasattr(cancellation, 'check'):
        cancellation.check()
    elif callable(cancellation):
        if cancellation():
            raise LoadCancelled()
    elif hasattr(cancellation, 'is_set') and cancellation.is_set():
        raise LoadCancelled()


def _mask(energy, regions):
    result = np.zeros(len(energy), dtype=bool)
    for lo, hi in regions:
        result |= (energy >= lo) & (energy <= hi)
    return result


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 - 15*x + 6*x*x)


def _support(energy, settings):
    lo, hi = settings.core_region
    width = settings.taper_width
    support = _smoothstep((energy-lo+width)/width) * _smoothstep((hi+width-energy)/width)
    # Native-stencil guard plus smooth shoulders around protected regions.
    # Joint screening below also checks the actual derivative/regridding path;
    # a native-stencil guard alone is never accepted as preservation evidence.
    for p_lo, p_hi in settings.protected_regions:
        inside = np.flatnonzero((energy >= p_lo) & (energy <= p_hi))
        if inside.size:
            a = max(0, int(inside.min())-settings.protection_guard_pixels)
            b = min(len(energy)-1, int(inside.max())+settings.protection_guard_pixels)
            p_lo, p_hi = min(p_lo, energy[a], energy[b]), max(p_hi, energy[a], energy[b])
        distance = np.maximum(p_lo-energy, energy-p_hi)
        support *= _smoothstep(distance/width)
    return support


def _rms(z):
    return float(np.sqrt(np.mean(np.asarray(z)**2)))


def _polynomial(energy, mask):
    x = energy[mask]
    x = 2*(x-x.min())/np.ptp(x)-1
    return np.column_stack([np.ones(len(x)), x, x*x])


def _settings_error(settings):
    intervals = [settings.core_region, settings.train_region, settings.validation_region, settings.gain_bounds]
    intervals += list(settings.train_regions)
    intervals += list(settings.trusted_regions)+list(settings.uncertain_regions)+list(settings.protected_regions)
    if settings.gate_range is not None:
        intervals.append(settings.gate_range)
    for interval in intervals:
        if len(interval) != 2 or not np.isfinite(interval).all() or interval[0] >= interval[1]:
            return 'invalid_settings_region'
    for name in ('taper_width', 'max_shift_pixels', 'min_absolute_improvement',
                 'min_absolute_d2_improvement', 'max_local_degradation_fraction',
                 'max_absolute_drr_degradation', 'max_absolute_d2_degradation',
                 'protected_drr_absolute_tolerance', 'protected_d2_absolute_tolerance',
                 'drr_noninferiority_fraction', 'drr_noninferiority_absolute',
                 'max_temporal_shift_difference', 'max_temporal_gain_difference',
                 'max_bn_shift_difference', 'max_bn_gain_difference'):
        if not np.isfinite(getattr(settings, name)) or getattr(settings, name) <= 0:
            return 'invalid_settings_threshold'
    for name, minimum in [('gate_count', 1), ('template_window', 5), ('template_polyorder', 1),
                          ('min_repeats', 4), ('min_band_points', 6), ('protection_guard_pixels', 0)]:
        value = getattr(settings, name)
        if not isinstance(value, (int, np.integer)) or value < minimum:
            return 'invalid_settings_integer'
    for name, minimum in [('derivative_window', 5), ('derivative_polyorder', 2),
                          ('local_validation_window', 5)]:
        value = getattr(settings, name)
        if not isinstance(value, (int, np.integer)) or value < minimum:
            return 'invalid_settings_integer'
    if settings.derivative_polyorder >= settings.derivative_window:
        return 'invalid_settings_derivative'
    if settings.max_local_degradation_fraction >= 1 or settings.drr_noninferiority_fraction >= 1:
        return 'invalid_settings_threshold'
    if settings.template_window % 2 != 1 or settings.template_polyorder >= settings.template_window:
        return 'invalid_settings_template'
    if not np.isfinite(settings.min_improvement_fraction) or not 0 < settings.min_improvement_fraction < 1:
        return 'invalid_settings_improvement'
    if not 0 < settings.gain_bounds[0] < 1 < settings.gain_bounds[1]:
        return 'invalid_settings_gain_bounds'
    regions = fit_regions(settings)
    for number, region in enumerate(regions):
        if max(region[0], settings.validation_region[0]) <= min(region[1], settings.validation_region[1]):
            return 'training_and_validation_overlap'
        if any(max(region[0], other[0]) <= min(region[1], other[1]) for other in regions[:number]):
            return 'training_bands_overlap'
    for lo, hi in (*regions, settings.validation_region):
        if lo < settings.core_region[0] or hi > settings.core_region[1]:
            return 'scoring_band_outside_core'
    return None


def _raw_fit(data, background, train_ids, gate_ids, energy, train_mask, support, settings, method, cancellation):
    """Optimize BN-projected actual production D2 using complete train repeats only."""
    _check(cancellation)
    index = np.arange(len(energy), dtype=float)
    log_b = np.log(background)
    fine = log_b-savgol_filter(log_b, settings.template_window, settings.template_polyorder)
    spline = CubicSpline(index, fine)
    trend = _detrender(energy, train_mask, fit_regions(settings))
    base_t = fine[train_mask]
    texture_rms = _rms(base_t-trend@np.linalg.lstsq(trend, base_t, rcond=None)[0])
    if texture_rms < 1e-9:
        return background.copy(), dict(shift_pixel=0., fine_gain=1., training_log_rms=0.,
                                       near_parameter_bound=False, optimizer_success=False,
                                       reason='no_resolvable_bn_texture')

    return fit_d2_priority(data, background, train_ids, gate_ids, energy, train_mask,
        support, fine, spline, settings, method, lambda: _check(cancellation))


def _shrink_fit(background, full, parameters, family, alpha):
    """Fixed post-fit log correction, never an additional fitting parameter."""
    log_ratio = np.log(full/background)
    candidate = full.copy() if alpha == 1 else background*np.exp(alpha*log_ratio)
    params = deepcopy(parameters)
    params.update(family=family, correction_strength=alpha,
        unshrunk_parameters={name: parameters[name] for name in
            ('shift_pixel', 'fine_gain', 'near_parameter_bound', 'optimizer_success')},
        unshrunk_background=dict(sha256=hashlib.sha256(np.asarray(full, dtype='<f8').tobytes()).hexdigest(),
            point_count=len(full), log_correction_rms=_rms(log_ratio),
            max_absolute_log_correction=float(np.max(abs(log_ratio)))),
        shrinkage=dict(post_fit=True, alpha_is_fitted=False,
            formula='B_alpha = B_original * exp(alpha * log(B_full / B_original))',
            training_objective_background='unshrunk fitted BN'))
    return candidate, params


def _fit(data, background, train_ids, gate_ids, energy, train_mask, support, settings, method, cancellation):
    family, alpha = _METHOD_SPECS[method]
    full, params = _raw_fit(data, background, train_ids, gate_ids, energy, train_mask,
                          support, settings, family, cancellation)
    return _shrink_fit(background, full, params, family, alpha)


def _scores(data, background, repeat_ids, gate_ids, energy, mask):
    trend = _polynomial(energy, mask)
    channels = np.flatnonzero(mask)
    profiles = np.median(data[np.ix_(repeat_ids, gate_ids, channels)]/background[channels]-1, axis=1)
    residual = profiles-(trend@np.linalg.lstsq(trend, profiles.T, rcond=None)[0]).T
    if not np.isfinite(residual).all():
        raise FloatingPointError('Nonfinite DR/R residual')
    return np.sqrt(np.mean(residual*residual, axis=1))


def _comparison(original, candidate, ids):
    improvement = (original-candidate)/np.maximum(original, 1e-12)
    return dict(repeat_indices=[int(i) for i in ids], original_rms=original.tolist(),
                candidate_rms=candidate.tolist(), original_rms_percent=(100*original).tolist(),
                candidate_rms_percent=(100*candidate).tolist(),
                mean_original_rms=float(np.mean(original)), mean_candidate_rms=float(np.mean(candidate)),
                mean_improvement_fraction=float((np.mean(original)-np.mean(candidate))/max(np.mean(original), 1e-12)),
                per_repeat_improvement_fraction=improvement.tolist())


def _passes(comparison, settings, each=False):
    if comparison['mean_improvement_fraction'] < settings.min_improvement_fraction:
        return False
    if comparison['mean_original_rms']-comparison['mean_candidate_rms'] < settings.min_absolute_improvement:
        return False
    if each:
        if min(comparison['per_repeat_improvement_fraction']) < settings.min_improvement_fraction:
            return False
        if min(np.asarray(comparison['original_rms'])-comparison['candidate_rms']) < settings.min_absolute_improvement:
            return False
    return True


def _passes_d2(comparison, settings, each=False):
    """The derivative has its own unit-aware absolute acceptance threshold."""
    from dataclasses import replace
    return _passes(comparison, replace(settings, min_absolute_improvement=settings.min_absolute_d2_improvement), each)


def _noninferior(comparison, settings):
    before = np.asarray(comparison['original_rms'])
    after = np.asarray(comparison['candidate_rms'])
    allowance = np.maximum(settings.drr_noninferiority_absolute,
                           settings.drr_noninferiority_fraction*before)
    return bool(np.all(after-before <= allowance))


def _joint_evidence(original, candidate, energy, score_mask, support, ids, gates, settings, drr):
    comparison_d2 = _comparison(repeat_scores(original[1], energy, score_mask, gates),
                                repeat_scores(candidate[1], energy, score_mask, gates), ids)
    checks = joint_checks(original, candidate, energy, score_mask, support, ids, settings)
    checks['drr_benefit_passed'] = _passes(drr, settings, each=True)
    checks['drr_noninferiority_passed'] = bool(_noninferior(drr, settings)
        and checks['all_gate_drr']['passed'] and checks['protection']['local_drr']['passed']
        and checks['protection']['drr']['passed'] and checks['protection']['outside_support_drr']['passed'])
    checks['d2_benefit_passed'] = _passes_d2(comparison_d2, settings, each=True)
    if not checks['drr_noninferiority_passed']:
        checks['reasons'].append('drr_noninferiority_not_confirmed')
    if not checks['d2_benefit_passed']:
        checks['reasons'].append('d2_benefit_not_confirmed')
    checks['passed'] = not checks['reasons']
    checks['ranking_objective'] = comparison_d2['mean_candidate_rms']/max(comparison_d2['mean_original_rms'], 1e-12)
    return comparison_d2, checks


def _select_candidate(items):
    """Frozen deterministic rule; weaker/simpler candidates win a 0.01 tie."""
    selected, best = 'original', 1.
    for method in _METHODS[1:]:
        item = items[method]
        score = item['selection_joint']['ranking_objective']
        if (item['parameters']['optimizer_success'] and item['selection_joint']['passed']
                and item['training_influence_audit']['passed'] and score < best-.01):
            selected, best = method, score
    return selected


def _candidate_stage(z, b, train, selection_ids, gates, e, fit_mask, score_mask, support, settings, cancellation):
    """All candidate qualification/selection; this function receives no final IDs."""
    candidates = {'original': b.copy()}
    original_scores = _scores(z, b, selection_ids, gates, e, score_mask)
    original_products = products(e, z, b, selection_ids, settings)
    original_comparison = _comparison(original_scores, original_scores, selection_ids)
    d2, joint = _joint_evidence(original_products, original_products, e, score_mask,
                              support, selection_ids, gates, settings, original_comparison)
    items = {'original': dict(parameters={}, selection=original_comparison,
                             selection_d2=d2, selection_joint=joint)}
    train_q = _detrender(e, fit_mask, fit_regions(settings))

    def training_rms(values):
        band = values[..., fit_mask]
        band -= (band@train_q)@train_q.T
        return np.sqrt(np.mean(band*band, axis=-1))

    original_train = products(e, z, b, train, settings)
    original_train_drr = training_rms(original_train[0])[:, gates]
    original_train_d2 = training_rms(original_train[1])[:, gates]
    full_fits = {}
    refit_cache = {}
    forcing_cache = {}

    def refit(injected, family, alpha):
        # Cache only within this exact data-stage call. A perturbation always
        # reruns the raw fit; different alpha values share that same estimator.
        key = (family, hashlib.sha256(np.asarray(injected, dtype='<f8').tobytes()).digest())
        if key not in refit_cache:
            refit_cache[key] = _fit(injected, b, list(range(len(train))), gates, e,
                                   fit_mask, support, settings, family, cancellation)
        full, raw = refit_cache[key]
        return _shrink_fit(b, full, raw, family, alpha)
    for family in _FAMILIES:
        _check(cancellation)
        full_fits[family] = _fit(z, b, train, gates, e, fit_mask, support, settings, family, cancellation)
    for method, (family, alpha) in _METHOD_SPECS.items():
        full, raw_params = full_fits[family]
        candidate, params = _shrink_fit(b, full, raw_params, family, alpha)
        if not np.isfinite(candidate).all() or np.any(candidate <= 0):
            raise FloatingPointError('Nonfinite shrunk BN candidate')
        candidates[method] = candidate
        scores = _scores(z, candidate, selection_ids, gates, e, score_mask)
        comparison = _comparison(original_scores, scores, selection_ids)
        candidate_products = products(e, z, candidate, selection_ids, settings)
        d2, joint = _joint_evidence(original_products, candidate_products, e, score_mask,
                                   support, selection_ids, gates, settings, comparison)
        applied_train = products(e, z, candidate, train, settings)
        applied_drr = training_rms(applied_train[0])[:, gates]
        applied_d2 = training_rms(applied_train[1])[:, gates]
        allowance = np.maximum(settings.drr_noninferiority_absolute,
                               settings.drr_noninferiority_fraction*original_train_drr)
        params['applied_training_metrics'] = dict(
            interpretation='Actual post-shrinkage per-repeat/gate independent-band residual RMS; D2 is a display diagnostic, not the BN-projected fit objective.',
            original_drr_rms=original_train_drr.tolist(), candidate_drr_rms=applied_drr.tolist(),
            original_d2_rms=original_train_d2.tolist(), candidate_d2_rms=applied_d2.tolist(),
            drr_noninferiority_passed=bool(np.all(applied_drr-original_train_drr <= allowance)),
            repeat_indices=list(train), gate_indices=[int(i) for i in gates],
            used_window_length=applied_train[2], full_training_repeat_context=True)
        joint['applied_training_drr_noninferiority_passed'] = params['applied_training_metrics']['drr_noninferiority_passed']
        if not joint['applied_training_drr_noninferiority_passed']:
            joint['passed'] = False
            joint['reasons'].append('applied_training_drr_noninferiority_failed')
        objective = params.get('training_objective', {})
        bandwidth = objective.get('bandwidth_check', {})
        joint['training_bandwidth_passed'] = bandwidth.get('passed') is True
        if not joint['training_bandwidth_passed']:
            joint['passed'] = False
            joint['reasons'].append('insufficient_d2_training_bandwidth')
        audit = dict(schema_version=3, policy=INFLUENCE_POLICY,
            passed=False, training_only=True, selected_method=method, family=family,
            correction_strength=alpha, evaluated=False, reasons=['candidate_not_joint_qualified'])
        if params['optimizer_success'] and joint['passed'] and objective.get('d2_training_evaluated'):
            audit = audit_training_influence(z, b, train, gates, e, fit_mask, support,
                settings, method, candidate, params, lambda: _check(cancellation),
                lambda injected, frozen_family=family, frozen_alpha=alpha:
                    refit(injected, frozen_family, frozen_alpha),
                unshrunk_candidate=full, forcing_cache=forcing_cache)
            audit['evaluated'] = True
        joint['training_feature_influence_passed'] = audit['passed']
        items[method] = dict(parameters=params, selection=comparison, selection_d2=d2,
                            selection_joint=joint, training_influence_audit=audit)
    return candidates, items, _select_candidate(items), original_products[2]


def _selection_sensitivity(z, b, train, selection_ids, gates, e, fit_mask, score_mask,
                           support, settings, cancellation, selected, candidates, items):
    """Three predeclared high-risk probes rerun the entire pre-final selector."""
    from core.drr_bn_joint import production_d2
    influence = items[selected]['training_influence_audit']
    chosen = ranked_influence_probes(influence)
    eligible = lambda records: [name for name in _METHODS[1:] if records[name]['parameters']['optimizer_success']
        and records[name]['selection_joint']['passed'] and records[name]['training_influence_audit']['passed']]
    audit = dict(schema_version=2, policy='pre_final_selection_sensitivity_v2', passed=False,
        selection_aware=True, pre_final=True, final_repeats_read=False, training_only=False,
        train_repeat_indices=list(train), selection_repeat_indices=list(selection_ids),
        baseline_selected_method=selected, baseline_family=items[selected]['parameters']['family'],
        baseline_correction_strength=items[selected]['parameters']['correction_strength'],
        baseline_qualified_candidates=eligible(items), amplitude=2e-4,
        max_relative_error=.05, count=0, probes=[],
        probe_selection_rule='Worst fixed-candidate DR/R probe, worst D2 probe, then descending joint risk across coarse and refined grids; unique source/index references, deterministic ties.',
        chosen_probe_indices=[probe['probe_index'] for probe in chosen],
        chosen_probe_references=[dict(source=probe['probe_source'], probe_index=probe['probe_index']) for probe in chosen],
        qualification='Finite representative audit of three declared Gaussian training probes and the full pre-final selector; not a guarantee across all selection boundaries or unknown features.',
        rerun='Raw training injection; rerun both raw family fits, all fixed strengths, joint screening, feature qualification, and frozen selection rule; selection spectra unchanged; final never read.',
        reasons=[])
    # The rerun data physically excludes final spectra, including reordered IDs.
    pre_ids = list(dict.fromkeys([*train, *selection_ids]))
    local_train = [pre_ids.index(i) for i in train]
    local_selection = [pre_ids.index(i) for i in selection_ids]
    training = np.asarray(z[train], float)
    for probe in chosen:
        _check(cancellation)
        feature = probe['sign']*audit['amplitude']*np.exp(-.5*((e-probe['center_energy'])/probe['sigma_energy'])**2)
        perturbed = np.asarray(z[pre_ids], float).copy()
        perturbed[local_train] *= np.exp(feature)
        alternatives, records, winner, _ = _candidate_stage(perturbed, b, local_train, local_selection,
            gates, e, fit_mask, score_mask, support, settings, cancellation)
        injected = perturbed[local_train]
        known = (injected-training)/candidates[selected]
        change = injected*(1/alternatives[winner]-1/candidates[selected])
        known_d2, used = production_d2(e, known, settings)
        change_d2, _ = production_d2(e, change, settings)
        dr = np.max(abs(change), axis=-1)/np.maximum(np.max(abs(known), axis=-1), 1e-15)
        d2 = np.max(abs(change_d2), axis=-1)/np.maximum(np.max(abs(known_d2), axis=-1), 1e-15)
        params = records[winner]['parameters']
        audit['probes'].append(dict(probe_index=probe['probe_index'], probe_source=probe['probe_source'], center_energy=probe['center_energy'],
            sigma_energy=probe['sigma_energy'], sign=probe['sign'],
            ranking_drr_error_fraction=probe['drr_error_fraction'], ranking_d2_error_fraction=probe['d2_error_fraction'],
            baseline_selected_method=selected, perturbed_selected_method=winner,
            perturbed_family=params.get('family', 'original'), perturbed_correction_strength=params.get('correction_strength', 0.),
            qualified_candidates=eligible(records), used_window_length=used,
            original_fallback_jump_included=winner == 'original',
            max_absolute_drr_output_change=float(np.max(abs(change))),
            max_absolute_d2_output_change=float(np.max(abs(change_d2))),
            drr_error_fraction=float(np.max(dr)), d2_error_fraction=float(np.max(d2)),
            per_row_drr_error_fraction=dr.tolist(), per_row_d2_error_fraction=d2.tolist(),
            passed=bool(np.max(dr) <= .05 and np.max(d2) <= .05)))
    audit.update(count=len(audit['probes']), passed=all(x['passed'] for x in audit['probes']),
        max_drr_error_fraction=max(x['drr_error_fraction'] for x in audit['probes']),
        max_d2_error_fraction=max(x['d2_error_fraction'] for x in audit['probes']))
    if not audit['passed']:
        audit['reasons'].append('pre_final_selection_sensitivity_failed')
    return audit


def _evaluate_bn_group(energy, measurements, background, *, gates, settings=None,
                       repeat_order=None, background_repeats=None, cancellation=None):
    """Compare fixed local BN candidates without changing input or application state.

    ``measurements`` is [repeat, gate, energy]; a [gate, energy] single repeat is
    accepted for exploration. ``repeat_order`` is an oldest-first permutation
    of repeat indices (not timestamps). The caller establishes chronological
    order. ``background_repeats`` contains the positive raw reference spectra
    whose arithmetic mean is ``background``; without those spectra reference
    stability cannot be verified and automatic application is blocked. Two
    independently computed half-reference means are also accepted when their
    mean equals the reference; duplicated copies are not independent evidence.

    Explicit trusted regions must cover the entire nonzero correction support,
    including tapers. They do not silently override uncertain regions. A caller
    recording a user's trust decision should subtract that range from its
    uncertainty list. Protected regions always win and are never corrected.
    """
    _check(cancellation)
    settings = settings or BnCalibrationSettings()
    b = np.asarray(background, dtype=float).copy()
    candidates = {'original': b.copy()}
    report = dict(schema_version=4, optimization_policy=POLICY, selected_method='original', applied_method='original',
                  settings=asdict(settings), candidates={}, split={},
                  derivative_settings=dict(window_length=settings.derivative_window,
                      polyorder=settings.derivative_polyorder, used_window_length=None,
                      processing='production SG energy derivative'),
                  joint_validation=dict(schema_version=3, policy=POLICY, passed=False, derivative_checked=False,
                      d2_training_evaluated=False, fixed_bn_basis=False,
                      selection=None, final=None, protection=None, influence_audit=None, selection_sensitivity_audit=None,
                      thresholds_are='Configurable operational guardrails; no statistical confidence or inferred noise floor.'),
                  ground_truth_known=False,
                  metric='Per-repeat median selected-gate DR/R, quadratic detrended independently inside the scoring band; RMS dimensionless.',
                  metric_d2='Per-repeat median of selected-gate quadratic-detrended production second energy derivative RMS, (DR/R)/eV^2. All gates additionally pass independent local deterioration and protection checks.',
                  candidate_generation='Two constant shift / shift-gain raw fits using full training-repeat SG context, followed by fixed post-fit alpha in [0.25, 0.5, 0.75, 1]; no free alpha parameter.',
                  selection_rule='Only joint- and training-influence-qualified candidates compete on D2 display-residual fraction; deterministic family/strength order prefers weaker and simpler candidates within 0.01.',
                  candidate_order=list(_METHODS[1:]),
                  noise_floor_estimation='Unavailable; DR/R noninferiority margins are operational allowances, not an inferred noise floor.',
                  thresholds_are='Conservative operational thresholds, not physical/statistical confidence.',
                  final_holdout_use='Confirm or veto frozen selection only; never choose another candidate or refit output.')

    def finish(status, reasons, selected='original', eligible=False):
        joint = report['joint_validation']
        eligible = bool(eligible and joint['passed'] and joint['derivative_checked']
                        and joint['d2_training_evaluated'] and joint['fixed_bn_basis'])
        applied = selected if eligible else 'original'
        report.update(status=status, reasons=list(dict.fromkeys(reasons)), selected_method=selected,
                      applied_method=applied, eligible=eligible)
        return BnCalibrationResult(b.copy(), {k: v.copy() for k, v in candidates.items()}, selected, applied,
                                   candidates[applied].copy(), eligible, status, tuple(report['reasons']), _json_safe(report))

    try:
        problem = _settings_error(settings)
        if problem:
            return finish('invalid', [problem])
        e = np.asarray(energy, dtype=float)
        z = np.asarray(measurements, dtype=float)
        g = np.asarray(gates, dtype=float)
        if z.ndim == 2:
            z = z[None, ...]
        if e.ndim != 1 or b.shape != e.shape or g.ndim != 1 or z.ndim != 3 or z.shape[1:] != (len(g), len(e)) or min(z.shape) < 1:
            return finish('invalid', ['invalid_input_shape'])
        if not all(np.isfinite(v).all() for v in (e, b, g, z)):
            return finish('invalid', ['nonfinite_input'])
        if np.any(b <= 0) or np.any(z <= 0):
            return finish('invalid', ['nonpositive_intensity'])
        if not (np.all(np.diff(e) > 0) or np.all(np.diff(e) < 0)) or len(np.unique(g)) != len(g):
            return finish('invalid', ['invalid_grid'])
        if len(e) < settings.template_window:
            return finish('invalid', ['insufficient_template_bandwidth'])
        lo, hi = settings.core_region
        if e.min() > lo-settings.taper_width or e.max() < hi+settings.taper_width:
            return finish('invalid', ['correction_support_outside_grid'])
        count = len(z)
        order = list(range(count)) if repeat_order is None else list(repeat_order)
        if any(not isinstance(v, (int, np.integer)) for v in order) or sorted(order) != list(range(count)):
            return finish('invalid', ['invalid_repeat_order'])
        order = [int(v) for v in order]
        if count >= 5:
            train, selection, final = order[:2], order[2:-2], order[-2:]
        elif count == 4:
            train, selection, final = order[:1], order[1:2], order[2:]
        elif count >= 2:
            train, selection, final = order[:1], order[1:], []
        else:
            train, selection, final = order, [], []
        report['split'] = dict(train=train, selection=selection, final=final, chronological_order=order,
                               final_repeat_count=len(final), minimum_required_repeats=settings.min_repeats,
                               independent_final_confirmation_available=count >= settings.min_repeats and len(final) >= 2)
        available = np.arange(len(g))
        if settings.gate_range is not None:
            available = available[(g >= settings.gate_range[0]) & (g <= settings.gate_range[1])]
        if not len(available):
            return finish('invalid', ['no_gates_in_range'])
        wanted = np.linspace(g[available].min(), g[available].max(), min(settings.gate_count, len(available)))
        gate_ids = np.unique([available[np.argmin(abs(g[available]-v))] for v in wanted])
        support = _support(e, settings)
        active = support > 0
        fit_mask = _mask(e, fit_regions(settings)) & (support >= 1-1e-12)
        score_mask = _mask(e, [settings.validation_region]) & (support >= 1-1e-12)
        if (score_mask.sum() < settings.min_band_points or any(
                np.sum(fit_mask & _mask(e, [region])) < settings.min_band_points for region in fit_regions(settings))):
            return finish('invalid', ['insufficient_unprotected_bandwidth'])
        report['selected_gate_indices'] = gate_ids.tolist()
        report['selected_gate_values'] = g[gate_ids].tolist()
        report['scope'] = dict(core_region=list(settings.core_region), taper_width=settings.taper_width,
                               active_energy_min=float(e[active].min()), active_energy_max=float(e[active].max()),
                               unchanged_outside_support=True, protected_values_unchanged=True,
                               native_protection_guard_pixels=settings.protection_guard_pixels,
                               downstream_derivative_check_required=False,
                               derivative_check_is_specific_to_recorded_settings=True,
                               fit_channel_indices=np.flatnonzero(fit_mask).tolist(),
                               fit_regions=[list(region) for region in fit_regions(settings)],
                               explicit_multiple_fit_bands=bool(settings.train_regions),
                               fit_band_detrending='Independent quadratic per declared fit band; complete training spectra supply SG/regridding context.',
                               validation_channel_indices=np.flatnonzero(score_mask).tolist())
    except (TypeError, ValueError, IndexError):
        return finish('invalid', ['invalid_input_or_settings'])

    # No final repeat is read by this fitting/selection stage.
    selection_ids = selection or train
    report['selection_evidence'] = 'independent_repeats' if selection else 'training_only_exploration'
    candidates, items, selected, used = _candidate_stage(z, b, train, selection_ids, gate_ids,
        e, fit_mask, score_mask, support, settings, cancellation)
    report['candidates'] = items
    report['derivative_settings']['used_window_length'] = used
    report['joint_validation']['derivative_checked'] = True
    report['selection_frozen'] = True
    report['frozen_selection'] = dict(method=selected, final_may_reselect=False,
                                    final_may_refit=False, stage='pre_final')
    if selected == 'original':
        status = 'exploratory' if count < settings.min_repeats else 'original'
        reasons = ['insufficient_independent_repeats'] if status == 'exploratory' else []
        report['joint_validation']['selection'] = dict(passed=False,
            reasons=['no_reliable_joint_selection_benefit'], candidates={
                method: report['candidates'][method]['selection_joint'] for method in _METHODS[1:]})
        if all('insufficient_d2_training_bandwidth' in report['candidates'][method]['selection_joint']['reasons']
               for method in _METHODS[1:]):
            reasons.append('insufficient_d2_training_bandwidth')
        if any(item['training_influence_audit'].get('evaluated') and not item['training_influence_audit']['passed']
               for name, item in items.items() if name != 'original'):
            reasons.append('training_feature_influence_not_confirmed')
        if any(item['parameters']['near_parameter_bound'] for name, item in items.items() if name != 'original'):
            reasons.append('selected_parameter_at_bound')
        return finish(status, reasons+['no_reliable_selection_benefit', 'no_reliable_joint_selection_benefit'])

    reasons = []
    params = report['candidates'][selected]['parameters']
    objective = params.get('training_objective', {})
    report['joint_validation']['d2_training_evaluated'] = objective.get('d2_training_evaluated') is True
    report['joint_validation']['fixed_bn_basis'] = objective.get('fixed_bn_basis') is True
    if not (report['joint_validation']['d2_training_evaluated'] and report['joint_validation']['fixed_bn_basis']):
        reasons.append('d2_training_not_verified')
    report['joint_validation']['selection'] = report['candidates'][selected]['selection_joint']
    report['joint_validation']['derivative_checked'] = True
    if params['near_parameter_bound']:
        reasons.append('selected_parameter_at_bound')
    if count < settings.min_repeats or len(final) < 2:
        reasons.append('insufficient_independent_repeats')
    if not np.all(_mask(e, settings.trusted_regions)[active]):
        reasons.append('correction_support_not_explicitly_trusted')
    if np.any(_mask(e, settings.uncertain_regions) & active):
        reasons.append('correction_overlaps_uncertain_region')

    influence = items[selected]['training_influence_audit']
    report['joint_validation']['influence_audit'] = influence
    if not influence['passed']:
        reasons.extend(influence['reasons'])
        reasons.append('training_feature_influence_not_confirmed')

    selection_sensitivity = _selection_sensitivity(z, b, train, selection_ids, gate_ids,
        e, fit_mask, score_mask, support, settings, cancellation, selected, candidates, items)
    report['joint_validation']['selection_sensitivity_audit'] = selection_sensitivity
    if not selection_sensitivity['passed']:
        reasons.extend(selection_sensitivity['reasons'])

    if final:
        old = _scores(z, b, final, gate_ids, e, score_mask)
        new = _scores(z, candidates[selected], final, gate_ids, e, score_mask)
        confirmation = _comparison(old, new, final)
        report['final_confirmation'] = confirmation
        report['candidates'][selected]['final'] = confirmation
        if not _noninferior(confirmation, settings):
            reasons.append('final_holdout_benefit_not_confirmed')
            reasons.append('final_holdout_drr_noninferiority_not_confirmed')
        old_final_products = products(e, z, b, final, settings)
        new_final_products = products(e, z, candidates[selected], final, settings)
        confirmation_d2, final_joint = _joint_evidence(old_final_products, new_final_products,
            e, score_mask, support, final, gate_ids, settings, confirmation)
        report['final_confirmation_d2'] = confirmation_d2
        report['candidates'][selected]['final_d2'] = confirmation_d2
        report['joint_validation']['final'] = final_joint
        if not final_joint['d2_benefit_passed']:
            reasons.append('final_holdout_d2_benefit_not_confirmed')
        if not final_joint['passed']:
            reasons.append('final_holdout_joint_not_confirmed')

    # Fitting repeats also get actual full-axis fidelity checks after the
    # selection is frozen. These diagnostics cannot change the winner.
    old_train_products = products(e, z, b, train, settings)
    new_train_products = products(e, z, candidates[selected], train, settings)
    training_checks = joint_checks(old_train_products, new_train_products, e,
        score_mask, support, train, settings)
    report['joint_validation']['protection'] = dict(
        training=training_checks['protection'],
        selection=report['joint_validation']['selection']['protection'],
        final=(report['joint_validation']['final'] or {}).get('protection'))
    if not training_checks['passed']:
        reasons.append('training_full_axis_fidelity_not_confirmed')

    # Later fits are diagnostics only; they can veto but never replace the
    # already frozen selected parameters/candidate vector.
    temporal = []
    for repeat in order:
        _check(cancellation)
        _, diagnostic = _fit(z, b, [repeat], gate_ids, e, fit_mask, support, settings, selected, cancellation)
        temporal.append(dict(repeat_index=repeat, parameters=diagnostic))
        if (not diagnostic['optimizer_success'] or diagnostic['near_parameter_bound']
                or abs(diagnostic['shift_pixel']-params['shift_pixel']) > settings.max_temporal_shift_difference
                or abs(diagnostic['fine_gain']-params['fine_gain']) > settings.max_temporal_gain_difference):
            reasons.append('temporal_parameters_unstable')
        if (abs(diagnostic['shift_pixel']) > .02 and abs(params['shift_pixel']) > .02
                and diagnostic['shift_pixel']*params['shift_pixel'] < 0):
            reasons.append('temporal_shift_sign_reversal')
    report['temporal_diagnostics'] = temporal

    sensitivity = dict(available=False, halves=[])
    report['background_sensitivity'] = sensitivity
    reference_frames = None
    if background_repeats is not None:
        try:
            raw = np.asarray(background_repeats, dtype=float)
            if raw.ndim == 2 and raw.shape[1] == len(e) and len(raw) >= 2 and np.isfinite(raw).all() and np.all(raw > 0):
                if np.allclose(raw.mean(0), b, rtol=1e-6, atol=1e-10):
                    reference_frames = raw
                else:
                    reasons.append('background_repeat_mean_mismatch')
            else:
                reasons.append('invalid_background_repeats')
        except (ValueError, TypeError):
            reasons.append('invalid_background_repeats')
    if reference_frames is None:
        reasons.append('background_sensitivity_unverified')
    else:
        sensitivity['available'] = True
        middle = len(reference_frames)//2
        for label, half in [('first', reference_frames[:middle]), ('last', reference_frames[middle:])]:
            _check(cancellation)
            half_b = half.mean(0)
            half_candidate, half_params = _fit(z, half_b, train, gate_ids, e, fit_mask, support, settings, selected, cancellation)
            entry = dict(half=label, parameters=half_params, comparisons={}, comparisons_d2={}, joint_checks={})
            sensitivity['halves'].append(entry)
            if (not half_params['optimizer_success'] or half_params['near_parameter_bound']
                    or abs(half_params['shift_pixel']-params['shift_pixel']) > settings.max_bn_shift_difference
                    or abs(half_params['fine_gain']-params['fine_gain']) > settings.max_bn_gain_difference):
                reasons.append('background_parameters_sensitive')
            for role, ids in [('selection', selection), ('final', final)]:
                if not ids:
                    continue
                old = _scores(z, half_b, ids, gate_ids, e, score_mask)
                new = _scores(z, half_candidate, ids, gate_ids, e, score_mask)
                comparison = _comparison(old, new, ids)
                entry['comparisons'][role] = comparison
                if not _noninferior(comparison, settings):
                    reasons.append('background_half_benefit_not_confirmed')
                old_half_products = products(e, z, half_b, ids, settings)
                new_half_products = products(e, z, half_candidate, ids, settings)
                comparison_d2, half_joint = _joint_evidence(old_half_products, new_half_products,
                    e, score_mask, support, ids, gate_ids, settings, comparison)
                entry['comparisons_d2'][role] = comparison_d2
                entry['joint_checks'][role] = half_joint
                if not half_joint['passed']:
                    reasons.append('background_half_joint_not_confirmed')
    _check(cancellation)
    joint = report['joint_validation']
    joint['passed'] = bool(joint['selection']['passed'] and joint['final']
                          and joint['final']['passed'] and training_checks['passed']
                          and influence['passed']
                          and selection_sensitivity['passed']
                          and len(final) >= 2 and count >= settings.min_repeats)
    if not joint['passed']:
        reasons.append('joint_validation_not_confirmed')
    status = 'exploratory' if count < settings.min_repeats else ('fallback' if reasons else 'accepted')
    return finish(status, reasons, selected, eligible=not reasons)


def evaluate_bn_group(energy, measurements, background, *, gates, settings=None,
                      repeat_order=None, background_repeats=None, cancellation=None):
    """Public non-mutating API; see ``_evaluate_bn_group`` for split semantics.

    Invalid data/configuration return an explicit ``invalid`` result. Cooperative
    ``LoadCancelled`` exceptions deliberately propagate to the worker/caller.
    """
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return _evaluate_bn_group(energy, measurements, background, gates=gates, settings=settings,
                                      repeat_order=repeat_order, background_repeats=background_repeats,
                                      cancellation=cancellation)
    except (TypeError, ValueError, IndexError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        try:
            original = np.asarray(background, dtype=float).copy()
        except (TypeError, ValueError):
            original = np.empty(0, dtype=float)
        reason = 'numerical_failure' if isinstance(error, (FloatingPointError, OverflowError, np.linalg.LinAlgError)) else 'invalid_input_or_settings'
        report = dict(schema_version=4, optimization_policy=POLICY, status='invalid', eligible=False, selected_method='original',
                      applied_method='original', reasons=[reason], error_type=type(error).__name__,
                      derivative_settings=dict(window_length=getattr(settings, 'derivative_window', 21),
                          polyorder=getattr(settings, 'derivative_polyorder', 2), used_window_length=None,
                          processing='production SG energy derivative'),
                      joint_validation=dict(schema_version=3, policy=POLICY, passed=False, derivative_checked=False,
                          d2_training_evaluated=False, fixed_bn_basis=False,
                          selection=None, final=None, protection=None, influence_audit=None, selection_sensitivity_audit=None),
                      ground_truth_known=False, split={}, candidates={})
        return BnCalibrationResult(original.copy(), {'original': original.copy()}, 'original', 'original',
                                   original.copy(), False, 'invalid', (reason,), _json_safe(report))
