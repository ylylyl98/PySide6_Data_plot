"""Per-measurement shared BN trends with bounded changes to production D2.

Automatic windows are baseline hypotheses, not identified feature-free regions.
All gate rows share one low-order log gain. No gate-specific background is fitted.
"""
from dataclasses import asdict
import json

import numpy as np
from scipy.signal import savgol_filter

from core.drr_bn_calibration import BnCalibrationResult, _check
from core.drr_bn_joint import production_d2


STRENGTHS = (1., .75, .5, .25, .125)
POLICY = 'shared_bn_trend_v2'


def _strength_search():
    """Bounded search; every proposed strength must pass all guards explicitly."""
    upper = None
    for strength in STRENGTHS:
        if (yield strength):
            lower = strength
            break
        upper = strength
    else:
        return
    if upper is None:
        return
    for _ in range(3):
        middle = (lower+upper)/2
        if (yield middle):
            lower = middle
        else:
            upper = middle


def _automatic_regions(energy, log_rows, gates):
    """Screen fixed candidate windows without optimizing the fit residual."""
    order = np.argsort(gates, kind='stable')
    screening = order[::3] if len(gates) >= 6 else order
    rows = log_rows[screening]
    median = np.median(rows, axis=0)
    smooth = savgol_filter(median, min(21, (len(energy)-1)//2*2+1), 2)
    curvature = np.abs(np.gradient(np.gradient(smooth, energy), energy))
    spread = np.percentile(rows, 75, axis=0)-np.percentile(rows, 25, axis=0)
    span = float(np.ptp(energy))
    outer = ((energy <= energy[0]+.35*span) | (energy >= energy[-1]-.35*span))
    cscale = max(float(np.median(curvature[outer])), 1e-8)
    sscale = max(float(np.median(spread[outer])), 1e-6)
    candidates, selected = [], []
    for side in ('left', 'right'):
        options = []
        for fraction in (.10, .15, .20):
            width = max(fraction*span, 32*float(np.median(np.diff(energy))))
            if width > .35*span:
                continue
            lower = energy[0] if side == 'left' else energy[-1]-.35*span
            for offset in np.linspace(0., .35*span-width, 5):
                lo, hi = float(lower+offset), float(lower+offset+width)
                mask = (energy >= lo) & (energy <= hi)
                if np.count_nonzero(mask) < 24:
                    continue
                rough = float(np.median(curvature[mask])/cscale)
                varying = float(np.median(spread[mask])/sscale)
                score = rough+varying+.05*span/width
                item = dict(side=side, interval=[lo, hi], score=score,
                            curvature_score=rough, gate_spread_score=varying,
                            point_count=int(mask.sum()))
                options.append((score, -width, lo, len(candidates)))
                candidates.append(item)
        if not options:
            raise ValueError('insufficient_automatic_background_support')
        selected.append(min(options)[-1])
    regions = [tuple(candidates[index]['interval']) for index in selected]
    return regions, dict(policy='outer_windows_shape_and_gate_spread_v1', outer_fraction=.35,
                         screening_gate_indices=screening.tolist(), candidates=candidates,
                         selected_indices=selected, interpretation='baseline assumptions, not feature-free certificates')


def _gate_models(x, log_rows, gates, mask, degree, cancellation):
    from core.drr_bn_trend import _fit
    groups = np.array_split(np.argsort(gates, kind='stable'), min(3, len(gates)))
    return [dict(gate_indices=group.tolist(), coefficients=_fit(
        x, np.median(log_rows[group], axis=0), mask, degree, cancellation).tolist())
        for group in groups if len(group)]


def _relative_curves(x, coefficients, alternatives, strength):
    from core.drr_bn_trend import _model
    reference = _model(x, coefficients)
    return [float(np.max(np.abs(np.expm1(np.clip(strength*(
        _model(x, item['coefficients'])-reference), -50, 50))))) for item in alternatives]


def _local_rms(values, starts, width):
    cumulative = np.pad(np.cumsum(np.square(values), axis=-1), ((0, 0), (1, 0)))
    return np.sqrt(np.maximum(0., cumulative[:, starts+width]-cumulative[:, starts])/width)


def _d2_reference(energy, raw, baseline, settings):
    original, used = production_d2(energy, raw/baseline-1, settings)
    width = min(len(energy), max(9, settings.local_validation_window))
    starts = np.unique(np.r_[np.arange(0, len(energy)-width+1, max(1, width//2)), len(energy)-width])
    floor = max(1., float(np.percentile(np.abs(original), 95))*.01)
    rms = _local_rms(original, starts, width)
    denominator = np.maximum(rms, floor)
    return original, starts, width, denominator, floor, used, float(np.mean(rms < floor))


def _d2_guard(energy, raw, candidate, settings, reference):
    original, starts, width, denominator, floor, used, floor_fraction = reference
    changed, _ = production_d2(energy, raw/candidate-1, settings)
    delta = changed-original
    relative = _local_rms(delta, starts, width)/denominator
    maximum = float(np.max(relative))
    worst = np.unravel_index(np.argmax(relative), relative.shape)
    return dict(passed=maximum <= settings.single_trend_max_d2_change,
                max_relative_change=maximum, limit=settings.single_trend_max_d2_change,
                max_absolute_change=float(np.max(np.abs(delta))), floor=floor,
                floor_dominated_fraction=floor_fraction,
                energy_points=len(energy),
                floor_policy='max(1 eV^-2, 1% of full original |D2| 95th percentile)',
                windows=len(starts), window_points=width, gate_count=len(raw),
                worst_gate_index=int(worst[0]), worst_energy_interval=[
                    float(energy[starts[worst[1]]]), float(energy[starts[worst[1]]+width-1])],
                derivative_window=used, derivative_polyorder=settings.derivative_polyorder,
                metric='local RMS of D2 change / max(original local RMS, numerical floor)',
                interpretation='change guard, not fringe reduction or independent validation')


def _cv_reference(energy, x, target, regions, degree, cancellation):
    from core.drr_bn_trend import _fit, _masks, _model
    masks, _ = _masks(energy, regions)
    halves = [[], []]
    for mask in masks:
        a, b = np.array_split(np.flatnonzero(mask), 2)
        halves[0].extend(a)
        halves[1].extend(b)
    result = []
    for direction in (0, 1):
        train = np.zeros(len(energy), bool)
        train[halves[direction]] = True
        test = np.asarray(halves[1-direction])
        coefficients = _fit(x, target, train, degree, cancellation)
        result.append((target[test], _model(x, coefficients)[test]))
    return result


def _scaled_cv(reference, strength):
    return [dict(original_rms=float(np.sqrt(np.mean(target**2))),
                 candidate_rms=float(np.sqrt(np.mean((target-strength*predicted)**2))),
                 test_count=len(target)) for target, predicted in reference]


def evaluate_adaptive_trend(energy, measurements, background, *, gates, settings,
                            background_repeats=None, cancellation=None):
    from core.drr_bn_trend import _evaluate_degree, _masks, _model
    config = settings
    _check(cancellation)
    e, z, b, g = (np.asarray(value, float) for value in (energy, measurements, background, gates))
    report = dict(schema_version=2, optimization_policy=POLICY, measurement_count=1,
        independent_validation=False, shared_across_gates=True, candidate_generated=False,
        settings=asdict(config), joint_validation={'passed': False}, models={},
        single_pair_scope=dict(mode='full_range', model='shared_trend_normalization'),
        limitation='Data-selected baseline assumptions and D2 change guards are not physical validation.')
    arrays = {'original': b.copy()}

    def finish(selected='original', eligible=False, reasons=()):
        applied = selected if eligible else 'original'
        report.update(selected_method=selected, selected_candidate=selected, reasons=list(reasons))
        return BnCalibrationResult(b.copy(), arrays, selected, applied, arrays[applied].copy(),
            bool(eligible), 'single_pair' if eligible else 'preview_only', tuple(reasons), report)

    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if (not config.single_measurement or not config.single_trend_enabled or not config.single_trend_adaptive
            or e.ndim != 1 or b.shape != e.shape or g.ndim != 1 or z.shape != (len(g), len(e))
            or len(g) < 1 or len(e) < 60 or not all(np.isfinite(a).all() for a in (e, z, b, g))
            or np.any(z <= 0) or np.any(b <= 0)
            or not (np.all(np.diff(e) > 0) or np.all(np.diff(e) < 0))):
        return finish(reasons=['invalid_input_or_scope'])
    if (config.single_trend_order not in ('auto', 'linear', 'quadratic')
            or not all(np.isfinite(v) and 0 < v < 1 for v in
                       (config.single_trend_max_sensitivity, config.single_trend_max_d2_change))):
        return finish(reasons=['invalid_settings'])
    reverse = e[0] > e[-1]
    ee, zz, bb = (e[::-1], z[:, ::-1], b[::-1]) if reverse else (e, z, b)
    try:
        if background_repeats is not None:
            repeats = np.asarray(background_repeats, float)
            if repeats.ndim != 2 or repeats.shape[1] != len(e) or not np.isfinite(repeats).all() or np.any(repeats <= 0):
                raise ValueError('invalid_background_repeats')
            report['background_repeats'] = dict(count=len(repeats), used_for_fit=False)
        log_rows = np.log(zz)-np.log(bb)
        if config.single_trend_regions:
            regions = sorted(tuple(map(float, pair)) for pair in config.single_trend_regions)
            selection, source = dict(policy='user_supplied_background_assumption'), 'user'
        else:
            regions, selection = _automatic_regions(ee, log_rows, g)
            source = 'adaptive_candidates'
        if (any(len(pair) != 2 or not np.isfinite(pair).all() or pair[0] < ee[0] or pair[1] > ee[-1]
                or pair[0] >= pair[1] for pair in regions)
                or any(a[1] >= c[0] for a, c in zip(regions, regions[1:]))):
            raise ValueError('invalid_background_regions')
        _, mask = _masks(ee, regions)
        center, scale = float((ee[0]+ee[-1])/2), float(np.ptp(ee)/2)
        x = (ee-center)/scale
        target = np.log(np.median(zz, axis=0))-np.log(bb)
        report.update(coordinate=dict(center_ev=center, scale_ev=scale),
            background_regions=[list(pair) for pair in regions], background_regions_ev=[list(pair) for pair in regions],
            background_regions_source=source, automatic_background=selection,
            cv_method='contiguous half-window holdouts after same-scan window screening; not independent validation',
            candidates={'original': {'parameters': {}}})
        reference = _d2_reference(ee, zz, bb, config)
        for degree, name in ((1, 'trend_linear'), (2, 'trend_quadratic')):
            _check(cancellation)
            base = _evaluate_degree(ee, x, target, regions, degree, config.single_trend_max_sensitivity, cancellation)
            coefficients = np.asarray(base['coefficients'])
            gate_models = _gate_models(x, log_rows, g, mask, degree, cancellation)
            cv_reference = _cv_reference(ee, x, target, regions, degree, cancellation)
            attempts, best, best_index = [], None, None
            search = _strength_search()
            strength = next(search)
            while True:
                _check(cancellation)
                window_changes = _relative_curves(x, coefficients, base['sensitivity_variants'], strength)
                gate_changes = _relative_curves(x, coefficients, gate_models, strength)
                candidate = bb*np.exp(strength*_model(x, coefficients))
                if not np.isfinite(candidate).all() or np.any(candidate <= 0):
                    raise ValueError('invalid_candidate')
                guard = _d2_guard(ee, zz, candidate, config, reference)
                cv = _scaled_cv(cv_reference, strength)
                original_rms = float(np.sqrt(np.mean([v['original_rms']**2 for v in cv])))
                candidate_rms = float(np.sqrt(np.mean([v['candidate_rms']**2 for v in cv])))
                sensitivity, gate_sensitivity = max(window_changes), max(gate_changes)
                accepted = bool(guard['passed'] and sensitivity <= config.single_trend_max_sensitivity
                    and gate_sensitivity <= config.single_trend_max_sensitivity
                    and original_rms > 1e-12 and candidate_rms < original_rms)
                attempts.append(dict(strength=strength, passed=accepted,
                    max_window_change=sensitivity, max_gate_change=gate_sensitivity,
                    max_d2_change=guard['max_relative_change'], cv_candidate_rms=candidate_rms))
                if accepted and (best is None or strength > best[0]):
                    best = (strength, candidate, guard, cv, original_rms, candidate_rms,
                            sensitivity, gate_sensitivity, window_changes, gate_changes)
                    best_index = len(attempts)-1
                try:
                    strength = search.send(accepted)
                except StopIteration:
                    break
            accepted = best is not None
            if accepted:
                (strength, candidate, guard, cv, original_rms, candidate_rms,
                 sensitivity, gate_sensitivity, window_changes, gate_changes) = best
            final_coefficients = strength*coefficients
            model = dict(degree=degree, coefficients=final_coefficients.tolist(),
                base_coefficients=coefficients.tolist(), strength=strength, eligible=accepted,
                cv_blocks=cv, cv_original_rms=original_rms, cv_candidate_rms=candidate_rms,
                cv_improvement_fraction=(original_rms-candidate_rms)/max(original_rms, 1e-12),
                max_sensitivity=sensitivity, sensitivity_pass=sensitivity <= config.single_trend_max_sensitivity,
                sensitivity_variants=base['sensitivity_variants'], window_changes=window_changes,
                gate_stability=dict(passed=gate_sensitivity <= config.single_trend_max_sensitivity,
                    max_relative_change=gate_sensitivity, groups=gate_models, changes=gate_changes,
                    limit=config.single_trend_max_sensitivity),
                d2_guard=guard, attempts=attempts, selected_attempt=best_index,
                gain_min=float(np.min(candidate/bb)), gain_max=float(np.max(candidate/bb)))
            report['models'][name] = model
            report['candidates'][name] = dict(parameters=dict(
                coefficients=[float(value/scale**i) for i, value in enumerate(final_coefficients)],
                energy_center_ev=center, strength=strength), selection=dict(mean_improvement_fraction=model['cv_improvement_fraction']),
                selection_joint=dict(passed=False, drr_noninferiority_passed=False,
                                     reasons=['within-scan adaptive trend; not joint validation']))
            arrays[name] = candidate[::-1].copy() if reverse else candidate.copy()
        linear, quadratic = (report['models'][name] for name in ('trend_linear', 'trend_quadratic'))
        if config.single_trend_order == 'auto':
            selected = ('trend_quadratic' if quadratic['eligible'] and
                        (not linear['eligible'] or quadratic['cv_candidate_rms'] <= .85*linear['cv_candidate_rms'])
                        else 'trend_linear')
        else:
            selected = 'trend_'+config.single_trend_order
        model = report['models'][selected]
        passed = model['eligible']
        selection_reason = ('explicit_order' if config.single_trend_order != 'auto' else
            'linear_failed_guard' if selected == 'trend_quadratic' and not linear['eligible'] else
            'quadratic_cv_improves_15_percent' if selected == 'trend_quadratic' else 'linear_default')
        report.update(candidate_generated=True,
            model_selection=dict(order_requested=config.single_trend_order, selected=selected,
                                 strength=model['strength'], reason=selection_reason),
            trend_validation=dict(passed=passed, finite_positive=True,
                same_scan_cv_improved=model['cv_candidate_rms'] < model['cv_original_rms'],
                window_sensitivity_pass=model['sensitivity_pass'], gate_stability_pass=model['gate_stability']['passed'],
                d2_change_guard_pass=model['d2_guard']['passed'], independent_validation=False))
        return finish(selected, passed, [] if passed else ['adaptive_trend_checks_failed'])
    except (ValueError, TypeError, IndexError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return finish(reasons=[str(exc)])


def adaptive_validation_passed(report, settings=None):
    """Verify v2 evidence and coefficient/strength consistency, never a joint claim."""
    from core.processing import clamp_sg_window
    try:
        saved = report['settings']
        expected = asdict(settings) if hasattr(settings, '__dataclass_fields__') else dict(settings or saved)
        if json.dumps(saved, sort_keys=True) != json.dumps(expected, sort_keys=True):
            return False
        if (report['schema_version'] != 2 or report['optimization_policy'] != POLICY
                or report['measurement_count'] != 1 or report['independent_validation'] is not False
                or report['shared_across_gates'] is not True or report['candidate_generated'] is not True
                or report['joint_validation']['passed'] is not False
                or not all(saved[key] is True for key in ('single_measurement', 'single_trend_enabled', 'single_trend_adaptive'))
                or report['single_pair_scope'] != dict(mode='full_range', model='shared_trend_normalization')):
            return False
        name = report['selected_method']
        if name not in ('trend_linear', 'trend_quadratic') or report['selected_candidate'] != name:
            return False
        model = report['models'][name]
        degree = 1 if name == 'trend_linear' else 2
        strength = model['strength']
        center, scale = report['coordinate']['center_ev'], report['coordinate']['scale_ev']
        if (not np.isfinite([center, scale, strength]).all() or scale <= 0
                or not .125 <= strength <= 1 or strength*64 != round(strength*64)
                or model['degree'] != degree or model['eligible'] is not True):
            return False
        coefficients, base = np.asarray(model['coefficients']), np.asarray(model['base_coefficients'])
        if (coefficients.shape != (degree+1,) or base.shape != coefficients.shape
                or not np.isfinite(base).all() or not np.allclose(coefficients, strength*base, rtol=1e-12, atol=1e-14)):
            return False
        params = report['candidates'][name]['parameters']
        if (params['strength'] != strength or params['energy_center_ev'] != center
                or not np.allclose(params['coefficients'], coefficients/scale**np.arange(degree+1), rtol=1e-12, atol=1e-14)):
            return False
        regions = report['background_regions_ev']
        if report['background_regions'] != regions:
            return False
        if (len(regions) < 2 or any(len(pair) != 2 or not np.isfinite(pair).all()
                or pair[0] < center-scale-1e-12 or pair[1] > center+scale+1e-12 or pair[0] >= pair[1] for pair in regions)
                or any(a[1] >= b[0] for a, b in zip(regions, regions[1:]))
                or not regions[0][0] < center < regions[-1][1]):
            return False
        if saved['single_trend_regions']:
            if report['background_regions_source'] != 'user' or not np.allclose(regions, sorted(saved['single_trend_regions'])):
                return False
        else:
            selection = report['automatic_background']
            if report['background_regions_source'] != 'adaptive_candidates' or selection['outer_fraction'] != .35:
                return False
            chosen = selection['selected_indices']
            if len(chosen) != 2 or any(type(index) is not int or index < 0 for index in chosen):
                return False
            for side, index, region in zip(('left', 'right'), chosen, regions):
                candidates = selection['candidates']
                item = candidates[index]
                if any(not np.isfinite([value['score'], value['curvature_score'], value['gate_spread_score'],
                                       *value['interval']]).all() or value['score'] < 0
                       or value['point_count'] < 24 for value in candidates):
                    return False
                options = [(value['score'], -(value['interval'][1]-value['interval'][0]), value['interval'][0], i)
                           for i, value in enumerate(candidates) if value['side'] == side]
                if item['side'] != side or index != min(options)[-1] or item['interval'] != region:
                    return False
        limit, dlimit = saved['single_trend_max_sensitivity'], saved['single_trend_max_d2_change']
        if not all(np.isfinite(v) and 0 < v < 1 for v in (limit, dlimit)):
            return False
        gate, d2 = model['gate_stability'], model['d2_guard']
        values = [model['max_sensitivity'], gate['max_relative_change'], d2['max_relative_change'],
                  model['cv_original_rms'], model['cv_candidate_rms'], model['gain_min'], model['gain_max'],
                  d2['floor'], d2['max_absolute_change'], d2['window_points'], d2['floor_dominated_fraction']]
        if (not np.isfinite(values).all() or min(values) < 0 or model['gain_min'] <= 0
                or model['gain_max'] < model['gain_min'] or not 0 <= model['cv_candidate_rms'] < model['cv_original_rms']
                or model['cv_original_rms'] <= 1e-12 or model['max_sensitivity'] > limit
                or gate['max_relative_change'] > limit or d2['max_relative_change'] > dlimit
                or gate['limit'] != limit or d2['limit'] != dlimit or d2['gate_count'] < 1 or d2['windows'] < 1
                or d2['floor'] < 1 or d2['energy_points'] < 60
                or d2['derivative_window'] != clamp_sg_window(saved['derivative_window'],
                    n_energy=d2['energy_points'], polyorder=saved['derivative_polyorder'])
                or d2['derivative_polyorder'] != saved['derivative_polyorder']
                or not all(v is True for v in (model['sensitivity_pass'], gate['passed'], d2['passed']))):
            return False
        if (d2['window_points'] < 9 or not 0 <= d2['floor_dominated_fraction'] <= 1
                or not 0 <= d2['worst_gate_index'] < d2['gate_count']
                or len(d2['worst_energy_interval']) != 2
                or not center-scale-1e-12 <= d2['worst_energy_interval'][0] < d2['worst_energy_interval'][1] <= center+scale+1e-12
                or not np.isclose(model['cv_improvement_fraction'], 1-model['cv_candidate_rms']/model['cv_original_rms'])):
            return False
        variants, groups = model['sensitivity_variants'], gate['groups']
        if (len(variants) != 4 or not 1 <= len(groups) <= 3
                or any(len(item['coefficients']) != degree+1 or not np.isfinite(item['coefficients']).all()
                       for item in [*variants, *groups])
                or len(model['window_changes']) != len(variants) or len(gate['changes']) != len(groups)
                or not np.isfinite([*model['window_changes'], *gate['changes']]).all()
                or min([*model['window_changes'], *gate['changes']]) < 0
                or not np.isclose(model['max_sensitivity'], max(model['window_changes']))
                or not np.isclose(gate['max_relative_change'], max(gate['changes']))):
            return False
        group_indices = [index for group in groups for index in group['gate_indices']]
        if sorted(group_indices) != list(range(d2['gate_count'])):
            return False
        attempts = model['attempts']
        chosen_attempt = model['selected_attempt']
        if (not attempts or type(chosen_attempt) is not int or not 0 <= chosen_attempt < len(attempts)
                or attempts[chosen_attempt]['strength'] != strength or attempts[chosen_attempt]['passed'] is not True
                or strength != max(item['strength'] for item in attempts if item['passed'])):
            return False
        search = _strength_search()
        proposed = next(search)
        for item in attempts:
            vals = [item['max_window_change'], item['max_gate_change'], item['max_d2_change'], item['cv_candidate_rms']]
            if not np.isfinite(vals).all() or min(vals) < 0:
                return False
            accepted = (vals[0] <= limit and vals[1] <= limit and vals[2] <= dlimit
                        and vals[3] < model['cv_original_rms'])
            if accepted != item['passed'] or item['strength'] != proposed:
                return False
            try:
                proposed = search.send(accepted)
            except StopIteration:
                proposed = None
        if proposed is not None:
            return False
        if not np.allclose([attempts[chosen_attempt][key] for key in ('max_window_change', 'max_gate_change', 'max_d2_change', 'cv_candidate_rms')],
                           [model['max_sensitivity'], gate['max_relative_change'], d2['max_relative_change'], model['cv_candidate_rms']]):
            return False
        cv = model['cv_blocks']
        if (len(cv) != 2 or any(item['test_count'] < 12 or not np.isfinite([item['original_rms'], item['candidate_rms']]).all()
                               or min(item['original_rms'], item['candidate_rms']) < 0 for item in cv)
                or not np.isclose(model['cv_original_rms'], np.sqrt(np.mean([item['original_rms']**2 for item in cv])))
                or not np.isclose(model['cv_candidate_rms'], np.sqrt(np.mean([item['candidate_rms']**2 for item in cv])))):
            return False
        flags = report['trend_validation']
        if flags['independent_validation'] is not False:
            return False
        if not all(flags[key] is True for key in ('passed', 'finite_positive', 'same_scan_cv_improved',
                                                'window_sensitivity_pass', 'gate_stability_pass', 'd2_change_guard_pass')):
            return False
        order = saved['single_trend_order']
        selection = report['model_selection']
        if selection['selected'] != name or selection['strength'] != strength or selection['order_requested'] != order:
            return False
        if order not in ('auto', 'linear', 'quadratic') or (order != 'auto' and name != 'trend_'+order):
            return False
        if order == 'auto' and name == 'trend_quadratic':
            linear = report['models']['trend_linear']
            if linear['eligible'] and model['cv_candidate_rms'] > .85*linear['cv_candidate_rms']:
                return False
        return True
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return False
