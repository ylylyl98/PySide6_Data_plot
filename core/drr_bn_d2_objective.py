"""BN-anchored, fixed-SG D2 training for the constant shift/gain families.

Only complete training repeats enter this fit. The scoring band is not an
independent energy holdout: actual SG/regridding uses all their native channels.
No sample-derived modes or free background spline are introduced.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from core.drr_bn_joint import products, production_d2

POLICY = 'd2_shrinkage_v1'
METRIC = 'BN-projected production D2 residual'
BASIS_SOURCE = 'BN-only shift/gain tangent responses'
CONTEXT = 'full training-repeat spectra; score band is not an independent energy holdout'
INFLUENCE_POLICY = 'training_feature_influence_shrinkage_v2'
REFINEMENT_OFFSETS = (-1., -.5, 0., .5, 1.)
REFINEMENT_WIDTHS = (1., 1.25, 1.5, 2., 2.5, 3.)


def ranked_influence_probes(audit, count=3):
    """Worst DR/R, worst D2, then joint risk across both declared grids."""
    probes = [dict(probe, probe_source=source)
              for source, items in [('coarse', audit['screening']['probes']),
                                    ('refinement', audit.get('refinement', {}).get('probes', []))]
              for probe in items]
    key = lambda probe: (0 if probe['probe_source'] == 'coarse' else 1, probe['probe_index'])
    dr = sorted(probes, key=lambda probe: (-probe['drr_error_fraction'], *key(probe)))
    d2 = sorted(probes, key=lambda probe: (-probe['d2_error_fraction'], *key(probe)))
    joint = sorted(probes, key=lambda probe: (-max(probe['drr_error_fraction'], probe['d2_error_fraction']), *key(probe)))
    chosen, seen = [], set()
    for probe in [dr[0], d2[0], *joint]:
        reference = (probe['probe_source'], probe['probe_index'])
        if reference not in seen:
            chosen.append(probe)
            seen.add(reference)
        if len(chosen) == count:
            break
    return chosen


def fit_regions(settings):
    """An explicit list overrides the legacy single training band."""
    return settings.train_regions or (settings.train_region,)


def _band_masks(energy, mask, regions=None):
    if regions is not None:
        return [mask & (energy >= lo) & (energy <= hi) for lo, hi in regions]
    ids = np.flatnonzero(mask)
    groups = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
    return [np.isin(np.arange(len(energy)), group) for group in groups if len(group)]


def _detrender(energy, mask, regions=None):
    """Block diagonal quadratic removal; gaps never connect separate bands."""
    bands = _band_masks(energy, mask, regions)
    result = np.zeros((int(mask.sum()), 3*len(bands)))
    for number, band in enumerate(bands):
        x = energy[band]
        if len(x) < 4 or np.ptp(x) <= 0:
            raise ValueError('Insufficient training band bandwidth')
        x = 2*(x-x.min())/np.ptp(x)-1
        result[np.flatnonzero(band[mask]), 3*number:3*number+3] = np.linalg.qr(
            np.column_stack((np.ones(len(x)), x, x*x)))[0]
    return result


def training_bandwidth(energy, score_mask, used_window, polyorder):
    """Require one contiguous scoring segment to contain a real SG window.

    This is a necessary operational identifiability guard, not a certificate
    for preservation of arbitrary unknown peaks or zero interpolation tails.
    """
    spacing = abs(np.diff(energy))
    uniform = np.allclose(spacing, np.median(spacing), rtol=1e-6, atol=1e-12)
    step = float(np.median(spacing)) if uniform else float(np.ptp(energy)/(len(energy)-1))
    # Match the finite, nonuniform production regridding path's minimum window.
    actual_window = used_window if uniform else max(used_window, polyorder+3)
    if actual_window >= len(energy):
        actual_window = len(energy)-(1-len(energy)%2)
    footprint = (actual_window-1)*step
    channels = np.flatnonzero(score_mask)
    segments = np.split(channels, np.flatnonzero(np.diff(channels) != 1)+1)
    segments = [segment for segment in segments if len(segment)]
    if not segments:
        raise ValueError('Empty training scoring mask')
    segment = max(segments, key=lambda ids: min(len(ids)/actual_window,
                  float(np.ptp(energy[ids]))/max(footprint, 1e-15)))
    span = float(np.ptp(energy[segment]))
    passed = bool(len(segment) >= actual_window and span+1e-12 >= footprint)
    return dict(passed=passed, scoring_energy_span=span, sg_footprint_energy=footprint,
        score_channel_count=len(segment), required_channel_count=int(actual_window),
        total_score_channel_count=len(channels), contiguous_scoring_segment_count=len(segments),
        scoring_channel_indices=segment.tolist(), production_spacing=step,
        segments=[dict(channel_indices=ids.tolist(), channel_count=len(ids),
                       energy_span=float(np.ptp(energy[ids])),
                       passed=bool(len(ids) >= actual_window and float(np.ptp(energy[ids]))+1e-12 >= footprint))
                  for ids in segments],
        criterion='One contiguous training score segment must contain at least one actual SG physical window.',
        reason=None if passed else 'insufficient_d2_training_bandwidth')


def fit_d2_priority(data, background, repeat_ids, gate_ids, energy, score_mask,
                    support, fine, spline, settings, method, check):
    """Fit D2 texture energy after per-repeat/per-gate projection, never spectra averaging."""
    check()
    index = np.arange(len(energy), dtype=float)
    # Differential DR/R response to the two BN parameters at original BN.
    # Both columns are fixed from the measured BN only, for both families.
    response = np.stack((-support*spline(index, 1), -support*fine))
    response_d2, used = production_d2(energy, response, settings)
    q = _detrender(energy, score_mask, fit_regions(settings))

    def detrend(values):
        band = values[..., score_mask]
        return band-(band@q)@q.T

    design = detrend(response_d2).T
    norms = np.linalg.norm(design, axis=0)
    valid = norms > 1e-12
    normalized = design[:, valid]/norms[valid]
    if normalized.shape[1]:
        u, singular, _ = np.linalg.svd(normalized, full_matrices=False)
        rank = int(np.sum(singular > singular[0]*1e-8))
        basis = u[:, :rank]
    else:
        singular, rank = np.empty(0), 0
        basis = np.empty((int(score_mask.sum()), 0))
    evidence = dict(policy=POLICY, metric=METRIC, d2_training_evaluated=False,
        fixed_bn_basis=True, basis_source=BASIS_SOURCE, basis_rank=rank,
        basis_singular_values=singular.tolist(), basis_response_norms=norms.tolist(),
        repeat_indices=[int(i) for i in repeat_ids], gate_indices=[int(i) for i in gate_ids],
        channel_indices=list(range(len(energy))), score_channel_indices=np.flatnonzero(score_mask).tolist(),
        context=CONTEXT, used_window_length=used,
        aggregation='Per-repeat/per-gate projected D2 squared RMS, then arithmetic mean; spectra never averaged.',
        display_metric_is_training_metric=False, evaluations=0)
    evidence['bandwidth_check'] = training_bandwidth(energy, score_mask, used, settings.derivative_polyorder)
    evidence['fit_bands'] = []
    for region, band in zip(fit_regions(settings), _band_masks(energy, score_mask, fit_regions(settings))):
        local_q = _detrender(energy, band)
        local_design = response_d2[:, band].T
        local_design -= local_q@(local_q.T@local_design)
        local_norms = np.linalg.norm(local_design, axis=0)
        local_rank = int(np.linalg.matrix_rank(local_design/np.maximum(local_norms, 1e-30), tol=1e-8))
        evidence['fit_bands'].append(dict(region=list(region), channel_indices=np.flatnonzero(band).tolist(),
            channel_count=int(band.sum()), energy_span=float(np.ptp(energy[band])),
            basis_response_norms=local_norms.tolist(), basis_rank=local_rank,
            bandwidth_check=training_bandwidth(energy, band, used, settings.derivative_polyorder)))
    if rank < (2 if method == 'local_shift_gain' else 1):
        return background.copy(), dict(shift_pixel=0., fine_gain=1., optimizer_success=False,
            near_parameter_bound=False, training_log_rms=0., reason='no_resolvable_bn_d2_basis',
            training_objective=evidence)

    # Copy training repeats/gates only. All energy channels provide real SG context.
    training = np.asarray(data[np.ix_(repeat_ids, gate_ids, np.arange(len(energy)))], float)
    ids = list(range(len(training)))

    def metrics(drr, d2):
        coefficients = detrend(d2)@basis
        projected = np.sqrt(np.sum(coefficients**2, axis=-1)/score_mask.sum())
        dr = np.sqrt(np.mean(detrend(drr)**2, axis=-1))
        return projected, dr

    base = products(energy, training, background, ids, settings)
    original, original_drr = metrics(base[0], base[1])
    original_energy = float(np.mean(original**2))
    normalizer = max(original_energy, 1e-16)
    allowance = np.maximum(settings.drr_noninferiority_absolute,
                           settings.drr_noninferiority_fraction*original_drr)
    dr_scale = np.maximum(original_drr, settings.drr_noninferiority_absolute)
    cached = None

    def evaluate(parameters):
        nonlocal cached
        check()
        key = tuple(float(v) for v in parameters)
        if cached is not None and cached[0] == key:
            return cached[1:]
        shift = key[0]
        gain = key[1] if len(key) == 2 else 1.
        delta = gain*spline(np.clip(index+shift, 0, len(index)-1))-fine
        candidate = background*np.exp(support*delta)
        result = products(energy, training, candidate, ids, settings)
        projected, dr = metrics(result[0], result[1])
        value = float(np.mean(projected**2)/normalizer)
        evidence['evaluations'] += 1
        cached = (key, value, (original_drr+allowance-dr)/dr_scale, candidate, projected, dr)
        return cached[1:]

    initial = [0., 1.] if method == 'local_shift_gain' else [0.]
    bounds = [(-settings.max_shift_pixels, settings.max_shift_pixels)]
    if len(initial) == 2:
        bounds.append(settings.gain_bounds)
    optimized = minimize(lambda p: evaluate(p)[0], initial, method='SLSQP', bounds=bounds,
        constraints=[dict(type='ineq', fun=lambda p: evaluate(p)[1].ravel())],
        options=dict(maxiter=60, ftol=1e-9, eps=1e-5))
    value, margins, candidate, projected, dr = evaluate(optimized.x)
    shift = float(optimized.x[0])
    gain = float(optimized.x[1]) if len(initial) == 2 else 1.
    bound = abs(shift) >= .98*settings.max_shift_pixels
    if len(initial) == 2:
        gain_margin = .02*(settings.gain_bounds[1]-settings.gain_bounds[0])
        bound |= gain <= settings.gain_bounds[0]+gain_margin or gain >= settings.gain_bounds[1]-gain_margin
    log_values = np.log(training/candidate)
    log_rms = float(np.sqrt(np.mean(detrend(log_values)**2)))
    band_diagnostics = []
    for band in _band_masks(energy, score_mask, fit_regions(settings)):
        local_q = _detrender(energy, band)
        local_design = response_d2[:, band].T
        local_design -= local_q@(local_q.T@local_design)
        local_design = local_design[:, :len(initial)]
        local_base = base[1][..., band]
        local_base = local_base-(local_base@local_q)@local_q.T
        shifts = -np.linalg.lstsq(local_design, local_base.reshape(-1, int(band.sum())).T, rcond=None)[0].T
        band_diagnostics.append(dict(interpretation='BN tangent least-squares diagnostic only; not an independent nonlinear refit or acceptance gate.',
            shift_pixel=float(np.mean(shifts[:, 0])),
            fine_gain=float(1+np.mean(shifts[:, 1])) if len(initial) == 2 else 1.))
    evidence['band_consistency'] = dict(diagnostic_only=True, bands=band_diagnostics,
        shift_range=float(np.ptp([v['shift_pixel'] for v in band_diagnostics])),
        gain_range=float(np.ptp([v['fine_gain'] for v in band_diagnostics])))
    evidence.update(d2_training_evaluated=True, original_projected_rms=original.tolist(),
        candidate_projected_rms=projected.tolist(), original_drr_rms=original_drr.tolist(),
        candidate_drr_rms=dr.tolist(), normalized_objective=value,
        original_normalized_objective=original_energy/normalizer,
        original_projected_mean_square=original_energy, normalization_mean_square=normalizer,
        drr_noninferiority_passed=bool(np.min(margins) >= -1e-6),
        optimizer_message=str(optimized.message), iterations=int(optimized.nit))
    return candidate, dict(shift_pixel=shift, fine_gain=gain, training_log_rms=log_rms,
        near_parameter_bound=bool(bound), optimizer_success=bool(optimized.success and np.min(margins) >= -1e-6),
        training_objective=evidence)


def audit_training_influence(data, background, train_ids, gate_ids, energy, score_mask,
        support, settings, method, candidate, parameters, check, refit, *,
        unshrunk_candidate=None, forcing_cache=None):
    """Finite-probe qualification of the frozen family, using training data only.

    Local Gauss-Newton influence is a screening approximation, never an upper
    bound or a proof for arbitrary unknown signals. The three ranked probes
    additionally undergo actual same-family refits; any failure only vetoes.
    """
    check()
    training = np.asarray(data[train_ids], float).copy()
    alpha = float(parameters.get('correction_strength', 1.))
    family = parameters.get('family', method)
    unshrunk = (background*np.exp(np.log(candidate/background)/alpha)
                if unshrunk_candidate is None else unshrunk_candidate)
    ratio = training/candidate
    fit_ratio = training/unshrunk
    index = np.arange(len(energy), dtype=float)
    log_b = np.log(background)
    fine = log_b-savgol_filter(log_b, settings.template_window, settings.template_polyorder)
    spline = CubicSpline(index, fine)
    q = _detrender(energy, score_mask, fit_regions(settings))

    def band(values):
        x = values[..., score_mask]
        return x-(x@q)@q.T

    fixed_response = np.stack((-support*spline(index, 1), -support*fine))
    fixed_d2, used = production_d2(energy, fixed_response, settings)
    design = band(fixed_d2).T
    norms = np.linalg.norm(design, axis=0)
    valid = norms > 1e-12
    u, singular, _ = np.linalg.svd(design[:, valid]/norms[valid], full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-8))
    basis = u[:, :rank]

    def project(curvature):
        return band(curvature)@basis

    shifted_index = np.clip(index+parameters['shift_pixel'], 0, len(index)-1)
    clip_slope = ((index+parameters['shift_pixel'] > 0)
                  & (index+parameters['shift_pixel'] < len(index)-1))
    tangent = [support*parameters['fine_gain']*spline(shifted_index, 1)*clip_slope]
    if family == 'local_shift_gain':
        tangent.append(support*spline(shifted_index))
    # Fit sensitivity belongs to the unshrunk estimator. Output sensitivity
    # belongs to the post-fit, fixed-alpha correction. Inverting the output
    # Jacobian would cancel alpha and audit a different estimator.
    fit_response = -np.asarray(tangent)[:, None, None, :]*fit_ratio[None, ...]
    fit_response_d2, _ = production_d2(energy, fit_response, settings)
    response = -alpha*np.asarray(tangent)[:, None, None, :]*ratio[None, ...]
    response_d2, _ = production_d2(energy, response, settings)
    coefficients = project(fit_response_d2)[:, :, gate_ids, :]
    jacobian = np.moveaxis(coefficients, 0, -1).reshape(-1, len(tangent))
    jnorm = np.linalg.norm(jacobian, axis=0)
    jrank = np.linalg.matrix_rank(jacobian/np.maximum(jnorm, 1e-30), tol=1e-8)
    inverse = np.linalg.pinv(jacobian/np.maximum(jnorm, 1e-30), rcond=1e-8)/np.maximum(jnorm[:, None], 1e-30)
    objective = parameters['training_objective']
    bw = objective['bandwidth_check']
    half = bw['sg_footprint_energy']/2
    active = energy[support > 0]
    centers = np.linspace(max(float(np.min(energy)), float(np.min(active))-half),
                          min(float(np.max(energy)), float(np.max(active))+half), 17)
    native_step = float(np.median(abs(np.diff(energy))))
    audit = dict(schema_version=3, policy=INFLUENCE_POLICY, passed=False,
        training_only=True, selected_method=method, train_repeat_indices=[int(i) for i in train_ids],
        family=family, correction_strength=alpha,
        fit_jacobian_background='unshrunk fitted BN', forcing_background='unshrunk fitted BN',
        output_response_background='fixed-alpha shrunk BN', output_response_strength=alpha,
        actual_refit_rule='Rerun unshrunk raw-data fit, then apply the same fixed alpha.',
        fit_gate_indices=[int(i) for i in gate_ids], checked_gate_indices=list(range(training.shape[1])),
        amplitude=2e-4, signs=[-1, 1], sigma_native_spacings=[1.5, 3],
        probe_mode='common_to_all_training_repeats_and_gates',
        position_count=17, positions_energy=centers.tolist(), native_median_spacing=native_step,
        max_relative_error=.05, used_window_length=used,
        qualification='Finite declared feature family only; local Gauss-Newton response is not a fidelity bound.',
        screening=dict(linear_approximation=True, passed=False, probes=[]),
        refinement=dict(schema_version=1, policy='native_feature_grid_v1', evaluated=False, passed=False,
            max_seed_centers=3, seed_center_indices=[], seed_centers_energy=[],
            center_offsets_native_spacings=list(REFINEMENT_OFFSETS),
            sigma_native_spacings=list(REFINEMENT_WIDTHS), signs=[-1, 1],
            center_bounds_energy=[float(np.min(energy)), float(np.max(energy))],
            center_clipping='Clip each seed-plus-offset center to the full native energy bounds.',
            duplicate_rule='Exact (center_energy, sigma_energy, sign) tuples; omit coarse probes and preceding refined probes.',
            max_probe_count=168, expected_probe_count=0, probe_count=0, probes=[],
            seed_selection_rule='Rank distinct coarse centers by maximum DR/R or D2 screening error across signs and widths; take the first three, deterministic center-index ties.',
            qualification='Finite native-spacing neighborhoods of three high-risk coarse centers; not exhaustive coverage or a universal fidelity bound.',
            reason='coarse_screen_not_passed'),
        actual_refits=dict(count=0, passed=False, probes=[]), reasons=[])
    raw_index = index+parameters['shift_pixel']
    if np.any(((abs(raw_index) < 1e-12) | (abs(raw_index-(len(index)-1)) < 1e-12)) & (support > 0)):
        audit['reasons'].append('influence_nonsmooth_parameter_response')
        return audit
    if jrank != len(tangent) or np.any(jnorm <= 1e-12):
        audit['reasons'].append('influence_response_unidentifiable')
        return audit
    old_dr = np.asarray(objective['original_drr_rms'])
    new_dr = np.asarray(objective['candidate_drr_rms'])
    allowance = np.maximum(settings.drr_noninferiority_absolute, settings.drr_noninferiority_fraction*old_dr)
    normalized_margin = (old_dr+allowance-new_dr)/np.maximum(old_dr, settings.drr_noninferiority_absolute)
    if np.min(normalized_margin) <= 1e-5:
        audit['reasons'].append('influence_active_training_constraint')

    def errors(delta_drr, delta_d2, known, known_d2):
        dr = np.max(abs(delta_drr), axis=-1)/np.maximum(np.max(abs(known), axis=-1), 1e-15)
        d2 = np.max(abs(delta_d2), axis=-1)/np.maximum(np.max(abs(known_d2), axis=-1), 1e-15)
        return dict(drr_error_fraction=float(np.max(dr)), d2_error_fraction=float(np.max(d2)),
                    per_row_drr_error_fraction=dr.tolist(), per_row_d2_error_fraction=d2.tolist())

    def screen(specs, source):
        screened = []
        # Batch only independent rows through the exact production pipeline.
        # No interpolation, derivative settings, averaging, or estimator changes.
        for start in range(0, len(specs), 8):
            check()
            block = specs[start:start+8]
            features = np.asarray([spec['sign']*2e-4*np.exp(
                -.5*((energy-spec['center_energy'])/spec['sigma_energy'])**2) for spec in block])
            added = np.expm1(features)[:, None, None, :]
            known = ratio[None, ...]*added
            known_d2, _ = production_d2(energy, known, settings)
            keys = [(family, spec['center_energy'], spec['sigma_energy'], spec['sign']) for spec in block]
            deltas = [None if forcing_cache is None else forcing_cache.get(key) for key in keys]
            missing = [i for i, delta in enumerate(deltas) if delta is None]
            if missing:
                fit_known = fit_ratio[None, ...]*added[missing]
                fit_known_d2, _ = production_d2(energy, fit_known, settings)
                forcing = project(fit_known_d2)[:, :, gate_ids, :].reshape(len(missing), -1)
                fitted_deltas = -forcing@inverse.T
                for offset, delta in zip(missing, fitted_deltas):
                    deltas[offset] = delta
                    if forcing_cache is not None:
                        forcing_cache[keys[offset]] = delta.copy()
            predicted = np.einsum('np,prge->nrge', np.asarray(deltas), response)
            predicted_d2 = np.einsum('np,prge->nrge', np.asarray(deltas), response_d2)
            for offset, spec in enumerate(block):
                screened.append(dict(probe_index=start+offset, probe_source=source, **spec,
                    **errors(predicted[offset], predicted_d2[offset], known[offset], known_d2[offset])))
        return screened

    specs = [dict(center_energy=float(center), sigma_energy=float(scale*native_step),
                  sigma_native_spacings=scale, sign=sign)
             for center in centers for scale in (1.5, 3.) for sign in (-1, 1)]
    audit['screening']['probes'] = screen(specs, 'coarse')
    screened = audit['screening']['probes']
    audit['screening'].update(probe_count=len(screened),
        batch_probe_count=8,
        max_drr_error_fraction=max(x['drr_error_fraction'] for x in screened),
        max_d2_error_fraction=max(x['d2_error_fraction'] for x in screened))
    audit['screening']['passed'] = bool(not audit['reasons'] and
        audit['screening']['max_drr_error_fraction'] <= .05 and
        audit['screening']['max_d2_error_fraction'] <= .05)
    if not audit['screening']['passed']:
        audit['reasons'].append('training_feature_influence_screen_failed')
    else:
        refinement = audit['refinement']
        ranked_centers = sorted(range(len(centers)), key=lambda i: (
            -max(max(item['drr_error_fraction'], item['d2_error_fraction']) for item in screened[4*i:4*i+4]), i))
        seeds = ranked_centers[:3]
        used_specs = {(spec['center_energy'], spec['sigma_energy'], spec['sign']) for spec in specs}
        refined_specs = []
        for center_index in seeds:
            for offset in REFINEMENT_OFFSETS:
                center = float(np.clip(centers[center_index]+offset*native_step, np.min(energy), np.max(energy)))
                for scale in REFINEMENT_WIDTHS:
                    sigma = float(scale*native_step)
                    for sign in (-1, 1):
                        key = (center, sigma, sign)
                        if key not in used_specs:
                            refined_specs.append(dict(center_energy=center, sigma_energy=sigma,
                                sigma_native_spacings=scale, sign=sign, seed_center_index=center_index,
                                center_offset_native_spacings=offset))
                            used_specs.add(key)
        refined = screen(refined_specs, 'refinement')
        refinement.update(evaluated=True, seed_center_indices=seeds,
            seed_centers_energy=[float(centers[i]) for i in seeds], probes=refined, probe_count=len(refined),
            expected_probe_count=len(refined_specs),
            batch_probe_count=8,
            max_drr_error_fraction=max(item['drr_error_fraction'] for item in refined),
            max_d2_error_fraction=max(item['d2_error_fraction'] for item in refined))
        refinement['passed'] = bool(refinement['max_drr_error_fraction'] <= .05
                                    and refinement['max_d2_error_fraction'] <= .05)
        refinement['reason'] = None if refinement['passed'] else 'training_feature_refinement_screen_failed'
        if not refinement['passed']:
            audit['reasons'].append(refinement['reason'])

    chosen = ranked_influence_probes(audit)
    audit['actual_refits']['chosen_probe_references'] = [
        dict(source=probe['probe_source'], probe_index=probe['probe_index']) for probe in chosen]
    for probe in chosen:
        check()
        center, sigma, sign = probe['center_energy'], probe['sigma_energy'], probe['sign']
        feature = sign*2e-4*np.exp(-.5*((energy-center)/sigma)**2)
        injected = training*np.exp(feature)
        fitted, diagnostic = refit(injected)
        known = (injected-training)/candidate
        known_d2, _ = production_d2(energy, known, settings)
        adaptation = injected*(1/fitted-1/candidate)
        adaptation_d2, _ = production_d2(energy, adaptation, settings)
        measured = errors(adaptation, adaptation_d2, known, known_d2)
        passed = bool(diagnostic['optimizer_success'] and not diagnostic['near_parameter_bound']
                      and measured['drr_error_fraction'] <= .05 and measured['d2_error_fraction'] <= .05)
        audit['actual_refits']['probes'].append(dict(probe_index=probe['probe_index'], probe_source=probe['probe_source'],
            center_energy=center, sigma_energy=sigma, sign=sign, passed=passed,
            optimizer_success=diagnostic['optimizer_success'],
            shift_pixel=diagnostic['shift_pixel'], fine_gain=diagnostic['fine_gain'],
            family=diagnostic.get('family', family), correction_strength=diagnostic.get('correction_strength', alpha),
            **measured))
    actual = audit['actual_refits']
    actual.update(count=len(actual['probes']), passed=all(x['passed'] for x in actual['probes']),
        max_drr_error_fraction=max(x['drr_error_fraction'] for x in actual['probes']),
        max_d2_error_fraction=max(x['d2_error_fraction'] for x in actual['probes']))
    if not actual['passed']:
        audit['reasons'].append('training_feature_influence_refit_failed')
    audit['passed'] = bool(audit['screening']['passed'] and audit['refinement']['passed'] and actual['passed'])
    return audit
