"""Exploratory single-scan BN texture correction; no independent validation."""
from __future__ import annotations
from dataclasses import asdict, replace
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from core.drr_bn_calibration import BnCalibrationResult, _check, _smoothstep
from core.drr_bn_joint import production_d2
from core.drr_bn_settings import BnCalibrationSettings


def _high_frequency_validation(info, saved):
    """Check the explicitly sample-assisted refinement contract."""
    before, after = info['phase_only_d2_metric'], info['refined_d2_metric']
    return (info['schema_version'] == 1 and info['passed'] is True and info['adopted'] is True
            and info['source'] == 'sample_shared_high_frequency'
            and info['independent_validation'] is False
            and info['cutoff_cycles_per_pixel'] == .20
            and info['transition_cycles_per_pixel'] == .008
            and info['estimator'] == 'gate_log_mean_v1'
            and info['shrink_policy'] == 'monotone_smoothed_factor_v1'
            and info['baseline_source'] == 'phase_candidate'
            and 0 <= info['strength_min'] <= info['strength_mean'] <= info['strength_max'] <= 1
            and info['checked_d2_windows'] > 0 and info['failed_d2_windows'] == []
            and all(info[k] is True for k in ('finite_positive', 'all_gate_raw_d2_pass',
                'all_gate_normalized_d2_pass', 'full_axis_drr_pass', 'improvement_pass',
                'energy_balanced_noninferiority_pass'))
            and len(info['full_axis_drr_reports']) == 4
            and all(v['passed'] is True for v in info['full_axis_drr_reports'])
            and info['whole_axis_drr_projection']['passed'] is True
            and before['metric'] == after['metric'] == 'energy_balanced_bn_projected_d2_rms'
            and np.isfinite([before['original'], before['candidate'], after['original'], after['candidate']]).all()
            and before['original'] == after['original'] == 1.
            and 0 <= after['candidate'] <= before['candidate']+1e-12
            and np.isfinite([info['original_shared_hf_rms'], info['candidate_shared_hf_rms']]).all()
            and 0 <= info['candidate_shared_hf_rms'] < info['original_shared_hf_rms']*(1-saved['min_improvement_fraction']))


def _phase_region_validation(region, saved, validation):
    """Check phase-model metadata without pretending phase is a pixel shift."""
    info = region['model_parameters']
    gains = saved['single_smooth_gain_bounds']
    combined = region['model'] == 'phase_with_shared_high_frequency'
    source_ok = (info['basis_source'] == 'measured_bn_and_sample_shared_response'
                 and validation['sample_shared_response'] is True
                 and _high_frequency_validation(info['shared_high_frequency'], saved)
                 and info['shared_high_frequency']['refined_d2_metric'] == validation['energy_balanced_d2']
                 and info['shared_high_frequency']['refined_d2_metric']['candidate'] == region['final_candidate_projected_rms']
                 and info['shared_high_frequency']['full_axis_drr_reports'] == validation['full_axis_drr_reports']
                 and info['shared_high_frequency']['whole_axis_drr_projection'] == region['drr_projection']) if combined else info['basis_source'] == 'measured_bn_only'
    return (saved['single_full_range'] is True and info['success'] is True
            and info['d2_shrink_policy'] == 'ratio_dependent_v1'
            and info['node_count'] == 10 and source_ok
            and np.isfinite(info['phase_limit_radians'])
            and 0 < info['phase_limit_radians'] <= 8*np.pi
            and len(info['phase_knots']) == len(info['gain_knots']) == 10
            and all(np.isfinite(x) and abs(x) <= info['phase_limit_radians']+1e-10
                    for x in info['phase_knots'])
            and all(np.isfinite(x) and gains[0]-1e-10 <= x <= gains[1]+1e-10
                    for x in info['gain_knots'])
            and list(info['gain_bounds']) == list(gains)
            and 0 <= info['strength_min'] <= info['strength_mean'] <= info['strength_max'] <= 1
            and info['checked_d2_windows'] > 0 and info['failed_d2_windows'] == []
            and all(info[k] is True for k in ('all_gate_raw_d2_pass',
                'all_gate_normalized_d2_pass', 'full_axis_drr_pass'))
            and len(info['full_axis_drr_reports']) == 4
            and all(v['passed'] is True for v in info['full_axis_drr_reports'])
            and validation['projected_metric'] == 'energy_balanced_bn_projected_d2_rms'
            and len(validation['energy_balanced_d2']['windows']) == 6
            and region['unprojected_d2_check'] is True
            and region['drr_projection']['passed'] is True
            and region['drr_projection']['gate_count'] > 0
            and np.isfinite([region['drr_projection']['old_mean'],
                            region['drr_projection']['new_mean']]).all()
            and region['final_candidate_projected_rms'] < region['final_original_projected_rms'])


def single_pair_validation_passed(report, settings=None):
    """Validate the single-pair numerical contract, never independent validity."""
    try:
        v = report['single_pair_validation']
        saved = report['settings']
        if settings is not None:
            expected = asdict(settings) if isinstance(settings, BnCalibrationSettings) else dict(settings)
            if json.dumps(saved, sort_keys=True) != json.dumps(expected, sort_keys=True):
                return False
        derivative = report['derivative_settings']
        regions = report['candidates']['single_pair']['parameters']['regions']
        accepted = [r for r in regions if r.get('accepted') is True]
        bounds = saved['gain_bounds']
        numeric = all(np.isfinite(v[k]) for k in ('original_projected_rms', 'candidate_projected_rms'))
        return (report['schema_version'] == 1 and report['optimization_policy'] == 'single_pair_v1'
                and report['independent_validation'] is False and report['measurement_count'] == 1
                and report['single_pair_scope']['mode'] == ('full_range' if saved['single_full_range'] else 'local')
                and (not saved['single_full_range'] or (report['single_pair_scope']['protected_regions'] == []
                     and len(v['full_axis_drr_reports']) == 4))
                and v['independent_validation'] is False
                and all(v[k] is True for k in ('passed', 'finite_positive', 'protection', 'measured_bn_basis', 'same_scan_improvement', 'all_gate_check', 'transition_check', 'drr_projection_check', 'full_axis_d2_check', 'full_axis_drr_check'))
                and numeric and v['original_projected_rms'] > 0 and v['candidate_projected_rms'] >= 0
                and v['candidate_projected_rms'] < v['original_projected_rms']*(1-saved['min_improvement_fraction'])
                and derivative['window_length'] == saved['derivative_window']
                and derivative['polyorder'] == saved['derivative_polyorder']
                and isinstance(derivative['used_window_length'], int)
                and bool(accepted)
                and all(_phase_region_validation(r, saved, v) if r.get('model') in ('phase_ten_knots', 'phase_with_shared_high_frequency') else (
                        np.isfinite(r['shift_pixels']) and np.isfinite(r['fine_gain'])
                        and np.isfinite(r['shift_limit_pixels']) and 0 < r['shift_limit_pixels'] <= saved['single_max_shift_pixels']+1e-12
                        and abs(r['shift_pixels']) <= r['shift_limit_pixels']+1e-12
                        and r['unprojected_d2_check'] is True
                        and r['drr_projection']['passed'] is True
                        and r['drr_projection']['gate_count'] > 0
                        and np.isfinite(r['drr_projection']['old_mean'])
                        and np.isfinite(r['drr_projection']['new_mean'])
                        and (saved['single_smooth_gain_bounds'] if r.get('model') == 'smooth_six_knots' else bounds)[0]-1e-12 <= r['fine_gain'] <= (saved['single_smooth_gain_bounds'] if r.get('model') == 'smooth_six_knots' else bounds)[1]+1e-12
                        and r['strength'] in (.25, .5, .75, 1.)
                        and (r.get('model') != 'smooth_six_knots' or (
                            r['model_parameters']['success'] is True
                            and v['projected_metric'] == 'energy_balanced_bn_projected_d2_rms'
                            and v['energy_balanced_d2']['metric'] == v['projected_metric']
                            and len(v['energy_balanced_d2']['windows']) == 6
                            and r['model_parameters']['node_count'] == 6
                            and r['model_parameters']['basis_source'] == 'measured_bn_only'
                            and r['model_parameters']['basis_method'] == 'clamped_cubic_bspline'
                            and r['model_parameters']['edge_policy'] == 'natural cubic extrapolation up to bounded shift; endpoints are locally screened'
                            and len(r['model_parameters']['shift_knots']) == 6
                            and len(r['model_parameters']['gain_knots']) == 6
                            and all(np.isfinite(x) and abs(x) <= r['shift_limit_pixels']+1e-12 for x in r['model_parameters']['shift_knots'])
                            and all(np.isfinite(x) and saved['single_smooth_gain_bounds'][0]-1e-12 <= x <= saved['single_smooth_gain_bounds'][1]+1e-12 for x in r['model_parameters']['gain_knots'])))
                        and r['final_candidate_projected_rms'] < r['final_original_projected_rms'])
                        for r in accepted))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _projected_drr_check(old_rows, new_rows, response, mask, q, settings):
    """Test BN-only DR/R texture in each gate after shared normalization.

    Quadratic detrending avoids scoring the broad spectral envelope. This
    same-scan check cannot distinguish a physical feature aligned with BN.
    """
    matrix = np.asarray(response, float)[:, mask].T
    matrix = matrix-q@(q.T@matrix)
    norms = np.linalg.norm(matrix, axis=0)
    valid = norms > 1e-12
    if not np.any(valid):
        return dict(passed=False, reason='no_resolvable_bn_drr_basis')
    u, singular, _ = np.linalg.svd(matrix[:, valid]/norms[valid], full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-8))
    if rank < 1:
        return dict(passed=False, reason='no_resolvable_bn_drr_basis')
    basis = u[:, :rank]
    def scores(rows):
        band = np.asarray(rows, float)[:, mask]
        band = band-(band@q)@q.T
        return np.linalg.norm(band@basis, axis=1)/np.sqrt(mask.sum())
    old = scores(old_rows); new = scores(new_rows)
    tolerance = np.maximum(settings.drr_noninferiority_absolute,
                           settings.drr_noninferiority_fraction*old)
    passed = bool(np.isfinite(old).all() and np.isfinite(new).all()
                  and np.all(new <= old+tolerance))
    return dict(passed=passed, old_mean=float(np.mean(old)), new_mean=float(np.mean(new)),
                old_max=float(np.max(old)), new_max=float(np.max(new)), gate_count=len(old),
                max_degradation=float(np.max(new-old)), basis_rank=rank,
                normalization='per-gate median S/BN', units='dimensionless DR/R',
                relative_tolerance=float(settings.drr_noninferiority_fraction),
                absolute_tolerance=float(settings.drr_noninferiority_absolute))


def _full_axis_drr_check(old_rows, new_rows, energy, fine, spline, config):
    edges = np.linspace(energy[0], energy[-1], 5)
    pix = np.arange(len(energy),dtype=float)
    response = np.stack((-spline(pix,1),-fine))
    checks = []
    for lo, hi in zip(edges,edges[1:]):
        mask = (energy >= lo)&(energy <= hi)
        x = energy[mask]
        if len(x) < max(config.min_band_points,4):
            checks.append(dict(region=[float(lo),float(hi)],passed=False,reason='insufficient_bandwidth'))
            continue
        x = 2*(x-x.min())/np.ptp(x)-1
        q,_ = np.linalg.qr(np.column_stack((np.ones(len(x)),x,x*x)))
        result = _projected_drr_check(old_rows,new_rows,response,mask,q,config)
        checks.append(dict(region=[float(lo),float(hi)],**result))
    return checks


def evaluate_bn_single(energy, measurements, background, *, gates, settings=None, background_repeats=None, cancellation=None, _full_range_check=False):
    """Fit bounded BN-derived texture to one scan; results are exploratory.

    The fit shares one background across gates. Physical features aligned with
    the measured BN basis are not identifiable from one scan alone.
    """
    config = settings or BnCalibrationSettings()
    if not isinstance(config, BnCalibrationSettings):
        config = BnCalibrationSettings(**dict(config))
    _check(cancellation)
    e = np.asarray(energy, float); z = np.asarray(measurements, float); b = np.asarray(background, float)
    g = np.asarray(gates, float)
    report = dict(schema_version=1, measurement_count=1, optimization_policy='single_pair_v1', independent_validation=False,
                  limitation='Same-scan evidence only; real sample features may overlap BN texture.',
                  settings=asdict(config), candidates={}, joint_validation={'passed': False},
                  derivative_settings=dict(processing='production SG energy derivative',
                      window_length=config.derivative_window, polyorder=config.derivative_polyorder))
    def finish(candidate, eligible, reasons, validation=None, params=None):
        name = 'single_pair' if eligible else 'original'
        selected = 'single_pair' if candidate is not None and not np.array_equal(candidate, b) else name
        report['single_pair_validation'] = validation or dict(passed=False)
        report['reasons'] = list(reasons)
        report['candidates']['original'] = dict(parameters={}, selection={}, selection_d2={}, selection_joint={'passed': False, 'reasons': []})
        if candidate is not None:
            v = validation or {}
            old = v.get('original_projected_rms', 0.)
            new = v.get('candidate_projected_rms', old)
            fraction = (old-new)/old if old else 0.
            report['candidates']['single_pair'] = dict(parameters=params or {},
                selection={'mean_improvement_fraction': None}, selection_d2={'mean_improvement_fraction': fraction},
                selection_joint={'passed': eligible, 'reasons': list(reasons), 'drr_noninferiority_passed': bool(v.get('drr_projection_check', False))})
        arrays = {'original': b.copy()}
        if candidate is not None:
            arrays['single_pair'] = candidate.copy()
        return BnCalibrationResult(b.copy(), arrays, selected, name, arrays[name].copy(), eligible,
                                   'single_pair' if eligible else 'original', tuple(reasons), report)
    if z.ndim == 3 and z.shape[0] == 1:
        z = z[0]
    if (e.ndim != 1 or b.shape != e.shape or g.ndim != 1 or z.shape != (len(g), len(e))
            or len(e) < max(config.template_window, config.derivative_window+2) or len(g) < 1
            or not all(np.isfinite(x).all() for x in (e, z, b, g))
            or np.any(z <= 0) or np.any(b <= 0)
            or not (np.all(np.diff(e)>0) or np.all(np.diff(e)<0))):
        return finish(None, False, ['invalid_input'])
    if (config.template_window < 3 or config.template_window % 2 != 1 or
        config.template_polyorder < 0 or config.template_polyorder >= config.template_window or
        not np.isfinite(config.taper_width) or config.taper_width <= 0 or
        not np.isfinite(config.max_shift_pixels) or config.max_shift_pixels <= 0 or
        not np.isfinite(config.single_max_shift_pixels) or config.single_max_shift_pixels <= 0 or
        not isinstance(config.gate_count, int) or config.gate_count <= 0 or
        not np.isfinite(config.min_improvement_fraction) or not 0 <= config.min_improvement_fraction < 1 or
        not np.isfinite(config.max_local_degradation_fraction) or config.max_local_degradation_fraction < 0 or
        not np.isfinite(config.max_absolute_d2_degradation) or config.max_absolute_d2_degradation < 0 or
        not np.isfinite(config.drr_noninferiority_fraction) or config.drr_noninferiority_fraction < 0 or
        not np.isfinite(config.drr_noninferiority_absolute) or config.drr_noninferiority_absolute < 0 or
        len(config.gain_bounds) != 2 or not (0 < config.gain_bounds[0] < 1 < config.gain_bounds[1]) or
        len(config.single_smooth_gain_bounds) != 2 or not (0 < config.single_smooth_gain_bounds[0] < 1 < config.single_smooth_gain_bounds[1] < float('inf'))):
        return finish(None, False, ['invalid_settings'])
    # Work in ascending physical energy; spline shifts are native pixels.
    reverse = e[0] > e[-1]
    if reverse:
        e, z, b = e[::-1].copy(), z[:, ::-1].copy(), b[::-1].copy()
    def done(candidate, eligible, reasons, validation=None, params=None):
        nonlocal b
        if reverse:
            b = b[::-1].copy()
            candidate = None if candidate is None else candidate[::-1].copy()
        return finish(candidate, eligible, reasons, validation, params)
    log_b = np.log(b)
    fine = log_b - savgol_filter(log_b, config.template_window, config.template_polyorder)
    pix = np.arange(len(e), dtype=float); spline = CubicSpline(pix, fine)
    selected = np.arange(len(g))
    if config.gate_range is not None:
        selected = selected[(g >= config.gate_range[0]) & (g <= config.gate_range[1])]
    if not len(selected):
        return done(None, False, ['no_gates_in_range'])
    selected = selected[np.unique(np.linspace(0, len(selected)-1, min(config.gate_count, len(selected))).astype(int))]
    base = z[selected] / b
    scale = np.median(base, axis=1, keepdims=True)
    base_d2, used = production_d2(e, base / scale - 1, config)
    all_original = z/b
    all_original_normalized = all_original/np.median(all_original, axis=1, keepdims=True)-1
    all_original_d2, _ = production_d2(e, all_original_normalized, config)
    report['derivative_settings']['used_window_length'] = int(used)
    guard = max(config.protection_guard_pixels, used//2 + 3)
    protected = np.zeros(len(e), bool)
    active_protected_regions = () if config.single_full_range else config.protected_regions
    effective_regions = ((float(e[0]), float(e[-1])),) if config.single_full_range else config.single_regions
    report['single_pair_scope'] = dict(mode='full_range' if config.single_full_range else 'local',
        energy_range=[float(e[0]), float(e[-1])], effective_regions=[list(v) for v in effective_regions],
        protected_regions=[list(v) for v in active_protected_regions],
        warning=('Full-range fit has no default protected sample band; physical features can be absorbed.'
                 if config.single_full_range else 'Features outside protected bands can be absorbed.'))
    for lo, hi in active_protected_regions:
        protected |= (e >= lo) & (e <= hi)
    guardmask = protected.copy()
    for shift in range(1, guard+1):
        guardmask[shift:] |= protected[:-shift]; guardmask[:-shift] |= protected[shift:]
    candidate = b.copy(); preview = b.copy(); region_reports = []
    old_all, new_all, region_bases = [], [], []
    try:
        intervals = sorted((float(x[0]), float(x[1])) for x in effective_regions)
        if not intervals or any(not np.isfinite([lo, hi]).all() or lo >= hi for lo, hi in intervals):
            return done(None, False, ['invalid_region'])
        if any(a[1] > b_[0] for a, b_ in zip(intervals, intervals[1:])):
            return done(None, False, ['overlapping_regions'])
    except (TypeError, ValueError, IndexError):
        return done(None, False, ['invalid_region'])
    for region in effective_regions:
        _check(cancellation)
        if len(region) != 2 or not np.isfinite(region).all() or region[0] >= region[1]:
            return done(None, False, ['invalid_region'])
        lo, hi = region
        width = min(config.taper_width, (hi-lo)/4)
        support = _smoothstep((e-lo)/width)*_smoothstep((hi-e)/width)
        # C2 shoulders outside the expanded protected footprint, including its edges.
        if np.any(guardmask):
            guarded_energy = e[guardmask]
            left, right = guarded_energy.min(), guarded_energy.max()
            support *= _smoothstep(np.maximum(left-e, e-right)/width)
        mask = (support > .999) & ~guardmask
        if mask.sum() < max(config.min_band_points, used+2):
            region_reports.append(dict(region=list(region), reason='insufficient_unprotected_bandwidth'))
            continue
        # Fixed, measured-BN-only derivative response subspace; remove quadratic trend.
        qx = e[mask]; qx = 2*(qx-qx.min())/np.ptp(qx)-1
        q, _ = np.linalg.qr(np.column_stack((np.ones(len(qx)), qx, qx*qx)))
        def detrend(a):
            a = a[..., mask]
            return a-(a@q)@q.T
        response = np.stack((-support*spline(pix, 1), -support*fine))
        response_d2, _ = production_d2(e, response, config)
        matrix = detrend(response_d2).T
        norms = np.linalg.norm(matrix, axis=0)
        if np.any(norms < 1e-10):
            region_reports.append(dict(region=list(region), reason='no_resolvable_bn_texture'))
            continue
        u, singular, _ = np.linalg.svd(matrix/norms, full_matrices=False)
        if singular[-1] < singular[0]*1e-5:
            region_reports.append(dict(region=list(region), reason='unidentifiable_basis'))
            continue
        basis = u[:, :2]
        def projected(d2):
            return detrend(d2) @ basis
        original_projection = projected(base_d2)
        def make(p):
            return b*np.exp(support*(p[1]*spline(pix+p[0])-fine))
        def objective(p):
            _check(cancellation)
            trial = make(p)
            ratio = z[selected]/trial
            d2, _ = production_d2(e, ratio/np.median(ratio, axis=1, keepdims=True)-1, config)
            return projected(d2).ravel()
        # Period estimate uses measured BN only; absent/flat texture falls back
        # to the previous small shift bound. The cap excludes phase aliases.
        texture = fine[mask]
        texture = texture - np.mean(texture)
        frequencies = np.fft.rfftfreq(len(texture))
        power = np.abs(np.fft.rfft(texture*np.hanning(len(texture))))**2
        valid_frequency = (frequencies >= 1/60) & (frequencies <= 1/4)
        period = None
        if np.std(texture) > 1e-9 and np.any(valid_frequency):
            eligible_indices = np.flatnonzero(valid_frequency)
            peak = eligible_indices[np.argmax(power[eligible_indices])]
            if power[peak] > 1e-18:
                period = float(1/frequencies[peak])
        shift_limit = min(config.single_max_shift_pixels,
                          period/2 if period is not None else config.max_shift_pixels)
        shifts = np.linspace(-shift_limit, shift_limit, 9)
        scored = [(float(np.dot(values, values)), shift) for shift in shifts
                  for values in (objective([shift, 1.]),)]
        initial_shift = min(scored)[1]
        result = least_squares(objective, [initial_shift, 1.], bounds=([-shift_limit, config.gain_bounds[0]],
                                  [shift_limit, config.gain_bounds[1]]), max_nfev=35)
        before = float(np.sqrt(np.mean(original_projection**2)))
        after = float(np.sqrt(np.mean(result.fun**2)))
        accept = result.success and np.isfinite(after) and before > 1e-10 and after < before*(1-config.min_improvement_fraction)
        chosen = 0.
        rejection_reason = 'fit_has_no_required_improvement' if not accept else 'all_candidate_strengths_failed_validation'
        strength_checks = []
        if accept:
            fitted = make(result.x)
            preview *= fitted/b
            # Frozen fitted parameters, with predetermined shrinkage; no per-gate fitting.
            for strength in (1., .75, .5, .25):
                _check(cancellation)
                trial = b*np.exp(strength*np.log(fitted/b))
                all_ratio = z/trial
                all_normalized = all_ratio/np.median(all_ratio, axis=1, keepdims=True)-1
                all_d2, _ = production_d2(e, all_normalized, config)
                drr_check = _projected_drr_check(all_original_normalized, all_normalized, response, mask, q, config)
                old_proj = projected(all_original_d2); new_proj = projected(all_d2)
                old_r = np.sqrt(np.mean(old_proj**2)); new_r = np.sqrt(np.mean(new_proj**2))
                gates_pass = np.all(np.linalg.norm(new_proj,axis=1) <= np.linalg.norm(old_proj,axis=1)*(1+config.max_local_degradation_fraction)+1e-10)
                # Check full regional D2, not only the linearized BN projection.
                old_full = np.sqrt(np.mean(detrend(all_original_d2)**2, axis=1))
                new_full = np.sqrt(np.mean(detrend(all_d2)**2, axis=1))
                unprojected_pass = bool(np.all(new_full <= old_full*(1+config.max_local_degradation_fraction)+config.max_absolute_d2_degradation))
                shoulders_pass = True
                for a, c in ((lo, lo+width), (hi-width, hi)):
                    shoulder = (e >= a) & (e <= c) & ~guardmask
                    if np.any(shoulder):
                        old_local = np.sqrt(np.mean(all_original_d2[:, shoulder]**2, axis=1))
                        new_local = np.sqrt(np.mean(all_d2[:, shoulder]**2, axis=1))
                        shoulders_pass &= bool(np.all(new_local <= old_local*(1+config.max_local_degradation_fraction)+config.max_absolute_d2_degradation))
                full_local_pass = True
                full_drr_pass = True
                if config.single_full_range or _full_range_check:
                    span = max(int(config.local_validation_window), int(used))
                    starts = list(range(0, max(1,len(e)-span+1), max(1,span//2)))
                    starts.append(max(0,len(e)-span))
                    for start in set(starts):
                        stop = min(len(e),start+span)
                        old_window = np.sqrt(np.mean(all_original_d2[:,start:stop]**2,axis=1))
                        new_window = np.sqrt(np.mean(all_d2[:,start:stop]**2,axis=1))
                        if np.any(new_window > old_window*(1+config.max_local_degradation_fraction)+config.max_absolute_d2_degradation):
                            full_local_pass = False
                            break
                if config.single_full_range or _full_range_check:
                    full_drr_pass = all(v['passed'] for v in _full_axis_drr_check(all_original_normalized, all_normalized, e, fine, spline, config))
                strength_checks.append(dict(strength=strength, full_axis_d2=full_local_pass, full_axis_drr=full_drr_pass, d2_projection=bool(gates_pass), d2_unprojected=bool(unprojected_pass), transition=bool(shoulders_pass), drr_projection=bool(drr_check['passed'])))
                if gates_pass and unprojected_pass and shoulders_pass and full_local_pass and full_drr_pass and drr_check['passed'] and new_r < old_r*(1-config.min_improvement_fraction):
                    chosen = strength
                    candidate *= trial/b
                    region_bases.append((mask, q, basis, response, len(region_reports)))
                    break
        region_reports.append(dict(region=list(region), shift_pixels=float(result.x[0]), fine_gain=float(result.x[1]),
                                   strength=chosen, reason='accepted' if chosen else rejection_reason, strength_checks=strength_checks, shift_limit_pixels=shift_limit, dominant_period_pixels=period, unprojected_d2_check=bool(chosen), original_projected_rms=before, candidate_projected_rms=after, accepted=bool(chosen),
                                   at_parameter_bound=bool(np.any(np.isclose(result.x, [-shift_limit, config.gain_bounds[0]], atol=1e-5)) or np.any(np.isclose(result.x, [shift_limit, config.gain_bounds[1]], atol=1e-5)))))
    if not region_bases:
        if config.single_full_range:
            # Retry with contiguous coverage and independently guarded local
            # corrections. Each rejected segment stays original; no user data
            # or protected-band defaults are silently used in full-range mode.
            edges = np.linspace(e[0], e[-1], 5)
            segments = tuple((float(edges[i]), float(edges[i+1])) for i in range(4))
            local = replace(config, single_full_range=False, single_regions=segments, protected_regions=())
            fallback = evaluate_bn_single(e, z, b, gates=g, settings=local,
                                          background_repeats=background_repeats, cancellation=cancellation, _full_range_check=True)
            segment_coverage = fallback.report.get('candidates', {}).get('single_pair', {}).get('parameters', {}).get('regions', [])
            if not fallback.eligible:
                from core.drr_bn_single_smooth import fit_smooth_texture, energy_balanced_bn_d2_score
                seed_regions = fallback.report.get('candidates', {}).get('single_pair', {}).get('parameters', {}).get('regions', [])
                smooth, smooth_info = fit_smooth_texture(e, z, b, fine, config, cancellation=cancellation, segment_seeds=seed_regions)
                smooth_info['strength_checks'] = []
                if smooth is not None:
                    for strength in (1., .75, .5, .25):
                        _check(cancellation)
                        trial = b*np.exp(strength*np.log(smooth/b))
                        ratio = z/trial
                        new_norm = ratio/np.median(ratio,axis=1,keepdims=True)-1
                        new_d2,_ = production_d2(e,new_norm,config)
                        drr_reports = _full_axis_drr_check(all_original_normalized,new_norm,e,fine,spline,config)
                        drr_ok = all(v['passed'] for v in drr_reports)
                        span = max(int(config.local_validation_window),int(used)); stride=max(1,span//2)
                        starts = list(range(0,max(1,len(e)-span+1),stride)); starts.append(max(0,len(e)-span))
                        failed=[]
                        for start in sorted(set(starts)):
                            end=min(len(e),start+span)
                            old_win=np.sqrt(np.mean(all_original_d2[:,start:end]**2,axis=1))
                            new_win=np.sqrt(np.mean(new_d2[:,start:end]**2,axis=1))
                            if np.any(new_win>old_win*(1+config.max_local_degradation_fraction)+config.max_absolute_d2_degradation):
                                failed.append(dict(energy_range=[float(e[start]),float(e[end-1])],
                                    old_max=float(np.max(old_win)),new_max=float(np.max(new_win)),
                                    max_increase=float(np.max(new_win-old_win))))
                        d2_scores = energy_balanced_bn_d2_score(e,fine,all_original_d2,new_d2,config)
                        old_score = d2_scores['original'] if d2_scores else 0.
                        new_score = d2_scores['candidate'] if d2_scores else 0.
                        original_full_rms=float(np.sqrt(np.mean(all_original_d2**2)))
                        candidate_full_rms=float(np.sqrt(np.mean(new_d2**2)))
                        improves=bool(d2_scores and old_score>0 and new_score<old_score*(1-config.min_improvement_fraction))
                        smooth_info['strength_checks'].append(dict(strength=strength,drr_pass=drr_ok,
                            d2_local_pass=not failed,failed_d2_windows=failed,d2_metric=d2_scores,
                            diagnostic_full_d2_original_rms=original_full_rms,
                            diagnostic_full_d2_candidate_rms=candidate_full_rms,improvement_pass=improves))
                        if smooth_info.get('success') and drr_ok and not failed and improves:
                            # Represent the full-range region in the standard result schema.
                            full_mask=np.ones(len(e),bool)
                            x=np.linspace(-1,1,len(e)); q,_=np.linalg.qr(np.column_stack((np.ones(len(e)),x,x*x)))
                            response=np.stack((-spline(np.arange(len(e)),1),-fine))
                            whole_drr=_projected_drr_check(all_original_normalized,new_norm,response,full_mask,q,config)
                            if not whole_drr['passed']:
                                continue
                            validation=dict(passed=True,finite_positive=bool(np.isfinite(trial).all() and np.all(trial>0)),
                                protection=True, measured_bn_basis=True,same_scan_improvement=True,all_gate_check=True,
                                drr_projection_check=True,transition_check=True,full_axis_d2_check=True,
                                full_axis_drr_check=True,full_axis_drr_reports=drr_reports,full_axis_d2_failed_windows=[],
                                original_projected_rms=old_score,candidate_projected_rms=new_score,
                                projected_metric='energy_balanced_bn_projected_d2_rms', energy_balanced_d2=d2_scores,
                                protected_pixels=0,guard_pixels=0,independent_validation=False)
                            if not validation['finite_positive']:
                                continue
                            rec=dict(region=[float(e[0]),float(e[-1])],accepted=True,reason='accepted',strength=strength,
                                shift_pixels=float(max(smooth_info['shift_knots'],key=abs)),
                                shift_limit_pixels=float(smooth_info['shift_limit_pixels']),
                                fine_gain=float(np.mean(smooth_info['gain_knots'])),unprojected_d2_check=True,
                                final_original_projected_rms=old_score,final_candidate_projected_rms=new_score,
                                drr_projection=whole_drr,model='smooth_six_knots',model_parameters=smooth_info)
                            result=done(trial,True,['full_range_unprotected_weak_features_may_change'],validation,{'regions':[rec]})
                            result.report['single_pair_scope'].update(model='smooth_six_knots',
                                coverage=[dict(region=[float(e[0]),float(e[-1])],adjusted=True,reason='accepted',strength=strength)])
                            result.report['full_range_shared_attempt']=dict(accepted=False,regions=region_reports)
                            return result
                fallback.report['smooth_texture_attempt'] = smooth_info
                if smooth is not None and np.isfinite(smooth).all() and np.all(smooth>0):
                    fallback.candidate_backgrounds['single_pair'] = smooth.copy()
                    fallback.selected_method = 'single_pair'
                    entry = fallback.report['candidates'].setdefault('single_pair', {})
                    entry['parameters'] = dict(model='smooth_six_knots', model_parameters=smooth_info)
                    entry['selection_joint'] = dict(passed=False, drr_noninferiority_passed=False,
                        reasons=['smooth_candidate_not_qualified'])
                    metrics = (smooth_info['strength_checks'][0].get('d2_metric')
                               if smooth_info['strength_checks'] else None)
                    entry['selection_d2'] = dict(mean_improvement_fraction=(
                        (metrics['original']-metrics['candidate'])/metrics['original']
                        if metrics and metrics['original'] > 0 else None),
                        metric='energy_balanced_bn_projected_d2_rms')
                    fallback.reasons = ('smooth_candidate_not_qualified',)
                    fallback.report['reasons'] = list(fallback.reasons)
                    fallback.report['smooth_texture_attempt']['preview_candidate'] = 'single_pair'
                # A phase field can describe accumulated phase changes that a
                # bounded displacement cannot. Keep every original numerical
                # guard, including both raw and normalized full-gate D2.
                from core.drr_bn_single_phase import fit_phase_texture
                phase, phase_info = fit_phase_texture(e, z, b, fine, config, cancellation=cancellation)
                fallback.report['phase_texture_attempt'] = phase_info
                if phase is not None:
                    ratio = z/phase
                    normalized = ratio/np.median(ratio,axis=1,keepdims=True)-1
                    phase_d2,_ = production_d2(e,normalized,config)
                    scores = energy_balanced_bn_d2_score(e,fine,all_original_d2,phase_d2,config)
                    phase_info['d2_metric'] = scores
                    improves = bool(scores and scores['candidate'] < scores['original']*(1-config.min_improvement_fraction))
                    phase_info['improvement_pass'] = improves
                    x=np.linspace(-1,1,len(e)); q,_=np.linalg.qr(np.column_stack((np.ones(len(e)),x,x*x)))
                    response=np.stack((-spline(np.arange(len(e)),1),-fine))
                    whole_drr=_projected_drr_check(all_original_normalized,normalized,response,np.ones(len(e),bool),q,config)
                    phase_info['whole_axis_drr_projection'] = whole_drr
                    guarded = bool(phase_info.get('success') and whole_drr['passed']
                        and all(phase_info.get(k) is True for k in ('all_gate_raw_d2_pass',
                            'all_gate_normalized_d2_pass','full_axis_drr_pass')))
                    if guarded and improves:
                        model = 'phase_ten_knots'
                        from core.drr_bn_single_high_frequency import refine_shared_high_frequency
                        refined, hf_info = refine_shared_high_frequency(e, z, b, phase, fine, config, cancellation=cancellation)
                        hf_info['adopted'] = False
                        if hf_info['passed']:
                            refined_ratio = z/refined
                            refined_norm = refined_ratio/np.median(refined_ratio,axis=1,keepdims=True)-1
                            refined_d2,_ = production_d2(e,refined_norm,config)
                            refined_scores = energy_balanced_bn_d2_score(e,fine,all_original_d2,refined_d2,config)
                            hf_info['energy_balanced_noninferiority_pass'] = bool(refined_scores and refined_scores['candidate'] <= scores['candidate']+1e-12)
                            hf_info['phase_only_d2_metric'] = scores
                            hf_info['refined_d2_metric'] = refined_scores
                            if hf_info['energy_balanced_noninferiority_pass']:
                                hf_info['adopted'] = True
                                phase, scores = refined, refined_scores
                                whole_drr = hf_info['whole_axis_drr_projection']
                                phase_info = dict(phase_info, basis_source='measured_bn_and_sample_shared_response',
                                    d2_metric=scores, whole_axis_drr_projection=whole_drr,
                                    phase_only_guard_reports={k:phase_info[k] for k in ('checked_d2_windows',
                                        'failed_d2_windows','full_axis_drr_reports')})
                                for key in ('all_gate_raw_d2_pass','all_gate_normalized_d2_pass',
                                            'full_axis_drr_pass','full_axis_drr_reports','checked_d2_windows','failed_d2_windows'):
                                    phase_info[key] = hf_info[key]
                                model = 'phase_with_shared_high_frequency'
                        phase_info['shared_high_frequency'] = hf_info
                        validation=dict(passed=True,finite_positive=bool(np.isfinite(phase).all() and np.all(phase>0)),
                            protection=True,measured_bn_basis=True,same_scan_improvement=True,all_gate_check=True,
                            sample_shared_response=model == 'phase_with_shared_high_frequency',
                            drr_projection_check=True,transition_check=True,full_axis_d2_check=True,
                            full_axis_drr_check=True,full_axis_drr_reports=phase_info['full_axis_drr_reports'],
                            full_axis_d2_failed_windows=[],original_projected_rms=scores['original'],
                            candidate_projected_rms=scores['candidate'],projected_metric=scores['metric'],
                            energy_balanced_d2=scores,protected_pixels=0,guard_pixels=0,independent_validation=False)
                        rec=dict(region=[float(e[0]),float(e[-1])],accepted=True,reason='accepted',
                            strength=phase_info['strength_mean'],unprojected_d2_check=True,
                            final_original_projected_rms=scores['original'],final_candidate_projected_rms=scores['candidate'],
                            drr_projection=whole_drr,model=model,model_parameters=phase_info)
                        result=done(phase,True,['full_range_unprotected_weak_features_may_change'],validation,{'regions':[rec]})
                        result.report['single_pair_scope'].update(model=model,coverage=[dict(
                            region=[float(e[0]),float(e[-1])],adjusted=True,reason='accepted',strength=phase_info['strength_mean'])])
                        result.report['full_range_shared_attempt']=dict(accepted=False,regions=region_reports)
                        result.report['smooth_texture_attempt']=smooth_info
                        result.report['phase_texture_attempt']=phase_info
                        return result
                    if guarded and scores:
                        fallback.candidate_backgrounds['single_pair']=phase.copy()
                        fallback.selected_method='single_pair'
                        entry=fallback.report['candidates'].setdefault('single_pair',{})
                        entry['parameters']=dict(model='phase_ten_knots',model_parameters=phase_info)
                        entry['selection_joint']=dict(passed=False,drr_noninferiority_passed=True,
                            reasons=['phase_candidate_insufficient_improvement'])
                        entry['selection_d2']=dict(mean_improvement_fraction=1-scores['candidate']/scores['original'],metric=scores['metric'])
                        fallback.reasons=('phase_candidate_insufficient_improvement',)
                        fallback.report['reasons']=list(fallback.reasons)
            fallback.report['settings'] = asdict(config)
            fallback.report['single_pair_scope'] = dict(mode='full_range', model='four_segment_fallback',
                energy_range=[float(e[0]), float(e[-1])], effective_regions=[list(v) for v in segments],
                protected_regions=[], warning='No default sample band is protected; rejected segments remain original and physical features may be absorbed.')
            fallback.report['full_range_shared_attempt'] = dict(accepted=False, regions=region_reports)
            detailed = segment_coverage
            fallback.report['single_pair_scope']['coverage'] = [dict(region=r['region'], adjusted=bool(r.get('accepted')),
                reason=r.get('reason', 'not_qualified'), strength=r.get('strength', 0.)) for r in detailed]
            preview_model=fallback.report.get('candidates',{}).get('single_pair',{}).get('parameters',{}).get('model')
            if preview_model in ('smooth_six_knots','phase_ten_knots'):
                fallback.report['single_pair_scope'].update(model=preview_model,
                    effective_regions=[[float(e[0]),float(e[-1])]],coverage=[dict(
                        region=[float(e[0]),float(e[-1])],adjusted=False,reason='preview_only',strength=0.)])
            if reverse:
                fallback.original_background = fallback.original_background[::-1].copy()
                fallback.applied_background = fallback.applied_background[::-1].copy()
                fallback.candidate_backgrounds = {k:v[::-1].copy() for k,v in fallback.candidate_backgrounds.items()}
            return fallback
        return done(preview, False, ['no_same_scan_improvement'], params={'regions': region_reports})
    # Re-score the actual combined candidate on every gate, not fit gates only.
    full_old = z / b
    full_new = z / candidate
    old_norm_d2 = all_original_d2
    new_norm_d2, _ = production_d2(e, full_new/np.median(full_new, axis=1, keepdims=True)-1, config)
    all_gate_check = True
    drr_projection_check = True
    full_new_normalized = full_new/np.median(full_new, axis=1, keepdims=True)-1
    for mask, q, basis, response, index in region_bases:
        def projection(values):
            band = values[:, mask]
            return (band - (band@q)@q.T)@basis
        oldp = projection(old_norm_d2); newp = projection(new_norm_d2)
        old_r = float(np.sqrt(np.mean(oldp**2))); new_r = float(np.sqrt(np.mean(newp**2)))
        region_reports[index]['final_original_projected_rms'] = old_r
        region_reports[index]['final_candidate_projected_rms'] = new_r
        old_band = old_norm_d2[:, mask]; new_band = new_norm_d2[:, mask]
        old_band = old_band-(old_band@q)@q.T
        new_band = new_band-(new_band@q)@q.T
        old_full = np.sqrt(np.mean(old_band**2, axis=1))
        new_full = np.sqrt(np.mean(new_band**2, axis=1))
        full_pass = bool(np.all(new_full <= old_full*(1+config.max_local_degradation_fraction)+config.max_absolute_d2_degradation))
        drr_check = _projected_drr_check(all_original_normalized, full_new_normalized, response, mask, q, config)
        region_reports[index]['drr_projection'] = drr_check
        drr_projection_check &= drr_check['passed']
        region_reports[index]['unprojected_d2_check'] = full_pass
        all_gate_check &= full_pass
        old_all.extend(oldp.ravel()); new_all.extend(newp.ravel())
        all_gate_check &= bool(new_r < old_r and np.all(np.linalg.norm(newp,axis=1) <= np.linalg.norm(oldp,axis=1)*(1+config.max_local_degradation_fraction)+1e-10))
    # Full-axis, per-gate local D2 checks include rejected segments and seams.
    # Use overlapping native windows; this never averages across gates.
    local_checks = []
    span = max(int(config.local_validation_window), int(used))
    stride = max(1, span//2)
    starts = list(range(0, max(1,len(e)-span+1), stride))
    starts.append(max(0,len(e)-span))
    for start in sorted(set(starts)):
        stop = min(len(e), start+span)
        old_local = np.sqrt(np.mean(old_norm_d2[:,start:stop]**2, axis=1))
        new_local = np.sqrt(np.mean(new_norm_d2[:,start:stop]**2, axis=1))
        passed = bool(np.all(new_local <= old_local*(1+config.max_local_degradation_fraction)+config.max_absolute_d2_degradation))
        local_checks.append(dict(energy_range=[float(e[start]),float(e[stop-1])], passed=passed))
    full_axis_d2_check = not (config.single_full_range or _full_range_check) or all(item['passed'] for item in local_checks)
    full_axis_drr_reports = _full_axis_drr_check(all_original_normalized, full_new_normalized, e, fine, spline, config) if (config.single_full_range or _full_range_check) else []
    full_axis_drr_check = all(item['passed'] for item in full_axis_drr_reports)
    # Check the actual derivative in transition shoulders on every gate.
    transition_check = True
    transition_reports = []
    for region in region_reports:
        if not region.get('accepted'):
            continue
        lo, hi = region['region']; width = min(config.taper_width, (hi-lo)/4)
        for a, c in ((lo, lo+width), (hi-width, hi)):
            mask = (e >= a) & (e <= c) & ~guardmask
            if not np.any(mask):
                continue
            old_local = np.sqrt(np.mean(old_norm_d2[:, mask]**2, axis=1))
            new_local = np.sqrt(np.mean(new_norm_d2[:, mask]**2, axis=1))
            passed = bool(np.all(new_local <= old_local*(1+config.max_local_degradation_fraction)+config.max_absolute_d2_degradation))
            transition_reports.append(dict(region=[float(a), float(c)], passed=passed))
            transition_check &= passed
    finite = bool(np.isfinite(candidate).all() and np.all(candidate > 0))
    old_drr = z / b - 1; new_drr = z / candidate - 1
    old_d2, _ = production_d2(e, old_drr, config); new_d2, _ = production_d2(e, new_drr, config)
    protection = bool(np.allclose(old_drr[:, protected], new_drr[:, protected], atol=config.protected_drr_absolute_tolerance, rtol=0)
                      and np.allclose(old_d2[:, protected], new_d2[:, protected], atol=config.protected_d2_absolute_tolerance, rtol=0))
    oldr = float(np.sqrt(np.mean(np.square(old_all)))); newr = float(np.sqrt(np.mean(np.square(new_all))))
    validation = dict(passed=finite and protection and all_gate_check and transition_check and drr_projection_check and full_axis_d2_check and full_axis_drr_check and newr < oldr*(1-config.min_improvement_fraction), finite_positive=finite, protection=protection,
                      measured_bn_basis=True, same_scan_improvement=newr < oldr, all_gate_check=all_gate_check, full_axis_d2_check=full_axis_d2_check, full_axis_drr_check=full_axis_drr_check, full_axis_drr_reports=full_axis_drr_reports, full_axis_d2_failed_windows=[v for v in local_checks if not v['passed']], drr_projection_check=bool(drr_projection_check), transition_check=transition_check, transition_reports=transition_reports,
                      original_projected_rms=oldr, candidate_projected_rms=newr,
                      protected_pixels=int(protected.sum()), guard_pixels=int(guard), independent_validation=False)
    ok = validation['passed']
    return done(candidate, ok, (['parameter_at_bound_low_confidence'] if any(r.get('at_parameter_bound') for r in region_reports if r.get('accepted')) else []) if ok else ['numerical_or_protection_check_failed'], validation, {'regions': region_reports})
