"""Current-PN-only localized subtraction; experimental, non-unique decomposition.

generate is a pure numerical API. It never receives BN, historical calibration,
sample spectra, derivatives of sample spectra, Direct, or file paths.
"""
from __future__ import annotations

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "1a116b14785b794e0ad086a528b85a52d35834aceacd2f8163dcaede8ef8dd87"

import hashlib
import json

import numpy as np
from .source_runtime import least_squares
from scipy.signal import find_peaks

FAMILIES = ('local_subtraction', 'reliability_weighted', 'opposite_attribution')
SETTINGS = dict(edge_smoothing_ev=.004, edge_height_per_ev=2.,
                edge_prominence_per_ev=1.5, suppression_ev=.014,
                detection_contrast_log=.006, contrast_radius_ev=.015,
                edge_margin_ev=.060, maximum_components=8,
                plateau_radius_ev=.025, support_radius_ev=.050,
                center_radius_ev=.006, width_bounds_ev=[.0006, .020],
                q_bounds=[-3., 0.], log_c_bounds=[-6., 3.],
                nuisance_bounds=[-.5, .5], feature_penalty=.003,
                soft_l1_scale=.003, starts_width_ev=[.0015, .004, .010],
                max_nfev=800, fit_tolerance=1e-9,
                minimum_applied_feature_log=.005,
                local_residual_floor_log=.005,
                local_residual_target_fraction=.15, boundary_fraction=.01,
                reliability_scale_log=.015, flank_radii_ev=[.030, .050],
                flank_minimum_points_per_side=12, flank_max_residual_log=.010)


def _sha_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def _rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2))) if np.size(a) else 0.


def _quintic(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10.+x*(-15.+6.*x))


def _taper(e, center):
    inner = SETTINGS['plateau_radius_ev']
    outer = SETTINGS['support_radius_ev']
    return _quintic((outer-np.abs(e-center))/(outer-inner))


def _q4(e, values):
    """Native-coordinate local quadratic; used for detection, never output."""
    width = SETTINGS['edge_smoothing_ev']
    out = np.empty_like(values)
    for i, center in enumerate(e):
        lo = np.searchsorted(e, center-width/2)
        hi = np.searchsorted(e, center+width/2, side='right')
        if hi-lo < 7:
            lo = max(0, min(i-3, len(e)-7)); hi = lo+7
        x = (e[lo:hi]-center)/(width/2)
        design = np.column_stack((np.ones(len(x)), x, x*x))
        out[i] = np.linalg.lstsq(design, values[lo:hi], rcond=None)[0][0]
    return out


def _detect(e, ratio):
    smooth = _q4(e, ratio)
    edge = np.gradient(smooth, e)
    peaks, props = find_peaks(np.abs(edge),
        height=SETTINGS['edge_height_per_ev'],
        prominence=SETTINGS['edge_prominence_per_ev'])
    candidates = []
    for k, i in enumerate(peaks):
        local = np.abs(e-e[i]) <= SETTINGS['contrast_radius_ev']
        contrast = float(np.ptp(np.quantile(smooth[local], [.05, .95])))
        reasons = []
        if contrast < SETTINGS['detection_contrast_log']:
            reasons.append('insufficient_differential_contrast')
        if min(e[i]-e[0], e[-1]-e[i]) < SETTINGS['edge_margin_ev']:
            reasons.append('insufficient_two_sided_energy_support')
        candidates.append(dict(index=int(i), energy_ev=float(e[i]),
            edge_per_ev=float(edge[i]), prominence_per_ev=float(props['prominences'][k]),
            contrast_log=contrast, source=0 if edge[i] < 0 else 1,
            rejected_reasons=reasons))
    selected = []
    for row in sorted(candidates, key=lambda v: -abs(v['edge_per_ev'])):
        if row['rejected_reasons']:
            continue
        if any(abs(row['energy_ev']-r['energy_ev']) < SETTINGS['suppression_ev'] for r in selected):
            row['rejected_reasons'].append('stronger_edge_within_suppression_radius')
        elif len(selected) >= SETTINGS['maximum_components']:
            row['rejected_reasons'].append('fixed_component_count_limit')
        else:
            selected.append(row)
    selected.sort(key=lambda v: v['energy_ev'])
    return selected, candidates, smooth, edge


def _features(e, parameters, tapers):
    """Return per-component log features and their analytic parameter Jacobian."""
    count = len(tapers)
    components = np.empty((count, len(e)))
    jac = np.empty((count, len(e), 4))
    for k, (center, width, q, b) in enumerate(parameters.reshape(count, 4)):
        z = (e-center)/width
        eb = np.exp(b)
        numerator = (z+q)**2+eb
        denominator = z*z+1.
        f = np.log(numerator/denominator)
        dz = 2*(z+q)/numerator-2*z/denominator
        components[k] = tapers[k]*f
        jac[k, :, 0] = tapers[k]*(-dz/width)
        jac[k, :, 1] = tapers[k]*(-dz*z/width)
        jac[k, :, 2] = tapers[k]*(2*(z+q)/numerator)
        jac[k, :, 3] = tapers[k]*(eb/numerator)
    return components, jac


def _fit(e, ratio, selected):
    count, n = len(selected), len(e)
    x = 2*(e-(e[0]+e[-1])/2)/(e[-1]-e[0])
    nuisance_design = np.column_stack((np.ones(n), x, x*x))
    cell = np.gradient(e); weight = np.sqrt(n*cell/cell.sum())
    centers = np.asarray([r['energy_ev'] for r in selected])
    signs = np.asarray([1. if r['source'] == 0 else -1. for r in selected])
    tapers = np.asarray([_taper(e, c) for c in centers]).reshape(count, n)
    nuisance0 = np.linalg.lstsq(nuisance_design, ratio, rcond=None)[0]
    nuisance0 = np.clip(nuisance0, -.49, .49)
    lower, upper, scale = [], [], []
    for center in centers:
        lower.extend([center-SETTINGS['center_radius_ev'], SETTINGS['width_bounds_ev'][0], -3., -6.])
        upper.extend([center+SETTINGS['center_radius_ev'], SETTINGS['width_bounds_ev'][1], 0., 3.])
        scale.extend([.006, .004, 1., 1.])
    lower.extend([-.5]*3); upper.extend([.5]*3); scale.extend([.03]*3)
    lower, upper = np.asarray(lower), np.asarray(upper)
    penalty = SETTINGS['feature_penalty']
    def residual(parameters):
        components, unused = _features(e, parameters[:-3], tapers)
        model = signs@components+nuisance_design@parameters[-3:]
        return np.r_[weight*(model-ratio), penalty*components.ravel()]
    def jacobian(parameters):
        unused, fjac = _features(e, parameters[:-3], tapers)
        jac = np.zeros((n+n*count, 4*count+3))
        for k in range(count):
            jac[:n, k*4:k*4+4] = weight[:, None]*signs[k]*fjac[k]
            jac[n+k*n:n+(k+1)*n, k*4:k*4+4] = penalty*fjac[k]
        jac[:n, -3:] = weight[:, None]*nuisance_design
        return jac
    fits = []
    for width in SETTINGS['starts_width_ev']:
        start = np.r_[np.array([[c, width, -.3, -.3] for c in centers]).ravel(), nuisance0]
        fit = least_squares(residual, start, jac=jacobian, bounds=(lower, upper),
            x_scale=scale, loss='soft_l1', f_scale=SETTINGS['soft_l1_scale'],
            max_nfev=SETTINGS['max_nfev'], ftol=SETTINGS['fit_tolerance'],
            xtol=SETTINGS['fit_tolerance'], gtol=SETTINGS['fit_tolerance'])
        fits.append(fit)
    index = min(range(len(fits)), key=lambda i: (fits[i].cost, np.linalg.norm(fits[i].x)))
    best = fits[index]
    components, fjac = _features(e, best.x[:-3], tapers)
    nuisance = nuisance_design@best.x[-3:]
    model = signs@components+nuisance
    residual_log = ratio-model
    # Scale columns to comparable, declared parameter units for a diagnostic SVD.
    data_jac = jacobian(best.x)[:n]*np.asarray(scale)[None, :]
    singular = np.linalg.svd(data_jac, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    records, applied = [], []
    for k, selected_row in enumerate(selected):
        pars = best.x[k*4:k*4+4]
        distance = np.minimum(pars-lower[k*4:k*4+4], upper[k*4:k*4+4]-pars)/(upper[k*4:k*4+4]-lower[k*4:k*4+4])
        # q=0 admits a symmetric shape, with explicitly ambiguous attribution.
        hard_hits = [j for j in (0, 1, 3) if distance[j] <= SETTINGS['boundary_fraction']]
        if pars[2]-lower[k*4+2] <= SETTINGS['boundary_fraction']*(upper[k*4+2]-lower[k*4+2]):
            hard_hits.append(2)
        local = np.abs(e-selected_row['energy_ev']) <= SETTINGS['plateau_radius_ev']
        local_rms = _rms(residual_log[local])
        threshold = max(SETTINGS['local_residual_floor_log'], SETTINGS['local_residual_target_fraction']*_rms((ratio-nuisance)[local]))
        amplitude = float(np.max(np.abs(components[k])))
        reasons = []
        if not best.success: reasons.append('selected_optimizer_not_converged')
        if amplitude < SETTINGS['minimum_applied_feature_log']: reasons.append('feature_below_application_floor')
        if local_rms > threshold: reasons.append('differential_local_fit_residual')
        if hard_hits: reasons.append('component_parameter_boundary')
        use = not reasons
        applied.append(use)
        records.append(dict(**selected_row, fitted_parameters=pars.tolist(),
            parameter_order=['center_ev', 'width_ev', 'q', 'log_c'],
            normalized_bound_distance=distance.tolist(), hard_bound_indices=hard_hits,
            maximum_feature_log=amplitude, local_residual_rms_log=local_rms,
            local_residual_threshold_log=threshold, applied=use,
            nonapplication_reasons=reasons,
            symmetric_assignment_ambiguous=bool(pars[2] > -.03),
            support_ev=[selected_row['energy_ev']-.05, selected_row['energy_ev']+.05],
            plateau_ev=[selected_row['energy_ev']-.025, selected_row['energy_ev']+.025]))
    return dict(components=components, tapers=tapers, nuisance=nuisance,
        fitted_difference=model, residual=residual_log, applied=np.asarray(applied, bool),
        records=records, parameters=best.x, diagnostic=dict(
            selected_start=index, starts=[dict(width_ev=w, success=bool(f.success),
                status=int(f.status), message=str(f.message), cost=float(2*f.cost),
                optimality=float(f.optimality), nfev=int(f.nfev), parameters=f.x.tolist(),
                active_bounds=f.active_mask.tolist(), unweighted_ratio_rms_log=_rms(residual(f.x)[:n]/weight))
                for w, f in zip(SETTINGS['starts_width_ev'], fits)],
            nuisance_coefficients=best.x[-3:].tolist(),
            scaled_data_jacobian_singular_values=singular.tolist(),
            scaled_data_jacobian_condition=condition,
            differential_residual_rms_log=_rms(residual_log),
            differential_residual_max_log=float(np.max(np.abs(residual_log)))))


def _reliability(e, corrected_logs, records, tapers, fit_residual):
    """Smooth weighting of corrected sources with independently saved flanks."""
    n = len(e)
    severity = np.zeros((2, n)); response_sum=np.zeros(n); response_weight=np.zeros(n)
    anchors=np.zeros((len(records), n),bool); flank_records=[]
    ratio=corrected_logs[0]-corrected_logs[1]
    for k,row in enumerate(records):
        if not row['applied']:
            flank_records.append(dict(component=k, used=False, reason='component_not_applied'))
            continue
        center,width,unused_q,unused_b=row['fitted_parameters']
        delta=e-center
        left=(delta>=-.050)&(delta<=-.030)&(np.abs(fit_residual)<=.010)
        right=(delta>=.030)&(delta<=.050)&(np.abs(fit_residual)<=.010)
        mask=left|right; anchors[k]=mask
        if min(int(left.sum()),int(right.sum())) < SETTINGS['flank_minimum_points_per_side']:
            flank_records.append(dict(component=k,used=False,reason='insufficient_two_sided_corrected_ratio_flanks',
                left_points=int(left.sum()),right_points=int(right.sum())))
            continue
        design=np.column_stack((np.ones(n),delta/.050))
        coefficients=np.linalg.lstsq(design[mask],ratio[mask],rcond=None)[0]
        response=design@coefficients
        amplitude=row['maximum_feature_log']
        score=amplitude**2*tapers[k]**2/(1+((e-center)/width)**2)
        severity[row['source']]+=score
        response_sum+=score*response; response_weight+=score
        flank_records.append(dict(component=k,used=True,left_points=int(left.sum()),
            right_points=int(right.sum()),response_coefficients=coefficients.tolist(),
            response_basis='1,(energy-fitted_center)/0.050eV',
            anchor_rms_log=_rms((ratio-response)[mask]),
            maximum_extrapolated_center_offset_log=float(response[np.argmin(np.abs(delta))])))
    alignment=np.divide(response_sum,response_weight,out=np.zeros(n),where=response_weight>0)
    reliability=1/(1+severity/SETTINGS['reliability_scale_log']**2)
    weights=reliability/reliability.sum(axis=0,keepdims=True)
    return weights,alignment,severity,anchors,flank_records


def generate(e, pn, *, family='local_subtraction'):
    """Return a complete native-grid background and audit arrays, without I/O."""
    if family not in FAMILIES:
        raise ValueError('family must be one of '+repr(FAMILIES))
    e, pn = np.array(e, dtype=float, copy=True), np.array(pn, dtype=float, copy=True)
    if e.ndim != 1 or len(e) < 20 or not np.all(np.isfinite(e)) or not np.all(np.diff(e) > 0):
        raise ValueError('Increasing finite native energy with at least 20 samples required.')
    if pn.shape != (2, len(e)) or not np.all(np.isfinite(pn) & (pn > 0)):
        raise ValueError('Positive finite raw PN with source axis [P,N] required.')
    logs = np.log(pn); ratio = logs[0]-logs[1]
    baseline_log = np.mean(logs, axis=0); baseline = np.exp(baseline_log)
    selected, detections, smoothed_ratio, edge = _detect(e, ratio)
    n = len(e)
    source_features = np.zeros_like(pn)
    uncertain = np.zeros(n, bool); fitted_support = np.zeros(n, bool)
    if selected:
        fit = _fit(e, ratio, selected)
        for k, row in enumerate(fit['records']):
            support = fit['tapers'][k] > 0
            fitted_support |= support
            if row['applied']:
                source_features[row['source']] += fit['components'][k]
            else:
                uncertain |= support
        diagnostics = fit['diagnostic']
        components, component_tapers = fit['components'], fit['tapers']
        component_applied = fit['applied']
        records = fit['records']
        nuisance, fitted_ratio, ratio_residual = fit['nuisance'], fit['fitted_difference'], fit['residual']
    else:
        components = component_tapers = np.empty((0, n))
        component_applied = np.empty(0, bool); records=[]
        nuisance = np.full(n, float(np.median(ratio)))
        fitted_ratio = nuisance.copy(); ratio_residual = ratio-fitted_ratio
        diagnostics=dict(selected_start=None, starts=[], differential_residual_rms_log=_rms(ratio_residual))
    for row in detections:
        if row['rejected_reasons'] and 'stronger_edge_within_suppression_radius' not in row['rejected_reasons']:
            uncertain |= np.abs(e-row['energy_ev']) <= SETTINGS['plateau_radius_ev']
    uncertain |= (np.abs(ratio_residual) > .01) & fitted_support
    primary_source_features = source_features.copy()
    primary_corrected_logs=logs-primary_source_features
    reliability_weights,alignment,severity,flank_masks,flank_records=_reliability(
        e,primary_corrected_logs,records,component_tapers,ratio_residual)
    if family == 'opposite_attribution':
        source_features = -source_features[::-1].copy()
    corrected_logs = logs-source_features
    repair_delta = -.5*np.sum(source_features, axis=0)
    weights=np.full_like(pn,.5)
    if family == 'reliability_weighted':
        weights=reliability_weights
        aligned_logs=corrected_logs+np.array([-.5,.5])[:,None]*alignment[None,:]
        output_log=np.sum(weights*aligned_logs,axis=0)
        repair_delta=output_log-baseline_log
    background = np.exp(baseline_log+repair_delta)
    repair_mask = np.any(source_features != 0., axis=0)
    background[~repair_mask] = baseline[~repair_mask]
    source_residual = corrected_logs-(baseline_log+repair_delta)[None, :]
    reconstructed = np.exp((baseline_log+repair_delta)[None, :]+source_residual+source_features)
    closure = float(np.max(np.abs(reconstructed/pn-1.)))
    diagnostics.update(family=family, status='experimental_current_PN_only_not_validated',
        settings=json.loads(json.dumps(SETTINGS)), source_axis=['P','N'],
        input_hashes=dict(energy=_sha_array(e), pn=_sha_array(pn)),
        baseline='current_raw_geometric_mean_no_global_response_correction',
        source_assignment='ratio_edge_direction_prior; opposite attribution has exactly the same differential model',
        physical_metadata='Same sample and same position; only doping/gate changed, as stated by user.',
        BN_used=False, historical_BN_used=False, sample_used=False,
        detection_candidates=detections, components=records,
        applied_component_count=int(np.sum(component_applied)),
        detected_component_count=len(selected),
        source_reconstruction_max_relative_error=closure,
        source_reconstruction_interpretation='Bookkeeping closure with saved source residual, not independent physical accuracy.',
        differential_closure_max_log=float(np.max(np.abs(fitted_ratio+ratio_residual-ratio))),
        no_repair_identity_exact=bool(np.array_equal(background[~repair_mask],baseline[~repair_mask])),
        common_feature_identifiability='Shared equal resonances are unidentifiable and retained; ratio fit does not prove featureless background.',
        geometry_scope='Common multiplicative changes cancel in ratio; discrete detection can differ at exact numerical thresholds.',
        flank_calibration=flank_records,
        reliability_interpretation='Weights depend on fitted source-local severity, not source fit truth; alignment only uses two-sided corrected-ratio flanks. Insufficient flanks retain equal weights.',
        limitations=['Source-specific envelope/fringe differences are averaged by the declared raw baseline.',
            'Local fitted tails can be broad-response or texture rather than true resonance.',
            'Nonconverged or uncertain components are not repaired.',
            'Opposite attribution is a sensitivity record, never selected by BN or sample performance.',
            'Full dR/R and full d2E remain parallel evaluation conditions.'])
    return dict(energy=e, background=background, raw_geometric_mean=baseline,
        corrected_sources=np.exp(corrected_logs), source_feature_log=source_features,
        primary_source_feature_log=primary_source_features,
        source_weights=weights, weights=weights.copy(),
        reliability_source_weights=reliability_weights, source_severity_log_squared=severity,
        flank_alignment_log=alignment, flank_anchor_masks=flank_masks,
        repair_log_delta=repair_delta, repair_mask=repair_mask, modification_mask=repair_mask.copy(),
        uncertain_mask=uncertain, no_repair_mask=~repair_mask,
        common_feature_unidentifiable_mask=np.ones(n,bool),
        fit_support_mask=fitted_support, ratio_log=ratio,
        smoothed_ratio_log=smoothed_ratio, ratio_edge_per_ev=edge,
        differential_nuisance_log=nuisance, differential_model_log=fitted_ratio,
        differential_residual_log=ratio_residual,
        component_feature_log=components, component_tapers=component_tapers,
        component_applied=component_applied,
        source_residual_log=source_residual, pn_reconstruction=reconstructed,
        seams=np.empty(0), diagnostics=diagnostics)
