"""Conservative measured-reference fine texture in short shared-response gaps.

The caller explicitly selects and identifies the independent reference. Raw
log sample and reference are calibrated symmetrically; only the reference's
anchor-masked fine texture is returned. No sample protected value enters a
fit. Each left/right holdout and its +/- one-pixel shifts must pass separately.
"""
from collections import Counter
import numpy as np

from core.drr_background_structure import gate_anchor_masks, mask_regions, pooled_profile, shared_texture
from core.load_cancellation import check_load_cancelled

_MAX_GAP_EV = .025
_MAX_CORRECTION = .02
_MAX_WINDOWS = 12


def _rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def _correlation(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.
    return float(np.corrcoef(a, b)[0, 1])


def _smoothstep(value):
    t = np.clip(value, 0, 1)
    return t*t*t*(10+t*(-15+6*t))


def _solve(normal, rhs):
    return np.linalg.solve(normal, rhs[..., None])[..., 0]


def _calibrate(energy, values, anchors, template, reference_coverage, train, context):
    """Project out separate gate cubics, then fit one amplitude and shift.

    All calibration values are selected with the training mask before matrix
    operations. Keeping the raw-log reference avoids an asymmetric high-pass
    transfer function being absorbed by the nuisance trend or amplitude.
    """
    check_load_cancelled()
    columns = np.flatnonzero((energy >= context[0]) & (energy <= context[1]))
    x = (energy[columns]-np.mean(context))/(np.ptp(context)/2)
    polynomial = np.polynomial.legendre.legvander(x, 3)
    mask = anchors[:, columns] & train[columns] & (reference_coverage[columns] > .95)
    normal = np.einsum('gn,ni,nj->gij', mask, polynomial, polynomial)
    usable = (mask.sum(axis=1) >= 35) & (np.linalg.cond(normal) < 1e8)
    if usable.sum() < max(12, int(np.ceil(.5*len(values)))):
        return None
    rows = np.flatnonzero(usable)
    mask, normal = mask[rows], normal[rows]
    y = np.where(mask, values[np.ix_(rows, columns)], 0.)
    coefficient_y = _solve(normal, np.einsum('gn,ni->gi', y, polynomial))
    residual_y = y-coefficient_y@polynomial.T
    ids = np.arange(len(energy))
    best = None
    for shift in np.linspace(-1., 1., 21):
        check_load_cancelled()
        shifted_coverage = np.interp(ids+shift, ids, reference_coverage)
        if np.any(mask & (shifted_coverage[columns] <= .95)):
            continue
        shifted = np.interp(ids+shift, ids, template)
        t = shifted[columns]
        coefficient_t = _solve(normal, np.einsum('gn,n,ni->gi', mask, t, polynomial))
        residual_t = t-coefficient_t@polynomial.T
        denominator = np.sum(mask*residual_t*residual_t)
        if denominator < mask.sum()*1e-12:
            continue
        amplitude = float(np.sum(mask*residual_y*residual_t)/denominator)
        if not .5 <= amplitude <= 1.5:
            continue
        score = float(np.sum(mask*(residual_y-amplitude*residual_t)**2)/mask.sum())
        if best is None or score < best['training_mse']:
            best = dict(training_mse=score, amplitude=amplitude, shift_pixels=float(shift),
                        columns=columns, rows=rows,
                        trend=(coefficient_y-amplitude*coefficient_t)@polynomial.T,
                        prediction=amplitude*t, gate_training_count=mask.sum(axis=1).tolist())
    return best


def _score(energy, values, anchors, fitted, test):
    """Score oscillations after linear removal; never use that fit to predict."""
    columns, rows = fitted['columns'], fitted['rows']
    mask = anchors[np.ix_(rows, columns)] & test[columns]
    target = values[np.ix_(rows, columns)]-fitted['trend']
    prediction = np.broadcast_to(fitted['prediction'], target.shape)
    block = (float(energy[test][0]), float(energy[test][-1]))
    x = np.polynomial.legendre.legvander((energy[columns]-np.mean(block))/np.ptp(block), 1)
    normal = np.einsum('gn,ni,nj->gij', mask, x, x)
    usable = (mask.sum(axis=1) >= 10) & (np.linalg.cond(normal) < 1e8)
    if usable.sum() < max(12, int(np.ceil(.5*len(values)))):
        return dict(passed=False, reason='Insufficient heldout gate support', energy_eV=list(block))
    mask, target, prediction, normal = mask[usable], target[usable], prediction[usable], normal[usable]
    # Zero values outside heldout anchors even in this scoring-only operation.
    ty = target-_solve(normal, np.einsum('gn,ni->gi', np.where(mask, target, 0.), x))@x.T
    tp = prediction-_solve(normal, np.einsum('gn,ni->gi', np.where(mask, prediction, 0.), x))@x.T
    target_rms, prediction_rms = _rms(ty[mask]), _rms(tp[mask])
    if target_rms < 1e-12:
        return dict(passed=False, reason='No heldout oscillatory signal', energy_eV=list(block))
    error = _rms((ty-tp)[mask])
    ratio, correlation = error/target_rms, _correlation(ty[mask], tp[mask])
    raw_error = _rms((target-prediction)[mask])
    return dict(energy_eV=list(block), actual_span_eV=float(np.ptp(energy[test])),
                spectral_points=int(test.sum()), valid_gate_pixels=int(mask.sum()),
                amplitude=fitted['amplitude'], shift_pixels=fitted['shift_pixels'],
                raw_prediction_error_rms_log=raw_error,
                raw_error_over_heldout_oscillation_rms=raw_error/target_rms,
                oscillatory_residual_ratio=ratio, oscillatory_correlation=correlation,
                reference_rms_log=prediction_rms,
                low_order_error_energy_fraction=1-(error/max(raw_error, 1e-15))**2,
                passed=bool(test.sum() >= 15 and ratio <= .4 and correlation >= .9
                            and prediction_rms >= .0005),
                parameters_refitted_without_heldout_values=True, score_only_linear_detrending=True)


def _holdout(energy, supported, lo, hi, width):
    eligible = [(a, b) for a, b in mask_regions(energy, supported & (energy >= lo) & (energy <= hi))
                if b-a >= width]
    if not eligible:
        return None
    left, right = max(eligible, key=lambda pair: pair[1]-pair[0])
    center = (left+right)/2
    a = max(0, np.searchsorted(energy, center-width/2, side='right')-1)
    b = min(len(energy), np.searchsorted(energy, center+width/2, side='left')+1)
    return a, b


def reference_texture(energy, raw_log_sample, sample_anchors, reference_energy,
                      reference_log_frames, shared_coverage):
    """Return (weighted_log_texture, reference_coverage, diagnostics).

    Both energy grids must be exactly equal and increasing. The returned
    texture includes its own C2 coverage, but NOT (1-shared_coverage): the
    caller combines shared + (1-shared_coverage)*reference_texture, and may
    use (1-reference_coverage) to taper a lower-priority periodic prediction.
    No input array is modified; mismatched grids are never interpolated.
    """
    check_load_cancelled()
    energy = np.asarray(energy, float)
    reference_energy = np.asarray(reference_energy, float)
    values = np.asarray(raw_log_sample, float)
    reference = np.asarray(reference_log_frames, float)
    anchors = np.asarray(sample_anchors)
    shared = np.asarray(shared_coverage, float)
    if (energy.ndim != 1 or len(energy) < 24 or not np.isfinite(energy).all()
            or np.any(np.diff(energy) <= 0) or reference_energy.shape != energy.shape
            or not np.array_equal(reference_energy, energy)):
        raise ValueError('Reference texture requires exactly matching increasing energy grids.')
    if (values.ndim != 2 or values.shape[1] != len(energy) or anchors.shape != values.shape
            or anchors.dtype != bool or not np.isfinite(values[anchors]).all()
            or reference.ndim != 2 or reference.shape[1] != len(energy)
            or not np.isfinite(reference).all()):
        raise ValueError('Reference texture needs finite log intensities and matching boolean sample anchors.')
    if shared.shape != energy.shape or not np.isfinite(shared).all() or np.any((shared < 0) | (shared > 1)):
        raise ValueError('Shared coverage must match the energy grid and lie between zero and one.')
    zero = np.zeros(len(energy))
    report = dict(status='Rejected', corrected_fraction=0., windows=[], maximum_gap_width_eV=_MAX_GAP_EV,
                  maximum_correction_log=_MAX_CORRECTION,
                  calibration='Raw log sample versus raw log reference; per-gate cubic nuisance; '
                              'common amplitude [0.5,1.5] and +/-1 native pixel shift; training anchors only',
                  validation='Both contiguous heldouts and their +/-1 native pixel shifts must all pass; '
                             'span >= gap width; oscillatory ratio <=0.4 and correlation >=0.9; '
                             'heldout linear detrending is scoring only')
    def rejected(reason):
        report['reason'] = reason
        return zero, zero.copy(), report
    if len(values) < 12 or len(reference) < 24:
        return rejected('Insufficient sample gates or independent reference frames')
    with np.errstate(over='ignore', under='ignore'):
        reference_raw = np.exp(reference)
    if not np.isfinite(reference_raw).all() or np.any(reference_raw <= 0):
        raise ValueError('Reference log intensities must represent finite positive intensities.')
    reference_anchors = gate_anchor_masks(energy, reference_raw)
    fine, reference_coverage, extraction = shared_texture(energy, reference, reference_anchors)
    report['reference_extraction'] = extraction
    if not np.any(reference_coverage > .95):
        return rejected('No fully supported reference fine texture')
    template, template_support = pooled_profile(reference, reference_anchors,
                                               minimum=max(6, int(np.ceil(.3*len(reference)))))
    template = np.where(template_support, np.nan_to_num(template), 0.)
    reference_coverage = reference_coverage*template_support
    half = len(reference)//2
    first = shared_texture(energy, reference[:half], reference_anchors[:half])
    second = shared_texture(energy, reference[half:], reference_anchors[half:])
    sample_support = anchors.sum(axis=0) >= max(6, int(np.ceil(.3*len(values))))
    joint = sample_support & (reference_coverage > .95)
    missing = mask_regions(energy, shared < .95)
    texture, coverage = zero.copy(), zero.copy()
    candidates = mask_regions(energy, (shared < .95) & (reference_coverage > .95))
    for lo, hi in candidates[:_MAX_WINDOWS]:
        check_load_cancelled()
        if hi-lo < .003:
            continue
        context = (max(float(energy[0]), lo-.065), min(float(energy[-1]), hi+.065))
        window = dict(gap_eV=[lo, hi], gap_width_eV=hi-lo, context_eV=list(context),
                      status='Rejected', validation_blocks=[])
        report['windows'].append(window)
        enclosing = next((b-a for a, b in missing if a <= lo+1e-12 and b >= hi-1e-12), hi-lo)
        window['enclosing_shared_gap_width_eV'] = enclosing
        if enclosing > _MAX_GAP_EV+1e-12:
            window['reason'] = 'Enclosing shared gap exceeds 25 meV'
            continue
        common = ((energy >= context[0]) & (energy <= context[1])
                  & (first[1] > .95) & (second[1] > .95))
        correlation = _correlation(first[0][common], second[0][common])
        difference = (_rms((first[0]-second[0])[common])/
                      max(_rms((first[0]+second[0])[common]/2), 1e-12)) if common.any() else 1.
        window['reference_half_frames'] = dict(common_points=int(common.sum()), correlation=correlation,
                                               relative_difference=difference)
        if common.sum() < 35 or correlation < .95 or difference > .2:
            window['reason'] = 'Independent reference frame halves disagree'
            continue
        available = joint & (energy >= context[0]) & (energy <= context[1])
        width = max(.012, hi-lo)
        blocks = [_holdout(energy, joint, context[0], lo-.002, width),
                  _holdout(energy, joint, hi+.002, context[1], width)]
        if any(block is None for block in blocks):
            window['reason'] = 'No left and right contiguous holdouts at least as wide as the fill'
            continue
        valid = True
        for side, (a, b) in zip(('left', 'right'), blocks):
            for shift in (-1, 0, 1):
                check_load_cancelled()
                test = np.zeros(len(energy), bool)
                if a+shift < 0 or b+shift > len(energy):
                    valid = False
                    window['validation_blocks'].append(dict(side=side, center_shift_pixels=shift,
                                                            passed=False, reason='Holdout outside measured grid'))
                    continue
                test[a+shift:b+shift] = True
                if not np.all(joint[test]) or np.ptp(energy[test])+1e-12 < hi-lo:
                    valid = False
                    window['validation_blocks'].append(dict(side=side, center_shift_pixels=shift,
                                                            passed=False, reason='Incomplete holdout support or width'))
                    continue
                fitted = _calibrate(energy, values, anchors, template, reference_coverage, available & ~test, context)
                score = (_score(energy, values, anchors, fitted, test) if fitted is not None else
                         dict(passed=False, reason='Training-only reference calibration rejected'))
                score.update(side=side, center_shift_pixels=shift)
                window['validation_blocks'].append(score)
                valid &= score['passed']
        if not valid:
            window['reason'] = 'At least one predefined shifted holdout failed'
            continue
        fitted = _calibrate(energy, values, anchors, template, reference_coverage, available, context)
        if fitted is None:
            window['reason'] = 'Final training-only calibration rejected'
            continue
        ids = np.arange(len(energy))
        shifted = np.interp(ids+fitted['shift_pixels'], ids, fine)
        shifted_coverage = np.interp(ids+fitted['shift_pixels'], ids, reference_coverage)
        weight = _smoothstep((energy-lo)/.005)*_smoothstep((hi-energy)/.005)
        region = weight > 0
        prediction = fitted['amplitude']*shifted
        if np.any(shifted_coverage[region] <= .95):
            window['reason'] = 'Shifted reference lacks complete support in the proposed fill'
            continue
        if not region.any() or np.max(abs(prediction[region])) > _MAX_CORRECTION:
            window['reason'] = 'Calibrated reference texture exceeds 0.02 log correction'
            continue
        texture += (1-coverage)*weight*prediction
        coverage += (1-coverage)*weight
        window.update(status='Applied', amplitude=fitted['amplitude'], shift_pixels=fitted['shift_pixels'],
                      validated_gap_width_eV=min(v['actual_span_eV'] for v in window['validation_blocks']),
                      maximum_prediction_log=float(np.max(abs(prediction[region]))))
    if np.any(coverage > 0):
        report.update(status='Applied', corrected_fraction=float(np.mean(coverage > .5)),
                      supported_regions_eV=mask_regions(energy, coverage > .5))
        return texture, coverage, report
    reasons = Counter(w.get('reason', 'Rejected') for w in report['windows'])
    report['rejection_counts'] = dict(reasons)
    return rejected('; '.join(f'{reason} ({count})' for reason, count in reasons.items())
                    or 'No short shared gap supported by the reference')
