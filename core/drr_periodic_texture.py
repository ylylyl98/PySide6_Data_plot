"""Conservative, held-out prediction of periodic response across short gaps.

Only anchor values enter fitting or detrending.  Reproducibility is checked in
disjoint available-gate groups; this is not an experimental-repeat check and
does not identify a stationary sample feature as instrumental.  A prediction
is returned only after independently refitted models predict predefined
contiguous anchor bands at least as wide as the requested gap. The general
family uses two 34 meV bands. A fixed single energy sinusoid is attempted only
after that family fails, and must predict two gap-matched and two wider outer
bands. Long gaps and ambiguous models retain the caller's background policy.
"""
import warnings
import numpy as np
from scipy.optimize import minimize
from core.load_cancellation import check_load_cancelled


_MAX_WINDOWS = 12
_MAX_WINDOW_EV = .24
_MIN_PERIOD_EV = .003
_MAX_PERIOD_EV = .016


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _pooled(values, mask, minimum):
    supported = mask.sum(axis=0) >= minimum
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        profile = np.nanmedian(np.where(mask, values, np.nan), axis=0)
    return profile, supported & np.isfinite(profile)


def _nuisance(mask):
    """Separate quadratic trends on contiguous training bands; never fill gaps."""
    columns = []
    usable = np.zeros(len(mask), bool)
    for lo, hi in _runs(mask):
        if hi-lo < 10:
            continue
        usable[lo:hi] = True
        part = np.zeros((len(mask), 3))
        part[lo:hi] = np.polynomial.legendre.legvander(np.linspace(-1, 1, hi-lo), 2)
        columns.append(part)
    if not columns:
        return usable, np.empty((0, 0))
    return usable, np.column_stack(columns)[usable]


def _wave(x, frequency, chirp):
    phase = 2*np.pi*(frequency*x + chirp*x*x)
    return np.column_stack((np.sin(phase), np.cos(phase)))


def _fit(x, values, anchors, span_eV, maximum_components, chirped):
    """Select frequencies using training values only, including on CV calls."""
    check_load_cancelled()
    valid, nuisance = _nuisance(anchors & np.isfinite(values))
    count = int(valid.sum())
    if count < max(45, nuisance.shape[1]+24):
        return None
    xx, yy = x[valid], values[valid]
    base = nuisance
    qbase = np.linalg.qr(base, mode='reduced')[0]
    initial = yy-qbase@(qbase.T@yy)
    initial_ss = float(initial@initial)
    if initial_ss <= count*(.00035**2):
        return None
    lower = max(4., span_eV/_MAX_PERIOD_EV)
    # Missing anchors do not change the native spectral sampling. Using their
    # count here would exclude a valid frequency whenever a CV band is removed.
    native_step = np.diff(x)
    native_quadratic_step = np.diff(x*x)
    upper = min(span_eV/_MIN_PERIOD_EV, 1/(5*np.max(native_step)))
    if upper-lower < 2:
        return None
    frequencies = np.arange(lower, upper, .35)
    chirps = np.array([-5., -2.5, 0., 2.5, 5.]) if chirped else np.array([0.])
    candidates = np.array([(f, q) for f in frequencies for q in chirps if abs(q) <= .35*f])
    native_phase_step = candidates[:, :1]*native_step + candidates[:, 1:]*native_quadratic_step
    candidates = candidates[np.max(abs(native_phase_step), axis=1) <= .2+1e-12]
    components = []
    previous_ss = initial_ss
    for _ in range(maximum_components):
        check_load_cancelled()
        qbase = np.linalg.qr(base, mode='reduced')[0]
        residual = yy-qbase@(qbase.T@yy)
        possible = np.ones(len(candidates), bool)
        for f, _q in components:
            possible &= abs(candidates[:, 0]-f) >= 1.75
        proposed = candidates[possible]
        if not len(proposed):
            break
        phase = 2*np.pi*(proposed[:, :1]*xx + proposed[:, 1:]*xx*xx)
        sin, cos = np.sin(phase), np.cos(phase)
        sin -= (sin@qbase)@qbase.T
        cos -= (cos@qbase)@qbase.T
        ss, cc, sc = np.sum(sin*sin, axis=1), np.sum(cos*cos, axis=1), np.sum(sin*cos, axis=1)
        sy, cy = sin@residual, cos@residual
        determinant = ss*cc-sc*sc
        score = np.divide(cc*sy*sy + ss*cy*cy - 2*sc*sy*cy, determinant,
                          out=np.zeros_like(ss), where=determinant > 1e-8)
        seed = proposed[int(np.argmax(score))]

        def loss(parameters):
            check_load_cancelled()
            f = float(parameters[0])
            q = float(parameters[1]) if chirped else 0.
            if (abs(q) > .35*f
                    or np.max(abs(f*native_step+q*native_quadratic_step)) > .2+1e-12
                    or any(abs(f-old_f) < 1.5 for old_f, _ in components)):
                return initial_ss*10
            waves = _wave(xx, f, q)
            projected = waves-qbase@(qbase.T@waves)
            coefficients = np.linalg.lstsq(projected, residual, rcond=None)[0]
            error = residual-projected@coefficients
            return float(error@error)

        bounds = [(max(lower, seed[0]-.7), min(upper, seed[0]+.7))]
        if chirped:
            bounds.append((max(-6.5, seed[1]-2.5), min(6.5, seed[1]+2.5)))
        fitted = minimize(loss, seed if chirped else seed[:1], method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=45, ftol=1e-10, gtol=1e-9))
        chosen = (float(fitted.x[0]), float(fitted.x[1]) if chirped else 0.)
        new_ss = loss(fitted.x)
        if previous_ss-new_ss < .03*initial_ss:
            break
        added = _wave(xx, *chosen)
        trial = np.column_stack((base, added))
        coefficients = np.linalg.lstsq(trial, yy, rcond=None)[0]
        if np.linalg.norm(coefficients[-2:]) < .00035:
            break
        components.append(chosen)
        base = trial
        previous_ss = new_ss
    if not components:
        return None
    coefficients = np.linalg.lstsq(base, yy, rcond=None)[0][nuisance.shape[1]:]
    prediction = sum((_wave(x, *item)@coefficients[2*i:2*i+2] for i, item in enumerate(components)), np.zeros(len(x)))
    ratio = float(np.sqrt(previous_ss/initial_ss))
    return dict(prediction=prediction, components=components, coefficients=coefficients,
                ratio=ratio, points=count,
                score=float(previous_ss/max(1, count-nuisance.shape[1]-len(components)*(4 if chirped else 3))))


def _coordinates(energy, index):
    return {'energy': energy, 'wavelength': -1/energy, 'pixel': index.astype(float)}


def _normalize(coordinate):
    return (coordinate-coordinate[0])/(coordinate[-1]-coordinate[0])-.5


def _select_model(coordinates, values, support, span, maximum_components):
    """Choose coordinate, phase model and frequencies from this training set."""
    candidates = []
    for coordinate, samples in coordinates.items():
        xx = _normalize(samples)
        fixed = _fit(xx, values, support, span, maximum_components, False)
        if fixed is not None:
            candidates.append((fixed['score'], coordinate, False, fixed))
        varying = _fit(xx, values, support, span, maximum_components, True)
        if varying is not None and (fixed is None or varying['score'] < .8*fixed['score']):
            candidates.append((varying['score']*1.05, coordinate, True, varying))
    return min(candidates, key=lambda value: value[0])[1:] if candidates else None


def _correlation(a, b):
    a, b = a-a.mean(), b-b.mean()
    divisor = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/divisor) if divisor > 1e-16 else 0.


def _test_error(observed, predicted):
    # The test band's own low-order trend affects scoring only, never training,
    # frequency selection, coefficients, or the prediction in the target gap.
    design = np.polynomial.legendre.legvander(np.linspace(-1, 1, len(observed)), 2)
    actual = observed-design@np.linalg.lstsq(design, observed, rcond=None)[0]
    model = predicted-design@np.linalg.lstsq(design, predicted, rcond=None)[0]
    scale = np.linalg.norm(actual)
    if scale <= np.sqrt(len(actual))*.0003:
        return None
    return dict(residual_rms_ratio=float(np.linalg.norm(actual-model)/scale), correlation=_correlation(actual, model))


def _holdouts(energy, support, target_center, width):
    possible = []
    for lo, hi in _runs(support):
        if hi-lo < 22 or energy[hi-1]-energy[lo] < width + .004:
            continue
        center = (energy[lo]+energy[hi-1])/2
        left = int(np.searchsorted(energy, center-width/2))
        right = int(np.searchsorted(energy, center+width/2, side='right'))
        left, right = max(lo, left), min(hi, right)
        if right-left < 16 or energy[right-1]-energy[left] < width-2*np.max(np.diff(energy)):
            continue
        # Place one held-out band on each side where available.
        possible.append((center < target_center, abs(center-target_center), left, right))
    possible.sort(key=lambda value: value[1])
    chosen = []
    for side in (True, False):
        options = [item for item in possible if item[0] == side]
        if options:
            chosen.append(options[0][2:])
    return chosen if len(chosen) == 2 else []


def _edge_holdouts(energy, support, target_center, width):
    """Two outer bands fixed by support geometry, never by prediction scores."""
    runs = _runs(support)
    left = [(lo, hi) for lo, hi in runs if energy[hi-1] < target_center]
    right = [(lo, hi) for lo, hi in runs if energy[lo] > target_center]
    if not left or not right:
        return []
    blocks = []
    for (lo, hi), is_left in ((left[0], True), (right[-1], False)):
        center = energy[lo]+width/2+.002 if is_left else energy[hi-1]-width/2-.002
        a = int(np.searchsorted(energy, center-width/2))
        b = int(np.searchsorted(energy, center+width/2, side='right'))
        if a < lo or b > hi or b-a < 16:
            return []
        blocks.append((a, b))
    return blocks


def _select_energy_one(coordinates, values, support, span, _maximum_components):
    model = _fit(_normalize(coordinates['energy']), values, support, span, 1, False)
    return ('energy', False, model) if model is not None else None


def _validate_family(family, coordinates, yy, valid, span, maximum_components,
                     ee, center, gap_width, gap_edges, profiles):
    """Evaluate one predefined family; failed test bands are never relocated."""
    select = _select_model if family == 'general_periodic' else _select_energy_one
    result = dict(family=family, status='Not validated', validation_blocks=[])
    selected = select(coordinates, yy, valid, span, maximum_components)
    if selected is None:
        result['status'] = 'No resolved periodic component'
        return result, None, []
    coordinate, chirped, model = selected
    result.update(coordinate=coordinate, chirped=chirped, training_residual_rms_ratio=model['ratio'])
    if model['ratio'] > .80 or np.max(abs(model['prediction'])) > .02:
        result['status'] = 'Weak or excessive periodic model'
        return result, None, []
    gap_with_margin = gap_width+2*float(np.max(np.diff(ee)))
    if family == 'general_periodic':
        requested = max(gap_with_margin, .034)
        geometry = [('central_34meV', requested, _holdouts(ee, valid, center, requested))]
    else:
        matched = max(gap_with_margin, .012)
        wider = max(gap_with_margin, .024)
        geometry = [('gap_matched', matched, _holdouts(ee, valid, center, matched)),
                    ('outer_wider', wider, _edge_holdouts(ee, valid, center, wider))]
    result['validation_geometry'] = [dict(family=name, requested_width_eV=width, available_blocks=len(blocks))
                                     for name, width, blocks in geometry]
    if any(len(blocks) != 2 for _, _, blocks in geometry):
        result['status'] = 'Incomplete predefined holdout geometry'
        return result, None, []
    widths = [float(ee[j-1]-ee[i]) for _, _, blocks in geometry for i, j in blocks]
    if min(widths) < gap_width:
        result['status'] = 'Holdout width does not cover the requested gap'
        return result, None, []
    passed = True
    for validation_family, _, blocks in geometry:
        for left, right in blocks:
            check_load_cancelled()
            training = valid.copy()
            training[left:right] = False
            selected_test = select(coordinates, yy, training, span, maximum_components)
            fitted = None if selected_test is None else selected_test[2]
            score = None if fitted is None else _test_error(yy[left:right], fitted['prediction'][left:right])
            validation_x = None if fitted is None else _normalize(coordinates[selected_test[0]])
            cycles = None if fitted is None else min(
                abs(f*(validation_x[right-1]-validation_x[left])
                    + q*(validation_x[right-1]**2-validation_x[left]**2))
                for f, q in fitted['components'])
            record = dict(validation_family=validation_family,
                          energy_eV=[float(ee[left]), float(ee[right-1])], width_eV=float(ee[right-1]-ee[left]),
                          independent_frequency_selection=True, independent_model_selection=True,
                          coordinate=None if selected_test is None else selected_test[0],
                          chirped=None if selected_test is None else selected_test[1], metrics=score,
                          minimum_phase_cycles=None if cycles is None else float(cycles))
            internal_gaps = [(i, j) for i, j in _runs(~training) if i and j < len(ee)]
            record['maximum_training_gap_width_eV'] = max((float(ee[j-1]-ee[i]) for i, j in internal_gaps), default=0.)
            record['passed'] = bool(score is not None and score['residual_rms_ratio'] <= .75
                                    and score['correlation'] >= .70 and cycles >= 2)
            result['validation_blocks'].append(record)
            passed &= record['passed']
    if not passed:
        result['status'] = 'Contiguous holdout prediction failed'
        return result, None, []
    independent = [select(coordinates, profile, valid, span, maximum_components) for profile in profiles]
    if any(value is None for value in independent):
        result['status'] = 'Independent gate-group fit unavailable'
        return result, None, []
    first, second = [value[2]['prediction'] for value in independent]
    region = (ee >= gap_edges[0]-.010) & (ee <= gap_edges[1]+.010)
    average = (first+second)/2
    group_ratio = float(np.linalg.norm((first-second)[region])/max(np.linalg.norm(average[region]), 1e-16))
    group_corr = _correlation(first[region], second[region])
    result['gate_group_validation'] = dict(relative_prediction_difference=group_ratio, correlation=group_corr)
    if group_ratio > .35 or group_corr < .90:
        result['status'] = 'Independent gate-group predictions disagree'
        return result, None, []
    result['status'] = 'Applied'
    return result, selected, widths


def _taper(distance, width=.005):
    t = np.clip(distance/width, 0, 1)
    return t*t*t*(10+t*(-15+6*t))


def _guard_long_gaps(energy, support, weight, validated_width):
    guarded = weight.copy()
    for lo, hi in _runs(~support):
        if energy[hi-1]-energy[lo] > validated_width:
            left = float((energy[lo-1]+energy[lo])/2) if lo else float(energy[0])
            right = float((energy[hi-1]+energy[hi])/2) if hi < len(energy) else float(energy[-1])
            guarded *= _taper(np.maximum(left-energy, energy-right))
    return guarded


def periodic_texture(energy, residual, anchors, *, maximum_components=3):
    """Return a validated log correction, coverage and JSON-safe diagnostics.

    Coverage is zero throughout any gap wider than the narrowest independently
    predicted validation band. The input arrays are never modified. The
    caller may combine this optional prediction with its shared-texture result.
    """
    check_load_cancelled()
    energy = np.asarray(energy, float)
    residual = np.asarray(residual, float)
    supplied = np.asarray(anchors)
    if (energy.ndim != 1 or residual.ndim != 2 or residual.shape[1] != len(energy)
            or supplied.shape != residual.shape or supplied.dtype != bool):
        raise ValueError('Periodic response requires a 1D energy axis and matching residual/boolean anchor matrices.')
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0) or np.any(np.diff(energy) <= 0):
        raise ValueError('Periodic response requires distinct increasing positive energy coordinates.')
    if isinstance(maximum_components, bool) or int(maximum_components) != maximum_components or not 1 <= int(maximum_components) <= 5:
        raise ValueError('Periodic response supports one to five components.')
    maximum_components = int(maximum_components)
    zero = np.zeros(len(energy))
    report = dict(status='Not validated', corrected_fraction=0., coordinate=None, components=[],
                  validated_gap_width_eV=0., maximum_requested_gap_width_eV=0.,
                  maximum_training_span_eV=0., maximum_held_out_width_eV=0., maximum_training_gap_width_eV=0.,
                  supported_regions_eV=[], windows=[], rejection_counts={},
                  validation=dict(method='Independent available-gate groups; general family uses two 34 meV holdouts, energy-one fallback requires two gap-matched plus two wider outer holdouts; all frequencies reselected from training anchors only',
                                  experimental_repeats=False),
                  caveat='Prediction validation does not distinguish stationary sample resonances from instrumental response. Keep peak protection and compare correction off.')

    def rejected(reason):
        report.update(status=reason, reason=reason, rejection_reason=reason)
        return zero, zero.copy(), report

    count, size = residual.shape
    if count < 12 or size < 80:
        return rejected('Insufficient gates or energy samples')
    anchors = supplied & np.isfinite(residual)
    rank = np.cumsum(anchors, axis=0)
    half = anchors.sum(axis=0)//2
    minimum = max(6, int(np.ceil(.15*count)))
    profiles, supports = [], []
    for group in (anchors & (rank <= half), anchors & (rank > half)):
        profile, support = _pooled(residual, group, minimum)
        profiles.append(profile)
        supports.append(support)
    support = supports[0] & supports[1]
    if support.sum() < 90:
        return rejected('Insufficient independent anchor support')
    gaps = [(lo, hi) for lo, hi in _runs(~support) if lo > 0 and hi < size]
    if not gaps:
        return rejected('No interior unsupported gap to predict')
    edges = np.r_[energy[0], (energy[:-1]+energy[1:])/2, energy[-1]]
    report['maximum_requested_gap_width_eV'] = float(max(edges[hi]-edges[lo] for lo, hi in gaps))
    # Prioritize short gaps that can plausibly be tested using nearby anchors.
    gaps.sort(key=lambda pair: edges[pair[1]]-edges[pair[0]])
    accum, weights, coverage = zero.copy(), zero.copy(), zero.copy()
    completed = 0
    for lo, hi in gaps:
        check_load_cancelled()
        gap_width = float(edges[hi]-edges[lo])
        center = float((edges[lo]+edges[hi])/2)
        item = dict(gap_eV=[float(edges[lo]), float(edges[hi])], gap_width_eV=gap_width,
                    status='Not validated', validation_blocks=[])
        report['windows'].append(item)
        if gap_width > .08:
            item.update(status='Gap exceeds bounded local validation width', validated_gap_width_eV=0.)
            continue
        if completed >= _MAX_WINDOWS:
            item['status'] = 'Local validation budget reached'
            continue
        margin = max(.060, gap_width*1.5)
        a = max(0, int(np.searchsorted(energy, center-min(_MAX_WINDOW_EV/2, gap_width/2+margin))))
        b = min(size, int(np.searchsorted(energy, center+min(_MAX_WINDOW_EV/2, gap_width/2+margin), side='right')))
        ee, valid = energy[a:b], support[a:b]
        if len(ee) < 90 or valid.sum() < 75:
            item['status'] = 'Insufficient local anchor support'
            continue
        span = float(ee[-1]-ee[0])
        item['training_span_eV'] = span
        report['maximum_training_span_eV'] = max(report['maximum_training_span_eV'], span)
        yy = (profiles[0][a:b]+profiles[1][a:b])/2
        completed += 1
        coordinates = _coordinates(ee, np.arange(a, b))
        item['family_attempts'] = []
        selected = None
        for family in ('general_periodic', 'energy_one'):
            attempt, candidate, widths = _validate_family(
                family, coordinates, yy, valid, span, maximum_components,
                ee, center, gap_width, (edges[lo], edges[hi]), [profile[a:b] for profile in profiles])
            item['family_attempts'].append(attempt)
            for record in attempt['validation_blocks']:
                report['maximum_training_gap_width_eV'] = max(
                    report['maximum_training_gap_width_eV'], record['maximum_training_gap_width_eV'])
                report['maximum_held_out_width_eV'] = max(report['maximum_held_out_width_eV'], record['width_eV'])
            item.update(attempt)
            if candidate is not None:
                selected = candidate
                break
        if selected is None:
            continue
        coordinate, chirped, model = selected
        # Return the prediction only near this validated gap, with a C2 blend
        # into the caller's supported shared texture. No long-range extrapolation.
        distance = np.minimum(ee-(edges[lo]-.010), (edges[hi]+.010)-ee)
        weight = _taper(distance)
        weight = _guard_long_gaps(ee, valid, weight, min(widths))
        prediction = model['prediction']
        accum[a:b] += weight*prediction
        weights[a:b] += weight
        # A maximum of smooth windows has a derivative corner where they
        # cross. The probabilistic-union identity preserves their C2 joins.
        coverage[a:b] += (1-coverage[a:b])*weight
        components = [dict(frequency_cycles_per_span=float(f), chirp_cycles_per_span2=float(q),
                           amplitude_log=float(np.linalg.norm(model['coefficients'][2*i:2*i+2])),
                           coordinate_range=[float(coordinates[coordinate][0]), float(coordinates[coordinate][-1])])
                      for i, (f, q) in enumerate(model['components'])]
        item.update(status='Applied', components=components, component_count=len(components),
                    validated_gap_width_eV=float(min(widths)))
        report['validated_gap_width_eV'] = max(report['validated_gap_width_eV'], float(min(widths)))
        report['components'].extend(components)
    applied = [item for item in report['windows'] if item['status'] == 'Applied']
    for item in report['windows']:
        if item['status'] != 'Applied':
            counts = report['rejection_counts']
            counts[item['status']] = counts.get(item['status'], 0)+1
    if not applied:
        categories = sorted(report['rejection_counts'].items(), key=lambda item: (-item[1], item[0]))[:2]
        reason = '; '.join(f'{name} ({count})' for name, count in categories)
        result = rejected('No locally validated periodic prediction')
        report.update(reason=f"No locally validated periodic prediction: {reason}", rejection_reason=reason)
        return result
    texture = np.divide(accum, weights, out=zero.copy(), where=weights > 0)*coverage
    report.update(status='Applied', corrected_fraction=float(np.mean(coverage > .5)),
                  coordinate=sorted({item['coordinate'] for item in applied}),
                  component_count=sum(item['component_count'] for item in applied),
                  supported_regions_eV=[[float(edges[lo]), float(edges[hi])] for lo, hi in _runs(coverage > 0)],
                  reason=None, rejection_reason=None)
    check_load_cancelled()
    return texture, coverage, report
