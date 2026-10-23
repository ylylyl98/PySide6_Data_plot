"""V12: smooth weighting of native PN differential fit residual only."""

# Private migrated numerical kernel; original file SHA is provenance, never input.
MIGRATED_FROM_SHA256 = "83d28adcd3f4b9ae110655bacd501c2eabbb5d4e6fa129c9c911034049bf04a8"
import copy
import hashlib
import json
import numpy as np

from . import source_v11 as _v11
V11_SHA256 = _v11.MIGRATED_FROM_SHA256
_sha_array = _v11._sha_array
_rms = _v11._rms
FAMILIES = ('v11_bandpass_binary', 'native_residual_weighted', 'fine_residual_weighted',
            'accepted_residual_weighted')
SETTINGS = dict(exposure_scale_log=.015, residual_lowpass_width_ev=.035,
                source_model='frozen V11 bandpass binary',
                unsupported_components='unchanged; all fitted components excluded from residual intervention',
                BN_used=False, historical_BN_used=False, sample_used=False)


def fit_bundle(e, pn):
    previous = _v11.fit_bundle(e, pn)
    model = previous['variants']['bandpass_binary_full_tail']
    energy = model['e']
    exposure = np.zeros_like(model['pn'])
    for k, row in enumerate(model['records']):
        if not row['applied']:
            continue
        center, width, _, _ = row['parameters']
        strength = np.max(abs(model['components'][k]))
        exposure[row['source']] += strength**2 / (1 + ((energy-center)/width)**2)
    reliability = 1 / (1 + exposure / .015**2)
    weights = reliability / reliability.sum(axis=0, keepdims=True)
    residual = model['raw_ratio'] - model['model']
    slow_residual = _v11._v7._quad(energy, residual, .035)[0]
    return dict(previous=previous, model=model, exposure=exposure,
                weights=weights, residual=residual, slow_residual=slow_residual)


def render(bundle, *, family='native_residual_weighted'):
    if family not in FAMILIES:
        raise ValueError('Unknown V12 family')
    result = _v11.render(bundle['previous'], family='bandpass_binary_full_tail')
    base = result['background'].copy()
    residual = bundle['residual']
    slow_residual = bundle['slow_residual']
    accepted_only = family == 'accepted_residual_weighted'
    if accepted_only:
        model = bundle['model']
        unapplied = np.array([not row['applied'] for row in model['records']], dtype=bool)
        features = np.asarray(model['components'])
        restored = (features[unapplied].sum(axis=0) if unapplied.any()
                    else np.zeros_like(residual))
        # Source subtraction uses only applied components. Using the all-fit
        # model for r would instead put each rejected component into
        # q=(C_P-C_N)-r, transferring half of it into the opposite source.
        residual = residual + restored
        slow_residual = _v11._v7._quad(model['e'], residual, .035)[0]
        result.update(
            v12_all_fitted_differential_model_log=np.asarray(model['model']).copy(),
            v12_unapplied_component_log=restored.copy(),
            differential_model_log=np.asarray(model['model'])-restored,
            differential_residual_log=residual.copy())
    if family == 'v11_bandpass_binary':
        selected = np.zeros_like(base)
        delta = np.zeros_like(base)
        weights = np.full((2, len(base)), .5)
    else:
        selected = residual.copy()
        if family == 'fine_residual_weighted':
            selected -= slow_residual
        weights = bundle['weights'].copy()
        delta = (weights[0]-.5) * selected
        result['background'] = base * np.exp(delta)
    result.update(v11_binary_background=base, source_weights=weights, weights=weights.copy(),
                  residual_source_exposure_log_squared=bundle['exposure'].copy(),
                  residual_source_weights=bundle['weights'].copy(),
                  residual_intervention_native_log=residual.copy(),
                  residual_intervention_slow_log=slow_residual.copy(),
                  residual_intervention_selected_log=selected,
                  residual_intervention_delta_log=delta)
    result['repair_log_delta'] = np.log(result['background']/result['raw_geometric_mean'])
    result['repair_mask'] = abs(result['repair_log_delta']) > 1e-12
    result['modification_mask'] = result['repair_mask'].copy()
    result['no_repair_mask'] = ~result['repair_mask']
    identity = dict(family=family, baseline_construction_sha256=str(result['construction_sha256']),
                    delta_sha256=_sha_array(delta), weights_sha256=_sha_array(weights),
                    raw_residual_sha256=_sha_array(residual))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result['construction_sha256'] = np.array(digest)
    result['diagnostics'].update(family=family, settings=copy.deepcopy(SETTINGS),
        construction=identity, construction_sha256=digest,
        residual_intervention_max_abs_log=float(np.max(abs(delta))),
        residual_intervention_rms_log=_rms(delta),
        output_construction='Frozen binary background times exp((wP-.5)*selected native PN differential residual)',
        source_weights_interpretation='Exposure weights apply only to fitted PN differential residual; they are not proof of featurelessness.',
        raw_PN_or_common_component_smoothed=False,
        fitted_nuisance_substituted_for_background=False,
        unsupported_components_assigned_by_residual=False,
        baseline_family='V11 bandpass_binary_full_tail')
    if accepted_only:
        result['diagnostics']['settings']['unsupported_components'] = (
            'retained in native residual; excluded from shared alignment; '
            'source-subtraction acceptance is unchanged')
        result['diagnostics'].update(
            source_alignment_policy='accepted_only',
            unsupported_components_retained_in_residual=True,
            unapplied_component_count=int(unapplied.sum()),
            unapplied_component_indices=np.flatnonzero(unapplied).tolist(),
            source_weights_interpretation=(
                'Weights act on the measured differential residual including '
                'unapplied fitted components; they do not certify clean sources.'))
    return result


def generate_all(e, pn):
    bundle = fit_bundle(e, pn)
    return {family: render(bundle, family=family) for family in FAMILIES}


def generate(e, pn, *, family='native_residual_weighted'):
    return render(fit_bundle(e, pn), family=family)
