"""Independent raw-data QA for D2-priority, two-parameter BN optimization.

Protected regions are frozen before any injection. Several injected peaks are
deliberately unprotected and in the fitting/scoring spectrum. No fitted result
or application support function enters the data generator.
"""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.optimize import brentq
from scipy.signal import savgol_coeffs
from scipy.special import betainc

from core.drr_bn_calibration import BnCalibrationSettings, evaluate_bn_group
from core.loader import DataCube
from core.processing import apply_sg_derivative_energy


SETTINGS = BnCalibrationSettings(trusted_regions=((1.88, 1.99),), uncertain_regions=(),
                                 protected_regions=((1.58, 1.86),))
INTERMEDIATE_SETTINGS = replace(SETTINGS, core_region=(1.885, 2.035),
                        train_region=(1.885, 1.990), validation_region=(2.005, 2.035),
                        trusted_regions=((1.87, 2.05),))
WIDE_SETTINGS = replace(SETTINGS, core_region=(1.885, 2.305),
                        train_region=(1.885, 2.235), validation_region=(2.265, 2.305),
                        trusted_regions=((1.87, 2.32),))


def sg_null_frequency(window=21):
    """A declared fixed-pattern-noise control, not a general noise estimator."""
    coefficients = savgol_coeffs(window, 2, deriv=2)
    offsets = np.arange(window)-window//2
    response = lambda omega: float(coefficients@np.cos(offsets*omega))
    grid = np.linspace(.40, 2.8, 2000)
    values = np.array([response(omega) for omega in grid])
    pairs = np.flatnonzero(values[:-1]*values[1:] < 0)
    if not len(pairs):
        raise AssertionError('No nonzero SG frequency root found')
    j = pairs[0]
    return brentq(response, grid[j], grid[j+1], xtol=1e-14)


def independent_input(*, nonuniform=False, neutral_noise=0., disagreement=False,
                       opposite_gate=False, final_opposite=False, repeats=5,
                       instrument_shift=.28, fine_gain=1.045, drift_core=(1.900, 1.970), energy_stop=2.10):
    """Analytic local instrument mismatch plus independent measurement texture.

    beta(4,4) edge blending differs from the production quintic. The optional
    SG-null sinusoid is explicitly a deterministic measurement-noise control:
    it dominates raw DR/R while leaving little SG21 curvature. The mismatch
    control gives that frequency the opposite instrument phase in held-out
    repeats, so a shared two-parameter correction can improve D2 while
    damaging raw DR/R despite a noninferior training optimum.
    """
    rng = np.random.default_rng(98172)
    energy = np.linspace(1.50, energy_stop, int(round((energy_stop-1.50)/.0005))+1)
    if nonuniform:
        energy += .014*(energy-1.81)**2
    gates = np.linspace(-1, 1, 9)
    pixel = np.arange(len(energy), dtype=float)
    omega_null = sg_null_frequency()
    frequencies = (2*np.pi/8.7, 2*np.pi/14.2)
    amplitudes = (.004, .0017)
    texture = sum(amplitude*np.sin(omega*pixel+phase)
                  for amplitude, omega, phase in zip(amplitudes, frequencies, (.21, -.47)))
    if disagreement:
        texture += .008*np.sin(omega_null*pixel+.4)
    background = np.exp(10+.24*(energy-1.8)+texture)
    envelope = betainc(4, 4, np.clip((energy-drift_core[0]+.01)/.01, 0, 1))
    envelope *= betainc(4, 4, np.clip((drift_core[1]+.01-energy)/.01, 0, 1))
    measurements, true_references = [], []
    for repeat in range(repeats):
        rows, references = [], []
        for gate_index, gate in enumerate(gates):
            shift = -instrument_shift if ((final_opposite and repeat >= 3) or (opposite_gate and gate_index == 8)) else instrument_shift
            shifted = sum(fine_gain*amplitude*np.sin(omega*(pixel+shift)+phase)
                          for amplitude, omega, phase in zip(amplitudes, frequencies, (.21, -.47)))
            if disagreement:
                null_shift = -.28 if repeat >= 2 else shift
                shifted += .008*np.sin(omega_null*(pixel+null_shift)+.4)
            reference = background*np.exp(envelope*(shifted-texture))
            sample = np.exp(.018+.008*(energy-1.9)+.0014*gate*(energy-1.9)**2)
            sample += neutral_noise*np.sin(omega_null*pixel+.07*gate_index)
            rows.append(reference*(sample+rng.normal(0, 1.5e-5, len(energy))))
            references.append(reference)
        measurements.append(rows)
        true_references.append(references)
    noise = rng.normal(0, 1.5e-5, (20, len(energy)))
    noise -= noise.mean(0)
    return dict(energy=energy, measurements=np.array(measurements), background=background,
                gates=gates, background_repeats=background*(1+noise),
                true_references=np.array(true_references))


def wide_input(**kwargs):
    """More independent BN texture information, restricted to synthetic QA."""
    return independent_input(drift_core=WIDE_SETTINGS.core_region, energy_stop=2.38, **kwargs)


def intermediate_input(**kwargs):
    return independent_input(drift_core=INTERMEDIATE_SETTINGS.core_region, **kwargs)


def inject(inputs, amplitude, center, width, mode='common'):
    result = {name: values.copy() for name, values in inputs.items()}
    energy, gates = inputs['energy'], inputs['gates']
    for index, gate in enumerate(gates):
        if mode == 'single_gate' and index != len(gates)-1:
            continue
        position = center+.0012*gate if mode == 'dispersive' else center
        weight = 1+.2*gate if mode == 'gate_amplitude' else 1.
        signal = amplitude*weight*np.exp(-.5*((energy-position)/width)**2)
        result['measurements'][:, index] *= np.exp(signal)
    return result


def evaluate(inputs, settings=SETTINGS):
    return evaluate_bn_group(inputs['energy'], inputs['measurements'], inputs['background'],
                             gates=inputs['gates'], background_repeats=inputs['background_repeats'],
                             settings=settings)


def d2(energy, gates, rows, window=21):
    cube = DataCube(energy, gates, rows, 'Gate', 'Independent D2-priority QA', 'DR/R')
    output, used = apply_sg_derivative_energy(cube, derivative=2, window_length=window, polyorder=2)
    if used != window:
        raise AssertionError('Unexpected SG clamp')
    return output.Z


def independent_d2_scores(inputs, background, repeat_ids, gate_ids, window=21,
                          validation_region=(1.945, 1.970)):
    energy, gates = inputs['energy'], inputs['gates']
    use = (energy >= validation_region[0]) & (energy <= validation_region[1])
    x = energy[use]
    x = (x-x.mean())/np.ptp(x)
    design = np.column_stack((np.ones(len(x)), x, x*x))
    values = []
    for repeat in repeat_ids:
        curvature = d2(energy, gates, inputs['measurements'][repeat]/background-1, window)
        profiles = curvature[np.asarray(gate_ids)][:, use]
        fit = np.linalg.lstsq(design, profiles.T, rcond=None)[0]
        residual = profiles-(design@fit).T
        values.append(float(np.median(np.sqrt(np.mean(residual**2, axis=1)))))
    return np.array(values)


def counterfactual_metrics(clean, injected, before, after, center, width, window=21, dispersive=False):
    raw_clean, raw_injected = clean['measurements'].mean(0), injected['measurements'].mean(0)
    # The fixed-clean correction is the counterfactual transfer operator.
    # Changes beyond this transfer isolate adaptation of the refitted model.
    known_frozen = (raw_injected-raw_clean)/before.applied_background
    recovered = raw_injected/after.applied_background-raw_clean/before.applied_background
    known_d2 = d2(clean['energy'], clean['gates'], known_frozen, window)
    recovered_d2 = d2(clean['energy'], clean['gates'], recovered, window)
    use = abs(clean['energy']-center) < max(.006 if dispersive else 0., 5*width, .5*window*.0005)
    amplitude_scale = np.max(abs(known_frozen[:, use]))
    curvature_scale = np.max(abs(known_d2[:, use]))
    return dict(drr_fractional_adaptation_error=float(np.max(abs(recovered[:, use]-known_frozen[:, use]))/max(amplitude_scale, 1e-15)),
                d2_fractional_adaptation_error=float(np.max(abs(recovered_d2[:, use]-known_d2[:, use]))/max(curvature_scale, 1e-15)),
                known_drr=known_frozen, recovered_drr=recovered,
                known_d2=known_d2, recovered_d2=recovered_d2)


class IndependentD2PriorityTests(unittest.TestCase):
    def assert_contract(self, result):
        self.assertEqual(result.report['schema_version'], 4)
        self.assertEqual(result.report['optimization_policy'], 'd2_shrinkage_v1')
        self.assertEqual(result.report['joint_validation']['schema_version'], 3)
        self.assertEqual(result.report['joint_validation']['policy'], 'd2_shrinkage_v1')
        json.dumps(result.report, allow_nan=False)
        if result.eligible:
            joint = result.report['joint_validation']
            self.assertTrue(joint['passed'])
            self.assertTrue(joint['selection']['passed'])
            self.assertTrue(joint['final']['passed'])
            audit = joint['influence_audit']
            self.assertTrue(audit['passed'])
            self.assertTrue(audit['training_only'])
            self.assertEqual(audit['train_repeat_indices'], result.report['split']['train'])
            self.assertEqual(audit['screening']['probe_count'], 68)
            self.assertEqual(audit['actual_refits']['count'], 3)
            self.assertEqual(audit['amplitude'], .0002)
            self.assertEqual(audit['sigma_native_spacings'], [1.5, 3])
            self.assertEqual(audit['signs'], [-1, 1])
            self.assertEqual(audit['schema_version'], 3)
            self.assertEqual(audit['policy'], 'training_feature_influence_shrinkage_v2')
            refinement = audit['refinement']
            self.assertTrue(refinement['evaluated'])
            self.assertTrue(refinement['passed'])
            self.assertEqual(refinement['policy'], 'native_feature_grid_v1')
            self.assertEqual(refinement['probe_count'], refinement['expected_probe_count'])
            self.assertGreater(refinement['probe_count'], 0)
            self.assertLessEqual(refinement['probe_count'], 168)
            sensitivity = joint['selection_sensitivity_audit']
            self.assertTrue(sensitivity['passed'])
            self.assertTrue(sensitivity['selection_aware'])
            self.assertFalse(sensitivity['training_only'])
            self.assertFalse(sensitivity['final_repeats_read'])
            self.assertGreaterEqual(result.report['final_confirmation_d2']['mean_improvement_fraction'], .10)

    def test_drr_neutrality_and_d2_gain_cannot_override_real_local_damage(self):
        inputs = wide_input(neutral_noise=.010)
        result = evaluate(inputs, WIDE_SETTINGS)
        self.assert_contract(result)
        self.assertFalse(result.eligible)
        self.assertLess(result.report['final_confirmation']['mean_improvement_fraction'], SETTINGS.min_improvement_fraction)
        self.assertGreaterEqual(result.report['final_confirmation_d2']['mean_improvement_fraction'], .10)
        final_drr = result.report['final_confirmation']
        before = np.asarray(final_drr['original_rms'])
        after = np.asarray(final_drr['candidate_rms'])
        allowance = np.maximum(SETTINGS.drr_noninferiority_absolute, SETTINGS.drr_noninferiority_fraction*before)
        self.assertTrue(np.all(after-before <= allowance))
        self.assertIn('training_full_axis_fidelity_not_confirmed', result.reasons)
        np.testing.assert_array_equal(result.applied_background, result.original_background)

    def test_no_drift_and_one_adverse_gate_do_not_claim_shared_correction(self):
        for inputs in (independent_input(instrument_shift=0., fine_gain=1.), independent_input(opposite_gate=True)):
            with self.subTest(no_drift=np.array_equal(inputs['true_references'][0, 0], inputs['background'])):
                result = evaluate(inputs)
                self.assertFalse(result.eligible)
                np.testing.assert_array_equal(result.applied_background, inputs['background'])

    def test_d2_priority_does_not_override_actual_drr_deterioration(self):
        result = evaluate(independent_input(disagreement=True))
        self.assert_contract(result)
        self.assertFalse(result.eligible)
        self.assertEqual(result.applied_method, 'original')
        # At least one generated D2-fit candidate must expose the specific raw
        # DR/R failure; otherwise the model-mismatch challenge was ineffective.
        checks = [item['selection_joint'] for name, item in result.report['candidates'].items() if name != 'original']
        self.assertTrue(any(item['d2_benefit_passed'] and not item['all_gate_drr']['passed'] for item in checks))

    def test_counterfactual_refits_preserve_unprotected_common_and_gate_peaks(self):
        clean = independent_input(nonuniform=True)
        before = evaluate(clean)
        self.assert_contract(before)
        # Regions are fixed once above. Injections never add protection masks.
        for amplitude, center, width, mode in ((.0002, 1.920, .0010, 'common'),
                                               (.0300, 1.920, .0010, 'common'),
                                               (.0002, 1.957, .0007, 'dispersive'),
                                               (.0020, 1.927, .0010, 'single_gate')):
            with self.subTest(amplitude=amplitude, center=center, mode=mode):
                injected = inject(clean, amplitude, center, width, mode)
                after = evaluate(injected)
                self.assert_contract(after)
                metrics = counterfactual_metrics(clean, injected, before, after, center, width, dispersive=mode == 'dispersive')
                if after.eligible:
                    self.assertLess(metrics['drr_fractional_adaptation_error'], .05)
                    self.assertLess(metrics['d2_fractional_adaptation_error'], .05)
                else:
                    # Rejected fits cannot flatten a peak: the applied output
                    # uses the exact original denominator. A mode change from
                    # a clean accepted fit is reported separately from signal
                    # transfer and is not called successful correction.
                    np.testing.assert_array_equal(after.applied_background, clean['background'])

    def test_wider_information_domain_accepts_and_retains_unprotected_weak_peak(self):
        clean = wide_input(nonuniform=True)
        injected = inject(clean, .0002, 1.920, .0010)
        before, after = evaluate(clean, WIDE_SETTINGS), evaluate(injected, WIDE_SETTINGS)
        self.assertTrue(before.eligible, before.reasons)
        self.assertTrue(after.eligible, after.reasons)
        metrics = counterfactual_metrics(clean, injected, before, after, 1.920, .0010)
        self.assertLess(metrics['drr_fractional_adaptation_error'], .05)
        self.assertLess(metrics['d2_fractional_adaptation_error'], .05)

    def test_narrow_band_sg21_alpha1_absorption_risk_remains_visible_under_shrinkage(self):
        clean = independent_input(nonuniform=True)
        injected = inject(clean, .0002, 1.9162815698437503, .00074979)
        before, after = evaluate(clean), evaluate(injected)
        self.assert_contract(before)
        self.assert_contract(after)
        # Inspect the un-applied family previews to prove that original-BN
        # fallback did not merely make the adversarial counterfactual zero.
        previews = [SimpleNamespace(applied_background=result.candidate_backgrounds['local_shift_gain'])
                    for result in (before, after)]
        metrics = counterfactual_metrics(clean, injected, *previews, 1.9162815698437503, .00074979)
        self.assertGreater(max(metrics['drr_fractional_adaptation_error'], metrics['d2_fractional_adaptation_error']), .05)
        # The frozen alpha=1 risk is retained above. A newly qualified smaller
        # fixed alpha must meet the same 5% physical-output threshold after a
        # complete refit and selection, rather than inheriting alpha=1's veto
        # or obtaining a zero error by applying the original BN.
        if after.eligible:
            self.assertNotEqual(after.applied_method, 'original')
            strength = after.report['candidates'][after.applied_method]['parameters']['correction_strength']
            self.assertLess(strength, 1.)
            applied_metrics = counterfactual_metrics(clean, injected, before, after,
                                                     1.9162815698437503, .00074979)
            self.assertLess(applied_metrics['drr_fractional_adaptation_error'], .05)
            self.assertLess(applied_metrics['d2_fractional_adaptation_error'], .05)
        else:
            np.testing.assert_array_equal(after.applied_background, clean['background'])

    def test_protected_peak_sg101_uses_actual_nonuniform_pipeline(self):
        clean = independent_input(nonuniform=True)
        injected = inject(clean, .0002, 1.75, .0007, 'gate_amplitude')
        settings = replace(SETTINGS, derivative_window=101)
        before, after = evaluate(clean, settings), evaluate(injected, settings)
        self.assert_contract(before)
        self.assert_contract(after)
        self.assertEqual(after.report['derivative_settings']['used_window_length'], 101)
        metrics = counterfactual_metrics(clean, injected, before, after, 1.75, .0007, 101)
        self.assertLess(metrics['d2_fractional_adaptation_error'], 1e-8)

    def test_bandwidth_policy_and_nearby_unprotected_edge_peaks(self):
        clean = independent_input(nonuniform=True)
        for window in (21, 51, 101):
            settings = replace(SETTINGS, derivative_window=window)
            before = evaluate(clean, settings)
            for center, width in ((1.897, .0005), (1.899, .0007), (1.901, .0010)):
                with self.subTest(window=window, center=center, width=width):
                    injected = inject(clean, .0002, center, width)
                    after = evaluate(injected, settings)
                    self.assert_contract(before)
                    self.assert_contract(after)
                    self.assertEqual(after.report['derivative_settings']['used_window_length'], window)
                    if window == 101:
                        self.assertFalse(after.eligible, 'Insufficient training bandwidth must veto automatic application')
                    if after.eligible:
                        metrics = counterfactual_metrics(clean, injected, before, after, center, width, window)
                        self.assertLess(metrics['drr_fractional_adaptation_error'], .05)
                        self.assertLess(metrics['d2_fractional_adaptation_error'], .05)
                    else:
                        np.testing.assert_array_equal(after.applied_background, clean['background'])

    def test_full_axis_edge_pseudofeature_control_is_not_accepted(self):
        from tests.test_drr_bn_calibration import synthetic
        energy, values, background, gates, frames = synthetic()
        result = evaluate_bn_group(energy, values, background, gates=gates, settings=SETTINGS,
                                   background_repeats=frames)
        self.assertFalse(result.eligible)
        np.testing.assert_array_equal(result.applied_background, background)

    def test_confirmation_metric_is_actual_d2_not_training_projection_loss(self):
        inputs = wide_input(nonuniform=True)
        snapshots = {name: array.copy() for name, array in inputs.items()}
        result = evaluate(inputs, WIDE_SETTINGS)
        self.assertTrue(result.eligible, result.reasons)
        ids = result.report['split']['final']
        gate_ids = result.report['selected_gate_indices']
        before = independent_d2_scores(inputs, inputs['background'], ids, gate_ids,
                                       validation_region=WIDE_SETTINGS.validation_region)
        after = independent_d2_scores(inputs, result.applied_background, ids, gate_ids,
                                      validation_region=WIDE_SETTINGS.validation_region)
        comparison = result.report['final_confirmation_d2']
        np.testing.assert_allclose(before, comparison['original_rms'], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(after, comparison['candidate_rms'], rtol=1e-10, atol=1e-9)
        for name, expected in snapshots.items():
            np.testing.assert_array_equal(inputs[name], expected)

    def test_selection_and_final_repeats_cannot_change_training_optima(self):
        clean = independent_input()
        before = evaluate(clean)
        changed = inject(clean, .04, 1.918, .0008)
        changed['measurements'][:2] = clean['measurements'][:2]
        after = evaluate(changed)
        for method in ('local_shift', 'local_shift_gain'):
            self.assertEqual(before.report['candidates'][method]['parameters'], after.report['candidates'][method]['parameters'])
            np.testing.assert_array_equal(before.candidate_backgrounds[method], after.candidate_backgrounds[method])

    def test_training_scope_discloses_full_sg_context_and_bn_only_fixed_basis(self):
        clean = independent_input(nonuniform=True)
        changed = inject(clean, .004, 1.939, .0008)
        changed['measurements'][2:] = clean['measurements'][2:]
        settings = replace(SETTINGS, derivative_window=101)
        before, after = evaluate(clean, settings), evaluate(changed, settings)
        changed_parameters = False
        for method in ('local_shift', 'local_shift_gain'):
            parameters = before.report['candidates'][method]['parameters']
            other_parameters = after.report['candidates'][method]['parameters']
            evidence = parameters['training_objective']
            other = other_parameters['training_objective']
            self.assertEqual(evidence['repeat_indices'], [0, 1])
            self.assertEqual(evidence['channel_indices'], list(range(len(clean['energy']))))
            self.assertEqual(evidence['score_channel_indices'], before.report['scope']['fit_channel_indices'])
            self.assertIn('not an independent energy holdout', evidence['context'])
            self.assertTrue(evidence['fixed_bn_basis'])
            self.assertIn('BN-only', evidence['basis_source'])
            for key in ('basis_rank', 'basis_singular_values', 'basis_response_norms'):
                self.assertEqual(evidence[key], other[key])
            changed_parameters |= ((parameters['shift_pixel'], parameters['fine_gain']) !=
                                   (other_parameters['shift_pixel'], other_parameters['fine_gain']))
        self.assertTrue(changed_parameters, 'This control should expose the declared full-SG training context')

    def test_final_repeat_change_can_only_veto_frozen_selection(self):
        before = evaluate(wide_input(nonuniform=True), WIDE_SETTINGS)
        after = evaluate(wide_input(nonuniform=True, final_opposite=True), WIDE_SETTINGS)
        self.assert_contract(before)
        self.assert_contract(after)
        self.assertTrue(before.eligible, before.reasons)
        self.assertEqual(before.selected_method, after.selected_method)
        self.assertFalse(after.eligible)
        self.assertEqual(before.report['joint_validation']['influence_audit'],
                         after.report['joint_validation']['influence_audit'])
        self.assertEqual(before.report['joint_validation']['selection_sensitivity_audit'],
                         after.report['joint_validation']['selection_sensitivity_audit'])
        for method in ('local_shift', 'local_shift_gain'):
            self.assertEqual(before.report['candidates'][method]['selection_d2'], after.report['candidates'][method]['selection_d2'])
            np.testing.assert_array_equal(before.candidate_backgrounds[method], after.candidate_backgrounds[method])


if __name__ == '__main__':
    unittest.main()
