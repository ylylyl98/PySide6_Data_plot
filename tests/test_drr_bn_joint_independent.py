"""Independent public-pipeline BN checks with known signal and nuisance truth.

The positive example is a declared model-matched local drift, not evidence of
generalization. Negative examples retain global drift and adverse individual
gates: a lower median RMS is insufficient evidence of spectral fidelity.
"""
from dataclasses import replace
import json
import unittest

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.special import betainc

from core.drr_bn_calibration import evaluate_bn_group
from core.loader import DataCube
from core.processing import apply_sg_derivative_energy
from tests.test_drr_bn_calibration import TRUSTED, synthetic
from tests.test_drr_bn_d2_priority_independent import WIDE_SETTINGS, wide_input


def production_d2(energy, gates, drr, window=21):
    cube = DataCube(energy, gates, drr, 'Gate (V)', 'Independent injection QA', 'DR/R')
    result, used = apply_sg_derivative_energy(cube, derivative=2, window_length=window, polyorder=2)
    if used != window:
        raise AssertionError('The requested derivative configuration was clamped')
    return result.Z


def local_analytic_input(amplitude=0., *, nonuniform=False, gate_peak=False, final_opposite=False,
                         envelope_order=3, periods=(7.3, 12.1), peak_center=1.75):
    """Independent local reference drift with exact injected sample peak.

    A beta CDF gives a C2 local drift envelope without importing a production
    support function. The nominal BN texture family is declared in the data
    generator; no fitted candidate or validation result enters this generator.
    """
    rng = np.random.default_rng(2077)
    energy = np.linspace(1.5, 2.1, 1201)
    if nonuniform:
        energy = energy + .02*(energy-1.8)**2
    gates = np.linspace(-1., 1., 9)
    pixel = np.arange(len(energy), dtype=float)
    log_bn = 10 + .3*(energy-1.8) + .004*np.sin(pixel*2*np.pi/periods[0]) + .002*np.cos(pixel*2*np.pi/periods[1])
    bn = np.exp(log_bn)
    texture = log_bn-savgol_filter(log_bn, 101, 3)
    spline = CubicSpline(pixel, texture)
    envelope = betainc(envelope_order, envelope_order, np.clip((energy-1.89)/.01, 0, 1))
    envelope *= betainc(envelope_order, envelope_order, np.clip((1.98-energy)/.01, 0, 1))
    # This narrow peak is in the declared protected spectrum, far enough from
    # support for either SG21 or SG101. Separate near-core tests stress tails.
    gaussian = np.exp(-.5*((energy-peak_center)/.0007)**2)
    measurements = []
    for repeat in range(5):
        shift = -.30 if final_opposite and repeat >= 3 else .30
        nuisance = envelope*(1.07*spline(np.clip(pixel+shift, 0, len(pixel)-1))-texture)
        rows = []
        for gate in gates:
            sample_log = .02 + .01*(energy-1.9) + .002*gate*(energy-1.9)**2
            sample_log += amplitude*(1+.2*gate if gate_peak else 1)*gaussian
            rows.append(bn*np.exp(sample_log+nuisance+rng.normal(0, 2e-5, len(energy))))
        measurements.append(rows)
    noise = rng.normal(0, 2e-5, (20, len(energy)))
    noise -= noise.mean(0)
    return energy, np.asarray(measurements), bn, gates, bn*(1+noise)


def evaluate(inputs, settings=TRUSTED):
    energy, values, bn, gates, frames = inputs
    return evaluate_bn_group(energy, values, bn, gates=gates, settings=settings, background_repeats=frames)


def wide_legacy_input(**kwargs):
    kwargs.setdefault('nonuniform', True)
    values = wide_input(**kwargs)
    return tuple(values[key] for key in ('energy', 'measurements', 'background', 'gates', 'background_repeats'))


class IndependentJointPipelineTests(unittest.TestCase):
    def assert_joint_contract(self, result, *, window=21):
        self.assertEqual(result.report['schema_version'], 4)
        self.assertEqual(result.report['optimization_policy'], 'd2_shrinkage_v1')
        self.assertEqual(result.report['derivative_settings']['window_length'], window)
        self.assertEqual(result.report['derivative_settings']['polyorder'], 2)
        json.dumps(result.report, allow_nan=False)
        if result.eligible:
            joint = result.report['joint_validation']
            self.assertTrue(joint['passed'])
            self.assertTrue(joint['derivative_checked'])
            self.assertTrue(joint['selection']['passed'])
            self.assertTrue(joint['final']['passed'])
            self.assertEqual(result.report['final_confirmation_d2']['repeat_indices'], [3, 4])
            self.assertEqual(result.report['derivative_settings']['used_window_length'], window)
            self.assertNotEqual(result.applied_method, 'original')

    def test_known_local_drift_is_accepted_after_actual_drr_and_d2_checks(self):
        inputs = wide_legacy_input()
        snapshots = [array.copy() for array in inputs]
        result = evaluate(inputs, WIDE_SETTINGS)
        self.assertTrue(result.eligible, result.reasons)
        self.assert_joint_contract(result)
        for actual, expected in zip(inputs, snapshots):
            np.testing.assert_array_equal(actual, expected)
        for role in ('selection', 'final'):
            check = result.report['joint_validation'][role]
            self.assertTrue(check['all_gate_drr']['passed'])
            self.assertTrue(check['all_gate_d2']['passed'])
            self.assertTrue(check['protection']['local_d2']['passed'])

    def test_full_refit_injection_recovers_strong_weak_and_gate_dependent_peaks(self):
        for amplitude, nonuniform, window in ((.07, False, 21), (.0002, True, 21), (.0002, True, 101)):
            with self.subTest(amplitude=amplitude, nonuniform=nonuniform, window=window):
                settings = replace(TRUSTED, derivative_window=window)
                clean = local_analytic_input(nonuniform=nonuniform)
                injected = local_analytic_input(amplitude, nonuniform=nonuniform, gate_peak=True)
                # Both public calls redo training, candidate selection and final
                # confirmation. No candidate is borrowed from the clean fit.
                before, after = evaluate(clean, settings), evaluate(injected, settings)
                self.assert_joint_contract(before, window=window)
                self.assert_joint_contract(after, window=window)
                energy, values, bn, gates, _ = injected
                protected = (energy >= 1.745) & (energy <= 1.755)
                expected = (values.mean(0)-clean[1].mean(0))/bn
                recovered = values.mean(0)/after.applied_background-clean[1].mean(0)/before.applied_background
                np.testing.assert_allclose(recovered[:, protected], expected[:, protected], rtol=0, atol=5e-16)
                expected_d2 = production_d2(energy, gates, expected, window)
                recovered_d2 = production_d2(energy, gates, recovered, window)
                scale = np.max(abs(expected_d2[:, protected]), axis=1)
                error = np.max(abs(recovered_d2[:, protected]-expected_d2[:, protected]), axis=1)
                self.assertTrue(np.all(error/scale < 1e-8), error/scale)

    def test_different_envelope_frequency_and_peak_position_retain_known_signal(self):
        clean = local_analytic_input(nonuniform=True, envelope_order=4, periods=(8.9, 15.7), peak_center=1.73)
        injected = local_analytic_input(.0002, nonuniform=True, gate_peak=True,
                                       envelope_order=4, periods=(8.9, 15.7), peak_center=1.73)
        before, after = evaluate(clean), evaluate(injected)
        self.assert_joint_contract(before)
        self.assert_joint_contract(after)
        # This family differs from the fitted quintic taper. Conservative
        # fallback is permitted and is not claimed as correction success.
        energy, values, bn, gates, _ = injected
        protected = (energy >= 1.725) & (energy <= 1.735)
        expected = (values.mean(0)-clean[1].mean(0))/bn
        recovered = values.mean(0)/after.applied_background-clean[1].mean(0)/before.applied_background
        expected_d2 = production_d2(energy, gates, expected)
        recovered_d2 = production_d2(energy, gates, recovered)
        scale = np.max(abs(expected_d2[:, protected]))
        error = np.max(abs(recovered_d2[:, protected]-expected_d2[:, protected]))
        self.assertLess(error/scale, 1e-8)
        if not after.eligible:
            np.testing.assert_array_equal(after.applied_background, bn)

    def test_full_refit_near_core_peak_and_sg101_require_actual_protection(self):
        settings = replace(TRUSTED, protected_regions=((1.938, 1.942),), derivative_window=101)
        clean = local_analytic_input(nonuniform=True, peak_center=1.94)
        injected = local_analytic_input(.0002, nonuniform=True, gate_peak=True, peak_center=1.94)
        before, after = evaluate(clean, settings), evaluate(injected, settings)
        self.assert_joint_contract(before, window=101)
        self.assert_joint_contract(after, window=101)
        energy, values, bn, gates, _ = injected
        protected = (energy >= 1.938) & (energy <= 1.942)
        expected = (values.mean(0)-clean[1].mean(0))/bn
        recovered = values.mean(0)/after.applied_background-clean[1].mean(0)/before.applied_background
        expected_d2 = production_d2(energy, gates, expected, 101)
        recovered_d2 = production_d2(energy, gates, recovered, 101)
        self.assertLess(np.max(abs(recovered_d2[:, protected]-expected_d2[:, protected])) /
                        np.max(abs(expected_d2[:, protected])), 1e-8)
        if after.eligible:
            self.assertTrue(after.report['joint_validation']['selection']['protection']['d2']['passed'])
            self.assertTrue(after.report['joint_validation']['final']['protection']['d2']['passed'])
        else:
            np.testing.assert_array_equal(after.applied_background, bn)

    def test_unsampled_adverse_gate_is_not_hidden_by_median_rms_improvement(self):
        inputs = list(local_analytic_input())
        adverse = local_analytic_input(final_opposite=True)
        # Keep the adverse gate outside the fixed fitting subset, so training
        # noninferiority cannot remove this challenge before the independent
        # all-gate check. The majority still supplies a favorable median fit.
        inputs[1] = inputs[1].copy()
        inputs[1][:, 7] = adverse[1][3, 7]
        result = evaluate(inputs, replace(TRUSTED, gate_count=5))
        self.assertEqual(result.report['selected_gate_indices'], [0, 2, 4, 6, 8])
        self.assertNotIn(7, result.report['selected_gate_indices'])
        self.assertFalse(result.eligible)
        challenge_seen = False
        for name, candidate in result.report['candidates'].items():
            if name == 'original':
                continue
            if candidate['selection']['mean_improvement_fraction'] > .50:
                challenge_seen = True
                self.assertFalse(candidate['selection_joint']['passed'])
                gate_failures = candidate['selection_joint']['all_gate_d2']['failing_repeat_gate_indices']
                self.assertTrue(any(gate == 7 for _, gate in gate_failures), gate_failures)
        self.assertTrue(challenge_seen, 'The negative control must retain an apparent aggregate RMS benefit')

    def test_drr_improvement_with_taper_edge_d2_pseudofeatures_is_rejected(self):
        inputs = synthetic()
        energy, values, bn, gates, _ = inputs
        result = evaluate(inputs)
        self.assertFalse(result.eligible)
        candidate = result.candidate_backgrounds['local_shift_gain']
        old_d2 = production_d2(energy, gates, values.mean(0)/bn-1)
        new_d2 = production_d2(energy, gates, values.mean(0)/candidate-1)
        edge = (abs(energy-1.90) < .006) | (abs(energy-1.97) < .006)
        # This independent pointwise peak check demonstrates the challenge;
        # the implementation uses its own full-axis local-window guard.
        self.assertGreater(np.max(abs(new_d2[:, edge])), 2*np.max(abs(old_d2[:, edge])))
        evidence = result.report['candidates']['local_shift_gain']
        self.assertGreater(evidence['selection']['mean_improvement_fraction'], .50)
        self.assertFalse(evidence['selection_joint']['protection']['local_d2']['passed'])
        np.testing.assert_array_equal(result.applied_background, bn)

    def test_final_data_can_veto_but_cannot_refit_or_choose_another_candidate(self):
        before = evaluate(wide_legacy_input(), WIDE_SETTINGS)
        after = evaluate(wide_legacy_input(final_opposite=True), WIDE_SETTINGS)
        self.assertTrue(before.eligible, before.reasons)
        self.assertFalse(after.eligible)
        self.assertEqual(before.selected_method, after.selected_method)
        for method in ('local_shift', 'local_shift_gain'):
            self.assertEqual(before.report['candidates'][method]['parameters'], after.report['candidates'][method]['parameters'])
            self.assertEqual(before.report['candidates'][method]['selection_d2'], after.report['candidates'][method]['selection_d2'])
            self.assertEqual(before.report['candidates'][method]['selection_joint'], after.report['candidates'][method]['selection_joint'])
            np.testing.assert_array_equal(before.candidate_backgrounds[method], after.candidate_backgrounds[method])
        self.assertFalse(after.report['joint_validation']['final']['passed'])
        self.assertIn('final_holdout_joint_not_confirmed', after.reasons)
        np.testing.assert_array_equal(after.applied_background, after.original_background)


if __name__ == '__main__':
    unittest.main()
