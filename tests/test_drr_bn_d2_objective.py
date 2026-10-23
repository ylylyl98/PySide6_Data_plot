"""D2-priority policy behavior, training isolation and fixed measured-BN basis."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq
from scipy.signal import savgol_coeffs

from core.drr_bn_calibration import evaluate_bn_group, _fit, _support
from core.drr_bn_d2_objective import training_bandwidth
from tests.test_drr_bn_calibration import TRUSTED, WIDE_TRUSTED, localized_synthetic, wide_synthetic


def evaluate(values, settings=TRUSTED):
    e, z, b, gates, frames = values
    return evaluate_bn_group(e, z, b, gates=gates, settings=settings, background_repeats=frames)


class D2ObjectiveTests(unittest.TestCase):
    def test_narrow_training_can_improve_display_yet_fail_feature_qualification(self):
        result = evaluate(localized_synthetic())
        self.assertEqual(result.selected_method, 'original')
        candidate = result.report['candidates']['local_shift_gain']
        self.assertTrue(candidate['selection_joint']['d2_benefit_passed'])
        self.assertFalse(result.eligible)
        self.assertEqual(result.applied_method, 'original')
        audit = candidate['training_influence_audit']
        self.assertEqual(len(audit['screening']['probes']), 68)
        self.assertEqual(audit['actual_refits']['count'], 3)
        self.assertFalse(audit['passed'])
        self.assertGreater(audit['actual_refits']['max_drr_error_fraction'], .05)
        self.assertIn('training_feature_influence_not_confirmed', result.reasons)
        self.assertNotIn('final_confirmation_d2', result.report)
        self.assertIsNone(result.report['joint_validation']['influence_audit'])

    def test_bandwidth_guard_uses_actual_nonuniform_grid_and_contiguous_score_segment(self):
        e = np.linspace(1.5, 2.1, 1201)
        e += .014*(e-1.81)**2
        mask = (e >= 1.900) & (e <= 1.935)
        moderate = training_bandwidth(e, mask, 51, 2)
        large = training_bandwidth(e, mask, 101, 2)
        self.assertTrue(moderate['passed'])
        self.assertFalse(large['passed'])
        self.assertAlmostEqual(large['sg_footprint_energy'], 100*np.ptp(e)/(len(e)-1))
        fragmented = np.zeros(len(e), bool)
        fragmented[100:140] = True
        fragmented[200:240] = True
        check = training_bandwidth(e, fragmented, 51, 2)
        self.assertEqual(check['total_score_channel_count'], 80)
        self.assertEqual(check['score_channel_count'], 40)
        self.assertFalse(check['passed'])

    def test_tiny_training_loss_reports_real_baseline_normalization(self):
        values = localized_synthetic()
        zero = np.zeros((2, 9, len(values[0])))
        with patch('core.drr_bn_d2_objective.products', return_value=(zero, zero, 21)):
            result = evaluate(values)
        for method in ('local_shift', 'local_shift_gain'):
            evidence = result.report['candidates'][method]['parameters']['training_objective']
            self.assertEqual(evidence['original_projected_mean_square'], 0.)
            self.assertEqual(evidence['original_normalized_objective'], 0.)
            self.assertGreater(evidence['normalization_mean_square'], 0.)

    def test_unresolved_bn_response_returns_original_without_training_certificate(self):
        values = localized_synthetic()
        empty_response = (np.zeros((2, len(values[0]))), 21)
        with patch('core.drr_bn_d2_objective.production_d2', return_value=empty_response):
            result = evaluate(values)
        self.assertFalse(result.eligible)
        self.assertEqual(result.applied_method, 'original')
        self.assertFalse(result.report['joint_validation']['d2_training_evaluated'])
        for method in ('local_shift', 'local_shift_gain'):
            params = result.report['candidates'][method]['parameters']
            self.assertFalse(params['optimizer_success'])
            self.assertEqual(params['training_objective']['basis_rank'], 0)
            self.assertFalse(params['training_objective']['d2_training_evaluated'])

    def test_collinear_bn_responses_cannot_identify_shift_and_gain(self):
        values = localized_synthetic()
        response = np.sin(np.arange(len(values[0]))*.7)
        identical = (np.stack((response, response)), 21)
        with patch('core.drr_bn_d2_objective.production_d2', return_value=identical):
            e, z, b, g, _ = values
            mask = (e >= TRUSTED.train_region[0]) & (e <= TRUSTED.train_region[1])
            candidate, gain = _fit(z, b, [0, 1], np.arange(len(g)), e, mask,
                _support(e, TRUSTED), TRUSTED, 'local_shift_gain', None)
        self.assertEqual(gain['training_objective']['basis_rank'], 1)
        self.assertFalse(gain['optimizer_success'])
        self.assertFalse(gain['training_objective']['d2_training_evaluated'])
        np.testing.assert_array_equal(candidate, values[2])

    def test_d2_gain_can_accept_when_drr_residual_is_almost_unchanged(self):
        values = list(wide_synthetic())
        # Declared fixed-pattern measurement noise in an SG-null frequency;
        # a numerical neutrality control, not an estimated experimental floor.
        coefficients = savgol_coeffs(21, 2, deriv=2)
        offsets = np.arange(21)-10
        response = lambda omega: float(coefficients@np.cos(offsets*omega))
        grid = np.linspace(.4, 2.8, 2000)
        roots = np.flatnonzero(np.array([response(x) for x in grid[:-1]])*
                              np.array([response(x) for x in grid[1:]]) < 0)
        omega = brentq(response, grid[roots[0]], grid[roots[0]+1])
        values[1] += values[2]*.010*np.sin(omega*np.arange(len(values[0])))
        result = evaluate(values, WIDE_TRUSTED)
        self.assertTrue(result.eligible, result.reasons)
        report = result.report
        selected = report['candidates'][result.selected_method]
        self.assertLess(selected['selection']['mean_improvement_fraction'], .10)
        self.assertGreater(selected['selection_d2']['mean_improvement_fraction'], .50)
        self.assertFalse(selected['selection_joint']['drr_benefit_passed'])
        self.assertTrue(selected['selection_joint']['drr_noninferiority_passed'])
        self.assertEqual(report['optimization_policy'], 'd2_shrinkage_v1')
        self.assertEqual(report['joint_validation']['schema_version'], 3)
        self.assertTrue(report['joint_validation']['d2_training_evaluated'])
        self.assertTrue(report['joint_validation']['fixed_bn_basis'])

    def test_training_evidence_uses_real_sg_and_full_training_repeat_context(self):
        values = wide_synthetic()
        result = evaluate(values, WIDE_TRUSTED)
        self.assertTrue(result.eligible, result.reasons)
        for name in ('local_shift', 'local_shift_gain'):
            evidence = result.report['candidates'][name]['parameters']['training_objective']
            self.assertEqual(evidence['metric'], 'BN-projected production D2 residual')
            self.assertEqual(evidence['basis_source'], 'BN-only shift/gain tangent responses')
            self.assertEqual(evidence['repeat_indices'], [0, 1])
            self.assertEqual(evidence['channel_indices'], list(range(len(values[0]))))
            self.assertEqual(evidence['score_channel_indices'], result.report['scope']['fit_channel_indices'])
            self.assertIn('not an independent energy holdout', evidence['context'])
            self.assertEqual(evidence['used_window_length'], 21)
            self.assertTrue(evidence['drr_noninferiority_passed'])
            self.assertGreater(evidence['evaluations'], 1)
            self.assertEqual(np.asarray(evidence['original_projected_rms']).shape, (2, 9))

    def test_selection_and_final_repeats_never_change_training_or_fixed_basis(self):
        original = list(localized_synthetic())
        before = evaluate(original)
        changed = original[1].copy()
        changed[2:] *= np.exp(.02*np.sin(np.arange(changed.shape[-1])*.7))
        original[1] = changed
        after = evaluate(original)
        for name in ('local_shift', 'local_shift_gain'):
            self.assertEqual(before.report['candidates'][name]['parameters'], after.report['candidates'][name]['parameters'])
            np.testing.assert_array_equal(before.candidate_backgrounds[name], after.candidate_backgrounds[name])

    def test_sg_configuration_actually_changes_training_objective_and_optimum(self):
        values = localized_synthetic()
        short = evaluate(values)
        long = evaluate(values, replace(TRUSTED, derivative_window=101))
        self.assertFalse(long.eligible)
        self.assertEqual(long.applied_method, 'original')
        self.assertIn('insufficient_d2_training_bandwidth', long.reasons)
        for name in ('local_shift', 'local_shift_gain'):
            a = short.report['candidates'][name]['parameters']
            b = long.report['candidates'][name]['parameters']
            self.assertEqual(b['training_objective']['used_window_length'], 101)
            self.assertTrue(b['training_objective']['d2_training_evaluated'])
            self.assertFalse(b['training_objective']['bandwidth_check']['passed'])
            self.assertNotEqual(a['training_objective']['basis_response_norms'], b['training_objective']['basis_response_norms'])
            self.assertNotEqual(a['shift_pixel'], b['shift_pixel'])

    def test_actual_d2_improvement_does_not_permit_drr_deterioration(self):
        e = np.linspace(1.5, 2.1, 1201)
        pixel = np.arange(len(e))
        envelope = (.5-.5*np.cos(np.pi*np.clip((e-1.89)/.01, 0, 1)))
        envelope *= .5-.5*np.cos(np.pi*np.clip((1.98-e)/.01, 0, 1))
        original_signal = envelope*.001*np.sin(pixel*2*np.pi/21)
        damaged_signal = envelope*.010*np.sin(pixel*2*np.pi/60)
        b = np.full(len(e), 1e4)
        z = np.tile(b*(1.02+original_signal), (5, 9, 1))
        candidate = b*(1.02+original_signal)/(1.02+damaged_signal)
        parameters = dict(shift_pixel=.3, fine_gain=1., training_log_rms=0.,
                          near_parameter_bound=False, optimizer_success=True)
        with patch('core.drr_bn_calibration._fit', return_value=(candidate, parameters)):
            result = evaluate((e, z, b, np.linspace(-1, 1, 9), np.stack((b, b))))
        item = result.report['candidates']['local_shift']
        self.assertGreater(item['selection_d2']['mean_improvement_fraction'], .50)
        self.assertLess(item['selection']['mean_improvement_fraction'], -.50)
        self.assertFalse(item['selection_joint']['drr_noninferiority_passed'])
        self.assertFalse(result.eligible)
        self.assertEqual(result.applied_method, 'original')


if __name__ == '__main__':
    unittest.main()
