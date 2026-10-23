"""Behavioral checks for dual-product selection and full-axis fidelity vetoes."""
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from core.drr_bn_calibration import evaluate_bn_group
from core.loader import DataCube
from core.processing import apply_sg_derivative_energy
from tests.test_drr_bn_calibration import TRUSTED, WIDE_TRUSTED, wide_synthetic, localized_synthetic, synthetic


def evaluate(values, settings=TRUSTED):
    e, z, b, gates, reference = values
    return evaluate_bn_group(e, z, b, gates=gates, settings=settings, background_repeats=reference)


def actual_d2(e, gates, z, settings):
    cube = DataCube(e, gates, z, 'Gate (V)', 'Independent QA', 'DR/R')
    result, _ = apply_sg_derivative_energy(cube, derivative=2,
        window_length=settings.derivative_window, polyorder=settings.derivative_polyorder)
    return result.Z


class JointCalibrationTests(unittest.TestCase):
    def test_real_dual_improvement_records_actual_pipeline_and_frozen_final(self):
        for seed in (10, 12):
            with self.subTest(seed=seed):
                values = wide_synthetic(seed=seed)
                result = evaluate(values, WIDE_TRUSTED)
                self.assertTrue(result.eligible, result.reasons)
                self.assertNotEqual(result.applied_method, 'original')
                report = result.report
                self.assertEqual(report['schema_version'], 4)
                self.assertEqual(report['derivative_settings'], dict(window_length=21,
                    polyorder=2, used_window_length=21, processing='production SG energy derivative'))
                joint = report['joint_validation']
                self.assertTrue(joint['passed'])
                self.assertTrue(joint['derivative_checked'])
                self.assertTrue(joint['selection']['drr_benefit_passed'])
                self.assertTrue(joint['selection']['d2_benefit_passed'])
                self.assertTrue(joint['final']['passed'])
                audit = joint['influence_audit']
                self.assertTrue(audit['passed'])
                self.assertEqual(audit['probe_mode'], 'common_to_all_training_repeats_and_gates')
                self.assertEqual(len(audit['screening']['probes']), 68)
                self.assertEqual(audit['actual_refits']['count'], 3)
                self.assertGreater(report['final_confirmation_d2']['mean_improvement_fraction'], .5)
                for method in ('local_shift', 'local_shift_gain'):
                    item = report['candidates'][method]
                    self.assertEqual(set(item['selection']), set(item['selection_d2']))
                    self.assertTrue(item['selection_joint']['protection']['local_d2']['passed'])
                json.dumps(report, allow_nan=False)

    def test_drr_improvement_cannot_compensate_for_derivative_deterioration(self):
        # This deliberately prescribed candidate reduces a broad residual but
        # introduces smaller, faster oscillations. Its actual production D2
        # worsens even though DR/R RMS improves; no aggregate tradeoff may pass.
        e = np.linspace(1.5, 2.1, 1201)
        pixel = np.arange(len(e))
        envelope = (.5-.5*np.cos(np.pi*np.clip((e-1.89)/.01, 0, 1)))
        envelope *= .5-.5*np.cos(np.pi*np.clip((1.98-e)/.01, 0, 1))
        slow = envelope*.004*np.sin(pixel*2*np.pi/60)
        fast = envelope*.0003*np.sin(pixel*2*np.pi/21)
        b = np.full(len(e), 1e4)
        z = np.tile(b*(1.02+slow), (5, 9, 1))
        candidate = b*(1.02+slow)/(1.02+fast)
        parameters = dict(shift_pixel=.3, fine_gain=1., training_log_rms=0.,
                          near_parameter_bound=False, optimizer_success=True)
        with patch('core.drr_bn_calibration._fit', return_value=(candidate, parameters)):
            result = evaluate((e, z, b, np.linspace(-1, 1, 9), np.stack((b, b))))
        item = result.report['candidates']['local_shift']
        self.assertGreater(item['selection']['mean_improvement_fraction'], .4)
        self.assertLess(item['selection_d2']['mean_improvement_fraction'], -.5)
        self.assertFalse(item['selection_joint']['d2_benefit_passed'])
        self.assertFalse(result.eligible)
        self.assertEqual(result.applied_method, 'original')

    def test_gate_not_used_for_training_still_vetoes_shared_bn(self):
        values = list(localized_synthetic())
        values[1][:, 1] = localized_synthetic(shifts=-.30)[1][:, 1]
        result = evaluate(values, replace(TRUSTED, gate_count=3))
        self.assertNotIn(1, result.report['selected_gate_indices'])
        item = result.report['candidates']['local_shift_gain']
        self.assertGreater(item['selection']['mean_improvement_fraction'], .5)
        self.assertFalse(item['selection_joint']['all_gate_drr']['passed'])
        self.assertIn([2, 1], item['selection_joint']['all_gate_drr']['failing_repeat_gate_indices'])
        self.assertFalse(result.eligible)

    def test_local_taper_curvature_damage_vetoes_otherwise_improving_candidate(self):
        result = evaluate(synthetic())
        item = result.report['candidates']['local_shift_gain']
        self.assertGreater(item['selection']['mean_improvement_fraction'], .5)
        self.assertGreater(item['selection_d2']['mean_improvement_fraction'], .5)
        self.assertFalse(item['selection_joint']['protection']['local_d2']['passed'])
        self.assertGreater(item['selection_joint']['protection']['local_d2']['failing_window_gate_count'], 0)
        self.assertFalse(result.eligible)

    def test_large_window_nonuniform_protection_uses_actual_output_and_vetoes(self):
        base_settings = replace(TRUSTED, protected_regions=((1.938, 1.942),))
        values = list(localized_synthetic(peak=(1.940, .0007, .0002),
            protected_regions=base_settings.protected_regions))
        values[0] = values[0]+.02*(values[0]-1.8)**2
        settings = replace(base_settings, derivative_window=101)
        result = evaluate(values, settings)
        e, z, b, gates, _ = values
        protected = (e >= 1.938) & (e <= 1.942)
        old = actual_d2(e, gates, z[2]/b-1, settings)
        self.assertEqual(result.report['derivative_settings']['used_window_length'], 101)
        for method in ('local_shift', 'local_shift_gain'):
            candidate = result.candidate_backgrounds[method]
            new = actual_d2(e, gates, z[2]/candidate-1, settings)
            delta = float(np.max(abs(new[:, protected]-old[:, protected])))
            protection = result.report['candidates'][method]['selection_joint']['protection']['d2']
            self.assertAlmostEqual(protection['max_absolute_change'], delta, places=10)
            self.assertGreater(delta, settings.protected_d2_absolute_tolerance)
            self.assertFalse(protection['passed'])
        self.assertFalse(result.eligible)

    def test_final_d2_veto_does_not_choose_runner_up_or_refit_candidate(self):
        values = list(wide_synthetic())
        accepted = evaluate(values, WIDE_TRUSTED)
        self.assertTrue(accepted.eligible, accepted.reasons)
        original_measurements = values[1].copy()
        values[1][3:] = wide_synthetic(shifts=-.35)[1][3:]
        vetoed = evaluate(values, WIDE_TRUSTED)
        self.assertEqual(vetoed.selected_method, accepted.selected_method)
        self.assertFalse(vetoed.eligible)
        self.assertEqual(vetoed.applied_method, 'original')
        self.assertIn('final_holdout_d2_benefit_not_confirmed', vetoed.reasons)
        self.assertFalse(vetoed.report['joint_validation']['final']['passed'])
        self.assertEqual(accepted.report['joint_validation']['influence_audit'],
                         vetoed.report['joint_validation']['influence_audit'])
        for method in ('local_shift', 'local_shift_gain'):
            np.testing.assert_array_equal(accepted.candidate_backgrounds[method], vetoed.candidate_backgrounds[method])
            self.assertEqual(accepted.report['candidates'][method]['parameters'],
                             vetoed.report['candidates'][method]['parameters'])
        np.testing.assert_array_equal(values[1][:3], original_measurements[:3])

    def test_missing_final_evidence_never_marks_joint_validation_passed(self):
        result = evaluate(localized_synthetic(repeats=3))
        self.assertFalse(result.eligible)
        self.assertFalse(result.report['joint_validation']['passed'])
        self.assertIsNone(result.report['joint_validation']['final'])
        self.assertNotIn('final_confirmation_d2', result.report)


if __name__ == '__main__':
    unittest.main()
