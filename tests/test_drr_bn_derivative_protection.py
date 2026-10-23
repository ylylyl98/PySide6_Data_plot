"""Check actual derivative settings and conservative protection decisions.

Every case runs the public fitting/selection/confirmation pipeline. Candidate
inspection independently demonstrates why unchanged native DR/R values alone
cannot certify downstream derivative fidelity.
"""
import unittest
from dataclasses import replace

import numpy as np

from core.drr_bn_calibration import evaluate_bn_group
from core.loader import DataCube
from core.processing import apply_sg_derivative_energy
from tests.test_drr_bn_calibration import TRUSTED, synthetic


def derivative(energy, gates, drr, window=21):
    cube = DataCube(energy, gates, drr, 'Gate (V)', 'Frozen calibration QA', 'DR/R')
    result, used = apply_sg_derivative_energy(cube, derivative=2, window_length=window, polyorder=2)
    if used != window:
        raise AssertionError('Unexpected derivative-window clamping')
    return result.Z


def nonuniform_input(peak=None):
    values = list(synthetic(peak=peak))
    e = values[0]
    # Strictly monotonic, intentionally outside the uniform-grid tolerance.
    values[0] = e + .02*(e-1.8)**2
    spacing = np.diff(values[0])
    assert not np.allclose(spacing, np.median(spacing), rtol=1e-6, atol=1e-12)
    return values


class ProductionDerivativeProtectionTests(unittest.TestCase):
    def test_strong_and_weak_peaks_do_not_hide_nonuniform_sg21_protection_damage(self):
        for amplitude in (.07, .0002):
            with self.subTest(amplitude=amplitude):
                e, z, b, gates, frames = nonuniform_input(peak=(1.940, .0007, amplitude))
                settings = replace(TRUSTED, protected_regions=((1.938, 1.942),))
                result = evaluate_bn_group(e, z, b, gates=gates, settings=settings, background_repeats=frames)
                self.assertFalse(result.eligible, result.reasons)
                protected = (e >= 1.938) & (e <= 1.942)
                original = z.mean(0)/b-1
                original_d2 = derivative(e, gates, original)
                no_peak = nonuniform_input()[1]
                known_peak_d2 = derivative(e, gates, (z.mean(0)-no_peak.mean(0))/b)
                peak_curvature_scale = np.max(abs(known_peak_d2[:, protected]))
                for method, candidate in result.candidate_backgrounds.items():
                    corrected = z.mean(0)/candidate-1
                    np.testing.assert_array_equal(original[:, protected], corrected[:, protected], err_msg=method)
                    candidate_d2 = derivative(e, gates, corrected)
                    # Cubic regridding is not compactly supported. Quantify its
                    # tail against the independently known injected peak, not
                    # the much larger background curvature. The QA criterion
                    # here is 10 ppm of that peak's curvature, not bitwise or
                    # machine-epsilon identity and not a universal guarantee.
                    delta = np.max(abs(original_d2[:, protected]-candidate_d2[:, protected]))
                    self.assertLessEqual(delta/peak_curvature_scale, 1e-5, method)
                    if method != 'original':
                        # The old peak-scaled tolerance above would overlook
                        # these tails. Actual acceptance uses a declared
                        # absolute tolerance independent of peak amplitude.
                        protection = result.report['candidates'][method]['selection_joint']['protection']['d2']
                        self.assertGreater(protection['max_absolute_change'], protection['absolute_tolerance'])
                        self.assertFalse(protection['passed'])
                        self.assertFalse(result.report['candidates'][method]['selection_joint']['passed'])
                np.testing.assert_array_equal(result.applied_background, b)

    def test_support_edges_do_not_imply_full_spectrum_derivative_identity(self):
        e, z, b, gates, frames = nonuniform_input()
        result = evaluate_bn_group(e, z, b, gates=gates, settings=TRUSTED, background_repeats=frames)
        self.assertFalse(result.eligible, result.reasons)
        original = z.mean(0)/b-1
        corrected = z.mean(0)/result.candidate_backgrounds['local_shift_gain']-1
        outside = (e <= 1.89) | (e >= 1.98)
        np.testing.assert_array_equal(original[:, outside], corrected[:, outside])
        d2_delta = derivative(e, gates, corrected)-derivative(e, gates, original)
        self.assertGreater(np.max(abs(d2_delta[:, outside])), 1e-5)
        protected = (e >= 1.58) & (e <= 1.85)
        np.testing.assert_allclose(d2_delta[:, protected], 0, rtol=0, atol=1e-8)
        self.assertFalse(result.report['scope']['downstream_derivative_check_required'])
        self.assertTrue(result.report['scope']['derivative_check_is_specific_to_recorded_settings'])
        self.assertFalse(result.report['candidates']['local_shift_gain']['selection_joint']['passed'])

    def test_actual_sg101_can_cross_native_guard_and_is_rejected(self):
        e, z, b, gates, frames = nonuniform_input(peak=(1.940, .0007, .0002))
        settings = replace(TRUSTED, protected_regions=((1.938, 1.942),), derivative_window=101)
        result = evaluate_bn_group(e, z, b, gates=gates, settings=settings, background_repeats=frames)
        self.assertFalse(result.eligible, result.reasons)
        self.assertEqual(result.report['derivative_settings']['window_length'], 101)
        self.assertEqual(result.report['derivative_settings']['used_window_length'], 101)
        protected = (e >= 1.938) & (e <= 1.942)
        original = z.mean(0)/b-1
        corrected = z.mean(0)/result.candidate_backgrounds['local_shift_gain']-1
        np.testing.assert_array_equal(original[:, protected], corrected[:, protected])
        delta = derivative(e, gates, corrected, window=101)-derivative(e, gates, original, window=101)
        self.assertGreater(np.max(abs(delta[:, protected])), 1e-5)
        check = result.report['candidates']['local_shift_gain']['selection_joint']
        self.assertFalse(check['protection']['d2']['passed'])
        self.assertFalse(check['passed'])
        np.testing.assert_array_equal(result.applied_background, b)


if __name__ == '__main__':
    unittest.main()
