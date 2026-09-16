import unittest
import numpy as np

from core import curie_weiss as cw


class CurieWeissTests(unittest.TestCase):
    def test_inverse_fit_uses_linear_regression_and_propagates_errors(self):
        # Inverse observations 2, 3, 5 have OLS m=1.5, b=1/3.
        result = cw.fit_inverse_curie_weiss([1, 2, 3], [.5, 1/3, .2],
                                            slope_se=[.025, .01, .008])
        self.assertAlmostEqual(result['theta_k'], -2/9)
        self.assertAlmostEqual(result['amplitude'], 2/3)
        np.testing.assert_allclose(result['inverse_slope_se'], [.1, .09, .2])
        self.assertEqual(result['fit_space'], 'inverse_slope')
        self.assertIsNone(result['theta_ci95_k'])  # Three noisy points do not constrain the ratio.

    def test_inverse_fit_rejects_pole_in_temperature_range_and_zero_crossing(self):
        for slopes in ([1, 2, 4], [1, 0, -1]):
            with self.assertRaises(ValueError):
                cw.fit_inverse_curie_weiss([1, 2, 3], slopes)

    def test_inverse_fit_optical_sign_and_fixed_background(self):
        t = np.array([3., 4., 6., 9., 15., 25.])
        result = cw.fit_inverse_curie_weiss(t, .03 - 2/(t+4),
                                            background_mode='fixed', background=.03)
        self.assertAlmostEqual(result['theta_k'], -4)
        self.assertAlmostEqual(result['amplitude'], -2)
        with self.assertRaises(ValueError):
            cw.fit_inverse_curie_weiss(t, 2/(t+4), background_mode='fit')

    def test_inverse_fit_window_normalization_does_not_change_theta_or_interval(self):
        t = np.array([3., 4., 6., 9., 15., 25.])
        s = 2/(t+4) + np.array([.001, -.001, .002, -.001, 0, .001])
        first = cw.fit_inverse_curie_weiss(t, s)
        scaled = cw.fit_inverse_curie_weiss(t, -5*s)
        self.assertIsNotNone(first['theta_ci95_k'])
        self.assertAlmostEqual(first['theta_k'], scaled['theta_k'])
        np.testing.assert_allclose(first['theta_ci95_k'], scaled['theta_ci95_k'])

    def test_recovers_signed_amplitude_and_weiss_sign(self):
        for theta, amplitude in [(-4.0, 2.0), (1.0, -3.0)]:
            t = np.array([3., 4., 6., 9., 15., 25.])
            result = cw.fit_curie_weiss(t, amplitude / (t - theta))
            self.assertAlmostEqual(result['theta_k'], theta, places=5)
            self.assertAlmostEqual(result['amplitude'], amplitude, places=5)
            self.assertGreater(result['r_squared'], .999999)

    def test_fixed_and_free_background_recover_parameters(self):
        t = np.array([3., 4., 6., 9., 15., 25., 40.])
        y = .03 + 2 / (t + 4)
        for mode in ['fixed', 'fit']:
            result = cw.fit_curie_weiss(t, y, background_mode=mode, background=.03)
            self.assertAlmostEqual(result['theta_k'], -4., places=4)
            self.assertAlmostEqual(result['background'], .03, places=5)

    def test_rejects_insufficient_distinct_temperatures_and_invalid_data(self):
        for t, y in [([1, 2], [1, .5]), ([1, 1, 2], [1, 1, .5]),
                     ([0, 1, 2], [1, .5, .3]), ([1, 2, 3], [1, np.nan, .3]),
                     ([1, 2, 3], [1, 1, 1]), ([1, 2, 3], [1, 0, -1])]:
            with self.assertRaises(ValueError):
                cw.fit_curie_weiss(t, y)

    def test_free_background_requires_residual_degrees_of_freedom(self):
        with self.assertRaises(ValueError):
            cw.fit_curie_weiss([2, 3, 4], [.3, .2, .1], background_mode='fit')

    def test_noisy_four_points_are_flagged_as_exploratory(self):
        result = cw.fit_curie_weiss([1.67, 2.2, 2.7, 3.2],
                                   [.07765, .06825, .07150, .05901])
        self.assertTrue(result['warnings'])
        self.assertEqual(len(result['theta_ci95_k']), 2)
        self.assertEqual(result['n'], 4)

    def test_linear_free_background_is_not_reported_as_identified_afm(self):
        t = np.array([3., 4., 6., 9., 15., 25.])
        result = cw.fit_curie_weiss(t, 1+.01*t, background_mode='fit')
        self.assertIsNone(result['theta_ci95_k'])
        self.assertEqual(result['interpretation'], 'Unresolved interaction sign')


if __name__ == '__main__':
    unittest.main()
