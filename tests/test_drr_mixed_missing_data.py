import unittest

import numpy as np
from scipy.signal import savgol_filter

from core.loader import DataCube
from core.processing import apply_sg_derivative_energy


class MixedMissingDataTests(unittest.TestCase):
    def differentiate(self, x, y, z, window=5, poly=2):
        cube = DataCube(x, y, z, 'Bias (V)', 'gappy', 'DR/R')
        before = z.copy()
        result, used = apply_sg_derivative_energy(
            cube, derivative=('mixed', window), window_length=window,
            polyorder=poly)
        np.testing.assert_array_equal(z, before)
        self.assertEqual(used, window)
        return result.Z

    def test_missing_gate_with_scan_wide_window_preserves_other_rows(self):
        x, y = np.linspace(1, 2, 9), np.linspace(-1, 1, 9)
        z = 3*y[:, None]*x[None, :] + x[None, :]**2
        z[4] = np.nan
        result = self.differentiate(x, y, z, window=9)
        np.testing.assert_array_equal(np.isnan(result), np.isnan(z))
        np.testing.assert_allclose(result[np.isfinite(z)], 3., atol=1e-9)

    def test_alignment_edges_and_internal_gaps_on_uneven_descending_axes(self):
        x = (1 + np.linspace(0, 1, 13)**1.2)[::-1]
        y = (np.linspace(0, 1, 11)**1.3)[::-1]
        for invalid in [np.nan, np.inf, -np.inf]:
            with self.subTest(invalid=invalid):
                z = 3*y[:, None]*x[None, :] + x[None, :]**2 + 2*y[:, None]**2
                z[2, :2] = invalid
                z[7, -2:] = invalid
                z[5, 6] = invalid
                result = self.differentiate(x, y, z)
                np.testing.assert_array_equal(np.isnan(result), ~np.isfinite(z))
                np.testing.assert_allclose(result[np.isfinite(z)], 3., atol=1e-9)

    def test_insufficient_local_support_stays_missing_on_either_axis(self):
        x = y = np.arange(11.)
        for transpose in [False, True]:
            with self.subTest(transpose=transpose):
                z = 3*y[:, None]*x[None, :]
                z[:, 2:5] = np.nan
                if transpose:
                    z = z.T.copy()
                result = self.differentiate(x, y, z)
                if transpose:
                    result = result.T
                # The first window has only two samples for a quadratic.
                self.assertTrue(np.isnan(result[:, :5]).all())
                np.testing.assert_allclose(result[:, 5:], 3., atol=1e-9)

    def test_all_missing_returns_nan(self):
        axis = np.arange(9.)
        result = self.differentiate(axis, axis, np.full((9, 9), np.nan))
        self.assertTrue(np.isnan(result).all())

    def test_finite_data_matches_separable_savgol_for_all_supported_orders(self):
        x, y = np.linspace(1, 2, 17), np.linspace(-1, 1, 13)
        z = np.random.default_rng(42).normal(size=(len(y), len(x)))
        for poly in range(1, 7):
            with self.subTest(poly=poly):
                expected = savgol_filter(z, 9, poly, deriv=1,
                                         delta=x[1]-x[0], axis=1, mode='interp')
                expected = savgol_filter(expected, 9, poly, deriv=1,
                                         delta=y[1]-y[0], axis=0, mode='interp')
                result = self.differentiate(x, y, z, window=9, poly=poly)
                np.testing.assert_allclose(result, expected, atol=1e-7, rtol=1e-9)
