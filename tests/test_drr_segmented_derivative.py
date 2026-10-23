import unittest

import numpy as np

from core.loader import DataCube
from core.processing import apply_sg_derivative_energy


class SegmentedDerivativeTests(unittest.TestCase):
    def cube(self, x, values, boundaries, *, y=None):
        y = np.arange(values.shape[0], dtype=float) if y is None else y
        cube = DataCube(
            np.asarray(x), np.asarray(y), np.asarray(values), "Doping", "segments", "DR/R",
            gate_unit="V", y_axis_semantic="doping",
            gate_coordinates={"TG": np.asarray(y) + 2},
        )
        cube.drr_background_boundaries = tuple(boundaries)
        return cube

    def assert_metadata(self, result, original):
        self.assertEqual(result.drr_background_boundaries, original.drr_background_boundaries)
        self.assertEqual(result.gate_unit, "V")
        self.assertEqual(result.y_axis_semantic, "doping")
        np.testing.assert_array_equal(result.gate_coordinates["TG"], original.gate_coordinates["TG"])
        self.assertIsNot(result.gate_coordinates["TG"], original.gate_coordinates["TG"])

    def test_first_and_second_derivative_ignore_large_background_jump(self):
        for uneven in (False, True):
            for descending in (False, True):
                for derivative in (1, 2):
                    with self.subTest(uneven=uneven, descending=descending, derivative=derivative):
                        x = np.linspace(1, 3, 101)
                        if uneven:
                            x = 1 + 2 * np.linspace(0, 1, 101) ** 1.3
                        seam = x[50]
                        curvature = np.where(x < seam, 2., -3.)
                        z = curvature*x*x + np.where(x < seam, 0., 10000.)
                        expected = 2*curvature*x if derivative == 1 else 2*curvature
                        if descending:
                            x, z, expected = x[::-1], z[::-1], expected[::-1]
                        cube = self.cube(x, np.vstack((z, z + 7)), [seam])
                        result, used = apply_sg_derivative_energy(
                            cube, derivative=derivative, window_length=9, polyorder=2)
                        order = np.argsort(x)
                        self.assertTrue(np.isnan(result.Z[:, order[46:54]]).all())
                        good = np.isfinite(result.Z[0])
                        self.assertEqual(np.count_nonzero(good), 93)
                        np.testing.assert_allclose(result.Z[0, good], expected[good], atol=1e-6)
                        np.testing.assert_allclose(result.Z[1, good], expected[good], atol=1e-6)
                        np.testing.assert_array_equal(result.energy, x)
                        self.assertEqual(used, 9)
                        self.assert_metadata(result, cube)

    def test_gappy_interpolation_stays_within_each_background(self):
        x = np.linspace(1, 3, 101)
        z = 2*x*x + np.where(x < 2, 0., 1e6)
        z[20] = np.nan
        z[80] = np.nan
        cube = self.cube(x, z[None, :], [2.])
        result, _ = apply_sg_derivative_energy(cube, derivative=2, window_length=9, polyorder=2)
        self.assertTrue(np.isnan(result.Z[:, 46:54]).all())
        np.testing.assert_allclose(result.Z[np.isfinite(result.Z)], 4., atol=1e-5)

    def test_adjacent_short_segment_remains_nan_and_windows_clamp_per_segment(self):
        x = np.arange(31, dtype=float)
        z = x*x + np.where(x >= 12, 1000., 0.) + np.where(x >= 15, 1000., 0.)
        cube = self.cube(x, z[None, :], [12., 15.])
        result, used = apply_sg_derivative_energy(cube, derivative=2, window_length=21, polyorder=2)
        # Segment 0 has 12 samples/window 11, segment 1 is too short,
        # segment 2 has 16 samples/window 15.
        self.assertTrue(np.isnan(result.Z[:, 7:22]).all())
        np.testing.assert_allclose(result.Z[:, :7], 2., atol=1e-9)
        np.testing.assert_allclose(result.Z[:, 22:], 2., atol=1e-9)
        self.assertEqual(used, 15)

    def test_mixed_derivative_keeps_segment_specific_slopes_and_seam_nan(self):
        for descending in (False, True):
            x = 1 + 2*np.linspace(0, 1, 81)**1.2
            y = np.linspace(-1, 1, 15)[::-1]
            seam = x[40]
            slopes = np.where(x < seam, 3., -7.)
            z = slopes[None, :]*x[None, :]*y[:, None] + 0.25*x[None, :]**2
            z += np.where(x < seam, 0., 1000.)[None, :]*(1 + 2*y[:, None])
            if descending:
                x, z, slopes = x[::-1], z[:, ::-1], slopes[::-1]
            cube = self.cube(x, z, [seam], y=y)
            result, used = apply_sg_derivative_energy(
                cube, derivative=("mixed", 7), window_length=9, polyorder=2)
            order = np.argsort(x)
            self.assertTrue(np.isnan(result.Z[:, order[36:44]]).all())
            good = np.isfinite(result.Z[0])
            np.testing.assert_allclose(result.Z[:, good], np.tile(slopes[good], (len(y), 1)), atol=1e-8)
            self.assertEqual(used, 9)
            self.assert_metadata(result, cube)

    def test_none_derivative_preserves_raw_values_and_identity(self):
        x = np.arange(9.)
        cube = self.cube(x, x[None, :], [4.])
        result, used = apply_sg_derivative_energy(cube, derivative=None, window_length=8)
        self.assertIs(result, cube)
        np.testing.assert_array_equal(result.Z, x[None, :])
        self.assertEqual(used, 8)

    def test_no_segment_has_enough_samples_returns_nan_for_all_derivative_modes(self):
        x = np.arange(12.)
        y = np.arange(7.)
        cube = self.cube(x, x[None, :]**2 + 3*y[:, None]*x[None, :], [4., 8.], y=y)
        for derivative in (1, 2, ("mixed", 5)):
            with self.subTest(derivative=derivative):
                result, _ = apply_sg_derivative_energy(cube, derivative=derivative, window_length=9)
                self.assertTrue(np.isnan(result.Z).all())

    def test_seams_outside_data_do_not_mask_outer_edges(self):
        x = np.linspace(1, 2, 21)
        cube = self.cube(x, (3*x*x)[None, :], [-10., 1., 3.])
        result, _ = apply_sg_derivative_energy(cube, derivative=2, window_length=9)
        np.testing.assert_allclose(result.Z, 6., atol=1e-9)


if __name__ == "__main__":
    unittest.main()
