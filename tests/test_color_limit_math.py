"""Independent fixtures for the three existing color-statistics policies."""
import unittest
from dataclasses import replace
import numpy as np
from core.loader import DataCube
from core.color_limits import ColorRequest, calculate_bounds, apply_pending_colors
from core.plotting import HeatmapParams, SplitColorScale


class ColorLimitMathTests(unittest.TestCase):
    def test_roi_and_center_zero_keep_exact_percentiles(self):
        cube = DataCube(np.arange(5.), np.array([0., 1.]),
                        np.array([[-2., 1., 3., 5., 7.], [100., np.nan, 200., np.inf, 300.]]),
                        'Gate', 'Math', 'Signal')
        request = ColorRequest((1., 3.), (0., 0.))
        np.testing.assert_allclose(calculate_bounds((cube,), request), (1.0004, 4.9996), atol=1e-14)
        np.testing.assert_allclose(calculate_bounds((cube,), replace(request, zero=True)),
                                   (-4.9996, 4.9996), atol=1e-14)
        np.testing.assert_allclose(calculate_bounds((cube,), replace(request, log=True)),
                                   (1.0004, 4.9996), atol=1e-14)
        self.assertIsNone(calculate_bounds((cube,), replace(request, ylim=(3., 4.))))

    def test_nonuniform_three_region_selection_uses_original_cell_boundary(self):
        cube = DataCube(np.array([1., 1.3, 2., 3.2, 4., 6.]), np.array([0.]),
                        np.array([[1., 2., 3., 5., 7., 8.]]), 'Gate', 'Math', 'Signal')
        request = ColorRequest((1., 6.), (0., 0.), (1.65, 3.6), 1)
        np.testing.assert_allclose(calculate_bounds((cube,), request), (3.0002, 4.9998), atol=1e-14)

    def test_legacy_flat_and_negative_log_policy_are_preserved(self):
        cube = DataCube(np.array([1., 2.]), np.array([0.]), np.array([[2., 2.]]), 'Gate', 'Math', 'Signal')
        request = ColorRequest((1., 2.), (0., 0.))
        self.assertEqual(calculate_bounds((cube,), request), (1.98, 2.02))
        self.assertEqual(calculate_bounds((cube,), replace(request, policy='legacy')), (2., 2.))
        cube = replace(cube, Z=np.array([[-3., -2.]]))
        self.assertIsNone(calculate_bounds((cube,), replace(request, log=True)))
        self.assertEqual(calculate_bounds((cube,), replace(request, log=True, policy='legacy')), (-3., -2.))
        self.assertEqual(calculate_bounds((cube,), replace(request, xlim=(10., 20.),
            policy='legacy', fallback_full=True)), (-2.9999, -2.0001))
        self.assertIsNone(calculate_bounds((cube,), replace(request, xlim=(10., 20.), policy='legacy')))

    def test_pending_export_merges_only_eligible_frozen_regions(self):
        cube = DataCube(np.array([1., 2., 3., 4.]), np.array([0.]),
                        np.array([[1., 3., 10., 20.]]), 'Gate', 'Math', 'Signal')
        params = HeatmapParams('Math', 'X', 'Y', 'Signal', -1., 99., (1., 4.), (0., 0.),
            split_scale=SplitColorScale(2.5, -50., 99., -20., 99.))
        request = ColorRequest((1., 4.), (0., 0.), (2.5,), 0)
        result, _ = apply_pending_colors(cube, cube, params, params,
                                        (('drr', 'left', request, ('left_vmax',), (cube,)),))
        self.assertEqual(result.split_scale.left_vmin, -50.)
        self.assertEqual(result.split_scale.right_vmax, 99.)
        self.assertAlmostEqual(result.split_scale.left_vmax, 2.9998)
        self.assertEqual(params.split_scale.left_vmax, 99.)

    def test_pending_export_uses_the_frozen_derivative_product(self):
        raw = DataCube(np.array([1., 2.]), np.array([0.]), np.array([[100., 200.]]), 'Gate', 'Raw', 'Signal')
        derivative = replace(raw, Z=np.array([[1., 2.]]))
        params = HeatmapParams('Raw', 'X', 'Y', 'Signal', -1., 99., (1., 2.), (0., 0.))
        request = ColorRequest((1., 2.), (0., 0.))
        result, _ = apply_pending_colors(raw, raw, params, params,
            (('drr', None, request, ('vmin', 'vmax'), (derivative,)),))
        self.assertAlmostEqual(result.vmin, 1.0001)
        self.assertAlmostEqual(result.vmax, 1.9999)

    def test_explicit_log_region_keeps_positive_values_below_legacy_floor(self):
        cube = DataCube(np.array([1., 2.]), np.array([0.]), np.array([[1e-15, 3e-15]]), 'Gate', 'Tiny', 'Signal')
        request = ColorRequest((1., 2.), (0., 0.), log=True, policy='explicit')
        np.testing.assert_allclose(calculate_bounds((cube,), request), (1.0002e-15, 2.9998e-15), rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
