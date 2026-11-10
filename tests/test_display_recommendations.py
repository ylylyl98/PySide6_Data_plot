import unittest
from types import SimpleNamespace

import numpy as np

from core.display_recommendations import recommend_color_bounds, recommend_x_range


def cube(z, x=None, y=None):
    z = np.asarray(z, dtype=float)
    return SimpleNamespace(energy=np.arange(z.shape[1], dtype=float) if x is None else np.asarray(x),
                           gate=np.arange(z.shape[0], dtype=float) if y is None else np.asarray(y), Z=z)


class DisplayRecommendationTests(unittest.TestCase):
    def test_overall_uses_finite_view_percentiles_without_mutation(self):
        z = np.arange(100., dtype=float).reshape(10, 10)
        z[0, 0] = np.nan
        data = cube(z); before = data.Z.copy()
        result = recommend_color_bounds(data, xlim=(2, 7), ylim=(1, 8))
        expected = np.percentile(z[1:9, 2:8], [1, 99])
        np.testing.assert_allclose([result.low, result.high], expected)
        np.testing.assert_equal(data.Z, before)
        self.assertGreater(result.saturated_fraction, 0)

    def test_weak_uses_quiet_noise_not_largest_peak(self):
        rng = np.random.default_rng(19)
        z = rng.normal(0, 50, (70, 300))
        z[:, 130:145] += 200000
        data = cube(z)
        weak = recommend_color_bounds(data, mode='weak')
        stronger = recommend_color_bounds(data, mode='weak', strength=2)
        self.assertLess(weak.high, 1000)
        self.assertGreater(weak.saturated_fraction, .04)
        self.assertLess(stronger.high - stronger.low, weak.high - weak.low)
        self.assertGreater(stronger.saturated_fraction, weak.saturated_fraction)

    def test_weak_drifting_baseline_limits_saturation_and_strength_is_nested(self):
        rng = np.random.default_rng(271)
        x = np.linspace(0, 2 * np.pi, 600)
        z = 6 * np.sin(x)[None, :] + rng.normal(0, .12, (50, len(x)))
        z[:, 285:290] += 80
        data = cube(z)
        before = (data.energy.copy(), data.gate.copy(), data.Z.copy())
        overall = recommend_color_bounds(data, mode='overall')
        results = [recommend_color_bounds(data, mode='weak', strength=s) for s in (.5, 1, 2, 4)]
        default = results[1]
        self.assertLessEqual(default.saturated_fraction, .4 + 1 / z.size)
        self.assertGreaterEqual(default.low, overall.low)
        self.assertLessEqual(default.high, overall.high)
        for broad, narrow in zip(results, results[1:]):
            self.assertLessEqual(broad.low, narrow.low)
            self.assertGreaterEqual(broad.high, narrow.high)
            self.assertGreaterEqual(broad.high - broad.low, narrow.high - narrow.low)
        for actual, expected in zip((data.energy, data.gate, data.Z), before):
            np.testing.assert_array_equal(actual, expected)

    def test_weak_sparse_derivative_peak_keeps_narrow_noise_range(self):
        rng = np.random.default_rng(272)
        z = rng.normal(0, .01, (40, 500))
        z[:, 240:248] += 10
        overall = recommend_color_bounds(cube(z), mode='overall')
        weak = recommend_color_bounds(cube(z), mode='weak')
        self.assertLess(weak.high - weak.low, .1)
        self.assertLess(weak.high - weak.low, (overall.high - overall.low) / 10)
        self.assertLessEqual(weak.saturated_fraction, .4)

    def test_weak_is_affine_equivariant_for_positive_scale(self):
        rng = np.random.default_rng(273)
        x = np.linspace(0, 2 * np.pi, 400)
        z = 4 * np.sin(x)[None, :] + rng.normal(0, .05, (30, len(x)))
        z[:, 195:200] += 30
        for strength in (.5, 1, 2, 4):
            original = recommend_color_bounds(cube(z), mode='weak', strength=strength)
            transformed = recommend_color_bounds(cube(3.25 * z + 17), mode='weak', strength=strength)
            np.testing.assert_allclose([transformed.low, transformed.high],
                                       3.25 * np.array([original.low, original.high]) + 17,
                                       rtol=1e-10, atol=1e-10)
            self.assertAlmostEqual(original.saturated_fraction, transformed.saturated_fraction)

    def test_weak_log_matches_existing_positive_percentile_behavior(self):
        z = np.linspace(-10, 20, 600).reshape(20, 30)
        expected = np.percentile(z[z > 0], [1, 99])
        for strength in (.5, 1, 2, 4):
            result = recommend_color_bounds(cube(z), mode='weak', strength=strength, log_scale=True)
            np.testing.assert_allclose([result.low, result.high], expected)

    def test_log_all_nonpositive_and_constant_inputs(self):
        self.assertIsNone(recommend_color_bounds(cube([[-2, 0], [np.nan, -1]]), log_scale=True))
        result = recommend_color_bounds(cube([[3, 3], [3, 3]]), log_scale=True, mode='weak')
        self.assertLess(result.low, 3); self.assertGreater(result.high, 3)
        self.assertGreater(result.low, 0)
        linear = recommend_color_bounds(cube([[0, 0], [0, 0]]), mode='weak')
        self.assertLess(linear.low, 0); self.assertGreater(linear.high, 0)
        self.assertIsNone(recommend_color_bounds(cube([[np.nan]])))

    def test_invalid_request_rejected(self):
        with self.assertRaises(ValueError): recommend_color_bounds(cube([[1]]), mode='weak', strength=0)
        with self.assertRaises(ValueError): recommend_color_bounds(cube([[1]]), mode='unknown')
        with self.assertRaises(ValueError): recommend_color_bounds(SimpleNamespace(energy=[1], gate=[1], Z=np.ones((2, 2))))

    def test_x_range_preserves_weak_separate_feature_and_pads(self):
        rng = np.random.default_rng(3)
        x = np.linspace(0, 10, 501)
        z = rng.normal(0, .1, (40, len(x)))
        z += 100 * np.exp(-((x - 4) / .2) ** 2)
        z += 1.2 * np.exp(-((x - 7) / .22) ** 2)
        result = recommend_x_range([cube(z, x=x)])
        self.assertIsNotNone(result)
        self.assertLess(result[0], 3.8)
        self.assertGreater(result[1], 7.2)
        self.assertGreater(result[0], 0)
        self.assertLess(result[1], 10)

    def test_x_range_preserves_branch_in_five_adjacent_scans(self):
        rng = np.random.default_rng(73); x = np.linspace(0, 10, 501)
        z = rng.normal(0, .1, (100, len(x)))
        z += 100 * np.exp(-((x - 4) / .2) ** 2)
        z[38:43] += 1.2 * np.exp(-((x - 8) / .2) ** 2)
        result = recommend_x_range([cube(z, x=x)])
        self.assertIsNotNone(result)
        self.assertGreater(result[1], 8.2)

    def test_x_range_keeps_true_edge_structure_and_rejects_noise(self):
        rng = np.random.default_rng(74); x = np.linspace(0, 10, 501)
        noise = rng.normal(0, .1, (100, len(x)))
        self.assertIsNone(recommend_x_range([cube(noise, x=x)]))
        z = noise.copy()
        z[20:25] += 2 * np.exp(-((x - 9.97) / .09) ** 2)
        z += 100 * np.exp(-((x - 4) / .2) ** 2)
        result = recommend_x_range([cube(z, x=x)])
        self.assertTrue(result is None or result[1] == x[-1])

    def test_x_range_handles_correlated_noise_and_slow_background(self):
        from scipy.ndimage import gaussian_filter1d
        rng = np.random.default_rng(101); x = np.linspace(0, 10, 501)
        noise = gaussian_filter1d(rng.normal(0, .2, (80, len(x))), 2, axis=1)
        z = noise + .7 * np.sin(x / 3)[None, :]
        z += 10 * np.exp(-((x - 4) / .2) ** 2)
        z[12:17] += 1.2 * np.exp(-((x - 7) / .12) ** 2)
        result = recommend_x_range([cube(z, x=x)])
        self.assertIsNotNone(result)
        self.assertGreater(result[0], 0)
        self.assertLess(result[0], 4)
        self.assertGreater(result[1], 7)
        self.assertLess(result[1], 10)

    def test_weak_constant_data_has_finite_nonzero_span(self):
        result = recommend_color_bounds(cube(np.full((8, 80), 42.)), mode='weak')
        self.assertLess(result.low, 42)
        self.assertGreater(result.high, 42)
        self.assertEqual(result.noise, 0)

    def test_x_range_sorts_scan_and_energy_axes(self):
        rng = np.random.default_rng(18); x = np.linspace(0, 10, 501)
        y = np.linspace(-5, 5, 100)
        z = rng.normal(0, .1, (len(y), len(x)))
        z += 50 * np.exp(-((x - 4) / .2) ** 2)
        z[40:45] += 1.5 * np.exp(-((x - 8) / .15) ** 2)
        expected = recommend_x_range([cube(z, x=x, y=y)])
        self.assertIsNotNone(expected)
        order = rng.permutation(len(y))
        shuffled = cube(z[order, ::-1], x=x[::-1], y=y[order])
        self.assertEqual(recommend_x_range([shuffled]), expected)
        self.assertGreater(expected[1], 8)

    def test_x_range_nonfinite_and_short_axes_are_safe(self):
        z = np.ones((10, 40)); z[0, 3] = np.nan
        self.assertIsNone(recommend_x_range([cube(z)]))
        self.assertIsNone(recommend_x_range([cube(np.ones((4, 40)))]))
        self.assertIsNone(recommend_x_range([cube(np.ones((10, 29)))]))
        x = np.linspace(0, 1, 40); x[1] = x[0]
        self.assertIsNone(recommend_x_range([cube(np.ones((10, 40)), x=x)]))
        y = np.arange(10.); y[1] = y[0]
        self.assertIsNone(recommend_x_range([cube(np.ones((10, 40)), y=y)]))

    def test_x_range_rejects_ambiguous_and_combines_inputs(self):
        self.assertIsNone(recommend_x_range([]))
        self.assertIsNone(recommend_x_range([cube(np.ones((10, 10)))]))
        self.assertIsNone(recommend_x_range([cube(np.full((10, 50), np.nan))]))
        rng = np.random.default_rng(7); x = np.linspace(0, 10, 501)
        def item(center):
            return cube(rng.normal(0, .1, (30, len(x))) + 5 * np.exp(-((x-center)/.2)**2), x=x)
        first, second = item(3), item(7)
        combined = recommend_x_range([first, second])
        self.assertLess(combined[0], 3); self.assertGreater(combined[1], 7)


if __name__ == '__main__':
    unittest.main()
