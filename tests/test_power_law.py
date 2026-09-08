import unittest
import numpy as np
from core.power_law import fit_power_law, local_slopes, suggest_power_range


class PowerLawTests(unittest.TestCase):
    def test_fit_recovers_known_power_law(self):
        p = np.logspace(0, 3, 20)
        out = fit_power_law(p, 2.5 * p**1.35, 1, 1000)
        self.assertAlmostEqual(out["alpha"], 1.35)
        self.assertAlmostEqual(out["amplitude"], 2.5)
        self.assertEqual((out["n"], out["excluded"]), (20, 0))

    def test_fit_excludes_invalid_and_masked_values(self):
        p = np.array([1, 2, 3, 4, 5, 6, 7, 8.])
        y = 4 * p
        y[1], y[2] = 0, np.nan
        out = fit_power_law(p, y, 1, 8, valid=[1, 1, 1, 1, 1, 1, 0, 1])
        self.assertEqual((out["n"], out["excluded"]), (5, 3))

    def test_requested_bounds_are_preserved(self):
        out = fit_power_law(np.arange(1., 11.), 2 * np.arange(1., 11.), 2.5, 8.5)
        self.assertEqual((out["lower"], out["upper"]), (2.5, 8.5))

    def test_suggest_selects_near_linear_plateau(self):
        p = np.logspace(0, 4, 41)
        y = p.copy()
        y[:8] *= 1 + 3 * (p[:8] / p[7]) ** -1
        y[28:] *= 1 + 2 * (p[28:] / p[28]) ** 2
        out = suggest_power_range(p, y)
        self.assertAlmostEqual(out["alpha"], 1)
        self.assertGreater(out["lower"], p[4])
        self.assertLess(out["upper"], p[-3])

    def test_suggest_rejects_noise_and_nonlinear_data(self):
        p = np.logspace(0, 4, 30)
        rng = np.random.default_rng(4)
        y = p ** (1.7 + .4 * np.sin(np.log(p))) * np.exp(rng.normal(0, .35, p.size))
        with self.assertRaisesRegex(ValueError, "No reliable"):
            suggest_power_range(p, y)

    def test_large_dataset_covers_high_power_region(self):
        p = np.logspace(0, 8, 1200)
        y = p ** 1.7
        y[p > 1e5] = p[p > 1e5]
        out = suggest_power_range(p, y, tolerance=.15, min_decades=.5)
        self.assertGreater(out["lower"], 1e5 / 2)

    def test_fit_requires_distinct_span_and_local_slopes_shape(self):
        with self.assertRaises(ValueError):
            fit_power_law([1, 1, 1, 1], [1, 2, 3, 4], 1, 2)
        p = np.logspace(0, 2, 10)
        x, slopes = local_slopes(p, 3 * p, window=5)
        self.assertEqual((len(x), len(slopes)), (6, 6))
        with self.assertRaisesRegex(ValueError, "No reliable"):
            suggest_power_range(p, p, min_decades=3)


if __name__ == "__main__":
    unittest.main()
