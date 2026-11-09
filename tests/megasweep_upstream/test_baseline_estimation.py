# Adapted from Megasweep Analysis commit 2612f6f.
import unittest

import numpy as np

from core.megasweep.megasweep_analysis import estimate_global_baseline


class BaselineEstimationTests(unittest.TestCase):
    def test_estimates_one_global_baseline_outside_peak_window(self):
        rng = np.random.default_rng(7)
        energy = np.linspace(1.0, 2.0, 101)
        intensity = 100.0 + rng.normal(0.0, 1.0, size=(60, energy.size))
        peak_mask = (energy > 1.4) & (energy < 1.6)
        intensity[:, peak_mask] += 1000.0
        intensity[:3, :] += 400.0

        baseline, info = estimate_global_baseline(
            intensity,
            energy,
            1.4,
            1.6,
            return_info=True,
        )

        self.assertAlmostEqual(100.0, baseline, delta=3.0)
        self.assertFalse(info["fallback"])
        self.assertEqual(82, info["baseline_channels"])

    def test_falls_back_to_full_spectrum_percentile_when_needed(self):
        energy = np.linspace(1.0, 2.0, 21)
        intensity = np.full((10, energy.size), 50.0)
        intensity[:, 8:13] = 500.0

        baseline, info = estimate_global_baseline(
            intensity,
            energy,
            0.0,
            3.0,
            return_info=True,
        )

        self.assertEqual(50.0, baseline)
        self.assertTrue(info["fallback"])
        self.assertEqual(21, info["baseline_channels"])


if __name__ == "__main__":
    unittest.main()
