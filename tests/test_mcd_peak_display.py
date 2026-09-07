import unittest
import numpy as np

from core.mcd_peak_display import second_derivative_map


class MCDPeakDisplayTests(unittest.TestCase):
    def test_reversed_quadratic_and_source_unchanged(self):
        energy = np.linspace(2.0, 1.0, 101)
        raw = np.vstack((energy ** 2, np.ones(energy.size)))
        before = raw.copy()
        grid, display, window = second_derivative_map(energy, raw, 35)
        self.assertEqual(window, 35)
        self.assertTrue(np.allclose(display[0, 20:-20], 2.0, atol=1e-10))
        self.assertTrue(np.allclose(display[1, 20:-20], 0.0, atol=1e-10))
        self.assertTrue(np.array_equal(raw, before))
        self.assertTrue(np.all(np.diff(grid) > 0))

    def test_nonuniform_quadratic_is_close_after_uniform_resampling(self):
        energy = np.array([1.0, 1.01, 1.04, 1.1, 1.19, 1.31, 1.46, 1.64, 1.85, 2.1])
        raw = (energy ** 2)[None, :]
        _, display, _ = second_derivative_map(energy, raw, 7)
        self.assertTrue(np.allclose(display[0, 3:-3], 2.0, atol=0.1))


if __name__ == "__main__":
    unittest.main()
