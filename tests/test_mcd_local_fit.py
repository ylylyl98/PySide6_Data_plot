from __future__ import annotations

import unittest

import numpy as np

from core.mcd_local_fit import FIT_GAP, FIT_OK, fit_local_resonance, mixed_model


class MCDLocalFitTests(unittest.TestCase):
    def test_mixed_fit_recovers_asymmetric_center_with_sloping_background(self):
        energy = np.linspace(1.630, 1.660, 151)
        seed = 1.6436
        x = (energy - seed) * 1000.0
        expected = np.array([0.25, 0.006, 1.4, -0.65, 2.1, 1.8])
        y = mixed_model(x, expected) + 0.001 * np.sin(np.arange(energy.size))

        fit = fit_local_resonance(energy, y, seed_energy_ev=seed, max_starts=150)

        self.assertEqual(fit.status, FIT_OK)
        self.assertIsNotNone(fit.center_ev)
        self.assertAlmostEqual(float(fit.center_ev), seed + expected[4] / 1000.0, delta=0.0002)
        self.assertIsNotNone(fit.gamma_mev)

    def test_nonfinite_point_is_explicit_gap(self):
        energy = np.linspace(1.630, 1.660, 151)
        values = np.ones_like(energy)
        values[70] = np.nan

        fit = fit_local_resonance(energy, values, seed_energy_ev=1.6436, max_starts=6)

        self.assertEqual(fit.status, FIT_GAP)
        self.assertIsNone(fit.center_ev)

    def test_flat_spectrum_is_rejected_without_model_center(self):
        energy = np.linspace(1.630, 1.660, 151)

        fit = fit_local_resonance(energy, np.full_like(energy, 10000.0), seed_energy_ev=1.6436, max_starts=150)

        self.assertNotEqual(fit.status, FIT_OK)
        self.assertIsNone(fit.center_ev)

    def test_quadratic_model_uses_explicit_optional_background(self):
        energy = np.linspace(1.630, 1.660, 151)
        seed = 1.6436
        x = (energy - seed) * 1000.0
        y = mixed_model(x, np.array([0.1, 0.01, 1.2, 0.4, -1.0, 1.6])) + 0.002 * (x / 10.0) ** 2

        fit = fit_local_resonance(energy, y, seed_energy_ev=seed, background_model="quadratic", max_starts=150)

        self.assertEqual(fit.status, FIT_OK)
        self.assertEqual(len(fit.parameters), 7)


if __name__ == "__main__":
    unittest.main()
