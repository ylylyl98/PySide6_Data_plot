from __future__ import annotations

import unittest

import numpy as np

from core.loader import DataCube
from core.processing import apply_sg_derivative_energy, clamp_sg_window


def _cube(energy: np.ndarray, z: np.ndarray) -> DataCube:
    return DataCube(
        energy=np.asarray(energy, float).copy(),
        gate=np.asarray([0.0], float),
        Z=np.asarray(z, float),
        gate_label="Gate",
        title="quadratic",
        cbar_label="DR/R",
    )


class DerivativeProcessingTests(unittest.TestCase):
    def test_second_derivative_of_quadratic_is_constant(self) -> None:
        energy = np.linspace(-2.0, 2.0, 41)
        cube = _cube(energy, np.array([energy**2]))

        result, used_win = apply_sg_derivative_energy(
            cube, derivative=2, window_length=9, polyorder=2
        )

        self.assertEqual(result.cbar_label, "d2(DR/R)/dE2")
        self.assertEqual(used_win, 9)
        np.testing.assert_allclose(result.Z[0], 2.0, atol=1e-12)

    def test_first_derivative_of_quadratic_is_linear(self) -> None:
        energy = np.linspace(-2.0, 2.0, 41)
        cube = _cube(energy, np.array([energy**2]))

        result, _used_win = apply_sg_derivative_energy(
            cube, derivative=1, window_length=9, polyorder=2
        )

        np.testing.assert_allclose(result.Z[0], 2.0 * energy, atol=1e-12)

    def test_nonuniform_energy_is_resampled_before_second_derivative(self) -> None:
        energy = np.array(
            [-2.0, -1.5, -1.0, -0.7, -0.3, 0.0, 0.4, 0.9, 1.5, 2.0]
        )
        cube = _cube(energy, np.array([energy**2]))

        result, used_win = apply_sg_derivative_energy(
            cube, derivative=2, window_length=7, polyorder=2
        )

        self.assertEqual(used_win, 7)
        np.testing.assert_allclose(result.Z[0], 2.0, atol=1e-12)

    def test_nan_in_spectrum_does_not_produce_all_nan_derivative(self) -> None:
        energy = np.linspace(-2.0, 2.0, 21)
        values = energy**2
        values[8:12] = np.nan
        cube = _cube(energy, np.array([values]))

        result, _used_win = apply_sg_derivative_energy(
            cube, derivative=2, window_length=7, polyorder=2
        )

        self.assertEqual(np.isfinite(result.Z).sum(), result.Z.size)
        np.testing.assert_allclose(result.Z[0], 2.0, atol=1e-12)

    def test_window_is_clamped_to_valid_odd_size_for_short_high_order_data(self) -> None:
        self.assertEqual(clamp_sg_window(20, n_energy=5, polyorder=4), 5)
        self.assertEqual(clamp_sg_window(20, n_energy=6, polyorder=4), 5)
        self.assertEqual(clamp_sg_window(20, n_energy=7, polyorder=6), 7)


if __name__ == "__main__":
    unittest.main()
