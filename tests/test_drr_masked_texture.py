from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from core import drr_background_structure as structure
from core.drr_fitted_background import BackgroundFitSettings, estimate_background
from core.loader import DataCube
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


def compact_peak(energy, centers, radius, amplitude):
    u = (energy[None, :]-np.asarray(centers)[:, None])/radius
    peak = np.zeros_like(u)
    inside = abs(u) < 1
    peak[inside] = amplitude*np.exp(1-1/(1-u[inside]**2))
    return peak


class MaskedTextureTests(unittest.TestCase):
    def setUp(self):
        self.energy = np.sort(1239.841984/np.linspace(610, 870, 1001))

    def test_protected_values_never_change_texture_or_coverage(self):
        energy = self.energy
        gate = np.linspace(-1, 1, 41)
        fringe = .003*np.sin(2*np.pi*energy/.0061)+.002*np.cos(2*np.pi*energy/.0093)
        values = 2+.3*energy+.1*energy**2 + fringe + gate[:, None]*(.2+.04*energy)
        anchors = np.ones(values.shape, bool)
        anchors[:, abs(energy-1.79)<.004] = False
        anchors &= abs(energy[None, :]-(1.56+.12*(gate[:, None]+1)/2))>.004
        original = values.copy()
        template, coverage, _ = structure.shared_texture(energy, values, anchors)
        self.assertGreater(np.count_nonzero(coverage>.5), len(energy)//2)
        for replacement in (.1, 100., np.nan):
            changed = values.copy()
            changed[~anchors] = replacement
            actual, applied, _ = structure.shared_texture(energy, changed, anchors)
            np.testing.assert_array_equal(actual, template)
            np.testing.assert_array_equal(applied, coverage)
        np.testing.assert_array_equal(values, original)

    def test_complete_windows_keep_the_original_projection_bit_for_bit(self):
        values = np.random.default_rng(49).normal(size=(3, len(self.energy)))
        anchors = np.ones(values.shape, bool)
        expected = structure._local_quadratic_trend(self.energy, values, .035)
        trend, usable = structure._masked_local_quadratic_trend(self.energy, values, anchors, .035)
        np.testing.assert_array_equal(trend, expected)
        self.assertTrue(usable.all())
        anchors[0, abs(self.energy-1.7)<.004] = False
        anchors[1, abs(self.energy-1.8)<.006] = False
        trend, usable = structure._masked_local_quadratic_trend(self.energy, values, anchors, .035)
        for row, center in ((0, 1.7), (1, 1.8)):
            unaffected = abs(self.energy-center)>.03
            np.testing.assert_array_equal(trend[row, unaffected], expected[row, unaffected])
        np.testing.assert_array_equal(trend[2], expected[2])

    def test_partial_windows_preserve_quadratics_on_measured_energy_grid(self):
        energy = self.energy
        values = np.asarray([2+3*energy-4*energy**2, -1+.7*energy+.2*energy**2])
        anchors = np.ones(values.shape, bool)
        anchors[0, abs(energy-1.7)<.008] = False
        anchors[1, abs(energy-1.8)<.004] = False
        trend, usable = structure._masked_local_quadratic_trend(energy, values, anchors, .035)
        self.assertTrue(np.all(usable==anchors))
        np.testing.assert_allclose(trend[usable], values[usable], rtol=0, atol=2e-11)
        corrupted = values.copy()
        corrupted[~anchors] = 1e100
        changed, new_usable = structure._masked_local_quadratic_trend(energy, corrupted, anchors, .035)
        np.testing.assert_array_equal(changed, trend)
        np.testing.assert_array_equal(new_usable, usable)

    def test_insufficient_and_ill_conditioned_partial_windows_are_not_usable(self):
        energy = self.energy
        anchors = np.zeros((1, len(energy)), bool)
        anchors[0, 490:499] = True  # Nine points cannot define the supported local fit.
        _, usable = structure._masked_local_quadratic_trend(energy, np.ones_like(anchors, float), anchors, .035)
        self.assertFalse(usable.any())
        # Ten almost coincident anchors give a numerically unidentifiable
        # quadratic even though their count passes the minimum support gate.
        energy = np.r_[np.linspace(1.4, 1.69, 500), 1.7+np.arange(10)*1e-10,
                       np.linspace(1.71, 2., 500)]
        anchors = np.zeros((1, len(energy)), bool)
        anchors[0, 500:510] = True
        _, usable = structure._masked_local_quadratic_trend(energy, np.ones_like(anchors, float), anchors, .035)
        self.assertFalse(usable.any())

    def test_protected_compact_stationary_and_moving_injections_preserve_final_background(self):
        energy = self.energy
        gate = np.linspace(-1, 1, 41)
        x = energy-energy.mean()
        fringe = .003*np.sin(2*np.pi*energy/.0061)+.002*np.cos(2*np.pi*energy/.0093)
        background = np.exp(8+.3*x+.1*x*x+fringe+gate[:, None]*(.2+.04*x))
        raw = DataCube(energy, gate, background, 'Gate', 'synthetic', 'counts')
        settings = BackgroundFitSettings(automatic_protection=True)
        for name, centers, amplitude in (
                ('weak stationary', np.full(len(gate), 1.79), .005),
                ('moving', 1.56+.12*(gate+1)/2, .03)):
            with self.subTest(name=name):
                peak = compact_peak(energy, centers, .004, amplitude)
                # No padding beyond the exact compact peak support.
                anchors = abs(energy[None, :]-centers[:, None])>=.004
                baseline, first = estimate_background(raw, settings, anchor_masks=anchors, assess_quality=False)
                altered = replace(raw, Z=raw.Z*(1+peak))
                result, second = estimate_background(altered, settings, anchor_masks=anchors, assess_quality=False)
                np.testing.assert_array_equal(second.background(), first.background())
                np.testing.assert_array_equal(second.log_texture, first.log_texture)
                np.testing.assert_array_equal(second.texture_coverage, first.texture_coverage)
                np.testing.assert_allclose(result.Z-baseline.Z, raw.Z/first.background()*peak, rtol=0, atol=5e-16)

    def test_partial_refits_honour_cancellation(self):
        values = np.ones((80, len(self.energy)))
        anchors = np.ones_like(values, bool)
        anchors[:, abs(self.energy-1.7)<.004] = False
        structure._local_quadratic_trend(self.energy, values, .035)  # Warm the existing projection.
        cancellation = LoadCancellation()
        eigenvalues = np.linalg.eigvalsh

        def cancel_during_refit(normal):
            result = eigenvalues(normal)
            cancellation.cancel()
            return result

        with load_scope(cancellation), patch.object(np.linalg, 'eigvalsh', side_effect=cancel_during_refit):
            with self.assertRaises(LoadCancelled):
                structure._masked_local_quadratic_trend(self.energy, values, anchors, .035)


if __name__ == '__main__':
    unittest.main()
