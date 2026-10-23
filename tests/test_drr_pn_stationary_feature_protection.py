"""Stationary sample features must survive final processing, not gate centering.

These analytic examples reproduce the failure mode at the measured weak-feature
energies; real-data evidence lives in artifacts/pn_feature_protection_20261021.
Passing these tests does not scientifically validate arbitrary P/N backgrounds.
"""
import copy
import inspect
import unittest

import numpy as np

from core.drr_pn_reconstruction.feature_protection import apply_feature_protection
from core.loader import DataCube
from core.processing import apply_sg_derivative_energy


class StationaryFeatureProtectionTests(unittest.TestCase):
    @staticmethod
    def fixture():
        energy = np.linspace(1.58, 1.95, 1481)
        gates = np.array([0., 5., 10., 20., 30.])
        source_p = np.exp(3.5 + .2*(energy-1.8))
        # Fixed across every gate: subtracting the gate mean would discard the
        # very signals this regression is intended to protect.
        signal = (-.0018*np.exp(-((energy-1.8436188930495865)/.0014)**2)
                  -.0009*np.exp(-((energy-1.8487636765647586)/.0012)**2))
        source_n = source_p*(1+signal)
        moving_elsewhere = np.array([
            .0008*np.exp(-((energy-(1.74+.0008*g))/.0015)**2) for g in gates])
        measurements = np.array([
            source_p*(1+scale*signal+moving_elsewhere)
            for scale in (1., .99, 1.02, 1.01)])
        return energy, gates, np.vstack((source_p, source_n)), measurements

    @staticmethod
    def derivative(energy, gates, intensity, background):
        cube = DataCube(energy, gates, intensity/background-1, 'TG+BG (V)', '', 'DR/R')
        result, window = apply_sg_derivative_energy(
            cube, derivative=2, window_length=21, polyorder=2)
        assert window == 21
        np.testing.assert_array_equal(result.energy, energy)
        return result.Z

    def test_heldout_stationary_peaks_survive_full_axis_sg21_without_gate_centering(self):
        e, gates, pn, samples = self.fixture()
        parent = pn[1].copy()
        regions = [dict(xmin_ev=1.82, xmax_ev=1.85, source='P')]
        background, arrays, report = apply_feature_protection(e, parent, pn, regions)
        core = (e >= 1.82) & (e <= 1.85)
        offset, tilt, center = arrays['source_affine_parameters'][0]
        source_control = pn[0]*np.exp(offset+tilt*(e-center))
        for repeat in (1, 3):
            with self.subTest(heldout_repeat=repeat):
                control = self.derivative(e, gates, samples[repeat], source_control)
                protected = self.derivative(e, gates, samples[repeat], background)
                before = self.derivative(e, gates, samples[repeat], parent)
                np.testing.assert_allclose(protected[:, core], control[:, core],
                                           rtol=2e-10, atol=2e-8)
                for center in (1.8436188930495865, 1.8487636765647586):
                    index = np.argmin(abs(e-center))
                    # The synthetic stationary dips have positive final d2 at
                    # both centers; their sign and every gate must be retained.
                    self.assertTrue(np.all(control[:, index] > 0))
                    self.assertGreater(protected[:, index].mean(),
                                       10*abs(before[:, index].mean()))
        # Compact curvature support permits affine exterior intensity changes;
        # demanding exact exterior intensity would force harmful closure terms.
        self.assertTrue(np.all(np.isfinite(background) & (background > 0)))
        self.assertFalse(report['scientifically_validated'])
        self.assertFalse(report['sample_used'])
        self.assertFalse(report['BN_used'])

    def test_source_swap_and_readonly_inputs_do_not_introduce_sample_fit(self):
        e, gates, pn, samples = self.fixture()
        parent = pn[1]*np.exp(.03+.07*(e-1.835))
        regions = [dict(xmin_ev=1.82, xmax_ev=1.85, source='P')]
        original = (e.copy(), pn.copy(), parent.copy(), copy.deepcopy(regions))
        for array in (e, pn, parent):
            array.setflags(write=False)
        background, arrays, report = apply_feature_protection(e, parent, pn, regions)
        swapped, swapped_arrays, _ = apply_feature_protection(
            e, parent, pn[::-1], [dict(xmin_ev=1.82, xmax_ev=1.85, source='N')])
        np.testing.assert_array_equal(background, swapped)
        np.testing.assert_array_equal(arrays['delta_log'], swapped_arrays['delta_log'])
        for actual, before in zip((e, pn, parent), original[:3]):
            np.testing.assert_array_equal(actual, before)
        self.assertEqual(regions, original[3])
        self.assertFalse(any('sample' in name or 'training' in name
                             for name in inspect.signature(apply_feature_protection).parameters))
        # Independent source control includes only the declared log affine
        # gauge. This separates preserved line shape from an amplitude boost.
        offset, tilt, center = arrays['source_affine_parameters'][0]
        control_background = pn[0]*np.exp(offset+tilt*(e-center))
        core = arrays['core_mask']
        for repeat in (1, 3):
            expected = self.derivative(e, gates, samples[repeat], control_background)
            actual = self.derivative(e, gates, samples[repeat], background)
            np.testing.assert_allclose(actual[:, core], expected[:, core],
                                       rtol=2e-10, atol=2e-8)
        outside = ~arrays['curvature_support_mask']
        edges = np.flatnonzero(np.diff(np.r_[False, outside, False].astype(int)))
        for start, stop in zip(edges[::2], edges[1::2]):
            x = e[start:stop]
            difference = np.log(background[start:stop])-np.log(parent[start:stop])
            polynomial = np.polyfit(x-x.mean(), difference, 1)
            residual = difference-np.polyval(polynomial, x-x.mean())
            self.assertLess(np.max(abs(residual)), 2e-11)
        self.assertFalse(report['scientifically_validated'])
        self.assertTrue(any('amplitude' in note or 'amplitudes' in note
                            for note in report['limitations']))


if __name__ == '__main__':
    unittest.main()
