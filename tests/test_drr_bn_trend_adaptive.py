"""Behavioral tests for data-selected shared BN trends and their change guards."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from core.drr_bn_joint import production_d2
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_trend import evaluate_bn_trend, trend_validation_passed
from core.load_cancellation import LoadCancelled


class AdaptiveTrendTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1.4, 2.1, 401)
        self.g = np.linspace(-1, 1, 9)
        self.b = 1000*(1+.01*np.sin(np.arange(len(self.e))*.2))
        self.x = (self.e-1.75)/.35
        self.s = BnCalibrationSettings(single_measurement=True, single_trend_enabled=True)

    def data(self, amplitude=.08, variation=0):
        peak = .02*np.exp(-((self.e[None, :]-1.75-.02*self.g[:, None])/.01)**2)
        return self.b*np.exp(amplitude+.003*self.x)[None, :]*(1+peak)*np.exp(variation*self.g[:, None])

    def run_fit(self, raw, **kwargs):
        return evaluate_bn_trend(self.e, raw, self.b, gates=self.g, settings=kwargs.pop('settings', self.s), **kwargs)

    def test_per_csv_strength_shared_gain_and_unchanged_inputs(self):
        strengths = []
        for amplitude in (.01, .08, .2):
            raw = self.data(amplitude)
            original = raw.copy()
            result = self.run_fit(raw)
            self.assertTrue(result.eligible, result.report)
            self.assertTrue(trend_validation_passed(result.report, self.s))
            model = result.report['models'][result.selected_method]
            strengths.append(model['strength'])
            self.assertLessEqual(model['d2_guard']['max_relative_change'], .05)
            np.testing.assert_array_equal(raw, original)
            np.testing.assert_allclose((raw/result.applied_background)[0]/(raw/result.applied_background)[-1],
                                       raw[0]/raw[-1], rtol=1e-12)
            final = model['attempts'][model['selected_attempt']]
            self.assertTrue(final['passed'])
        self.assertEqual(strengths[0], 1.)
        self.assertGreater(strengths[1], .5)  # refinement recovers room between coarse strengths
        self.assertGreater(strengths[1], strengths[2])

    def test_data_changes_window_selection(self):
        first = self.run_fit(self.data(.01))
        left = first.report['background_regions_ev'][0]
        mask = (self.e >= left[0]) & (self.e <= left[1])
        contaminated = self.data(.01)
        contaminated[:, mask] *= np.exp(.05*self.g[:, None])
        second = self.run_fit(contaminated)
        self.assertNotEqual(first.report['background_regions_ev'], second.report['background_regions_ev'])
        self.assertTrue(second.eligible, second.report)

    def test_gate_disagreement_shrinks_or_falls_back(self):
        weak = self.run_fit(self.data(.01, .2))
        self.assertTrue(weak.eligible)
        self.assertLess(weak.report['models'][weak.selected_method]['strength'], .25)
        strong = self.run_fit(self.data(.01, 1.))
        self.assertFalse(strong.eligible)
        self.assertFalse(trend_validation_passed(strong.report, self.s))
        np.testing.assert_array_equal(strong.applied_background, self.b)

    def test_production_d2_guard_recomputed_from_output(self):
        raw = self.data(.2)
        result = self.run_fit(raw)
        model = result.report['models'][result.selected_method]
        old, _ = production_d2(self.e, raw/self.b-1, self.s)
        new, _ = production_d2(self.e, raw/result.applied_background-1, self.s)
        width = model['d2_guard']['window_points']
        starts = sorted(set(list(range(0, len(self.e)-width+1, width//2))+[len(self.e)-width]))
        floor = max(1., .01*np.percentile(abs(old), 95))
        changes = [np.sqrt(np.mean((new[:, i:i+width]-old[:, i:i+width])**2, axis=1)) /
                   np.maximum(np.sqrt(np.mean(old[:, i:i+width]**2, axis=1)), floor) for i in starts]
        self.assertAlmostEqual(float(np.max(changes)), model['d2_guard']['max_relative_change'], places=9)
        self.assertLessEqual(float(np.max(changes)), .05)

    def test_reverse_energy_and_gate_order(self):
        raw = self.data()
        first = self.run_fit(raw)
        second = evaluate_bn_trend(self.e[::-1], raw[::-1, ::-1], self.b[::-1],
                                   gates=self.g[::-1], settings=self.s)
        self.assertTrue(second.eligible)
        np.testing.assert_allclose(first.applied_background, second.applied_background[::-1], rtol=1e-11)

    def test_user_regions_override_screening(self):
        settings = replace(self.s, single_trend_regions=((1.41, 1.51), (2., 2.09)))
        result = self.run_fit(self.data(), settings=settings)
        self.assertTrue(result.eligible, result.report)
        self.assertEqual(result.report['background_regions_source'], 'user')
        self.assertTrue(trend_validation_passed(result.report, settings))

    def test_quadratic_and_nonuniform_axis_use_production_settings(self):
        raw = self.data(.01)*np.exp(.001*self.x*self.x)
        settings = replace(self.s, derivative_window=20)
        result = self.run_fit(raw, settings=settings)
        self.assertTrue(result.eligible, result.report)
        self.assertEqual(result.selected_method, 'trend_quadratic')
        self.assertTrue(trend_validation_passed(result.report, settings))
        energy = 1240/np.linspace(880, 600, len(self.e))
        result = evaluate_bn_trend(energy, raw, self.b, gates=self.g, settings=settings)
        self.assertTrue(result.eligible, result.report)
        self.assertTrue(trend_validation_passed(result.report, settings))

    def test_full_refit_weak_fixed_and_moving_peak_injection(self):
        raw = self.data(.01)
        baseline = self.run_fit(raw)
        for moving in (False, True):
            center = 1.83 + (.02*self.g[:, None] if moving else np.zeros((len(self.g), 1)))
            signal = .0005*np.exp(-.5*((self.e[None, :]-center)/.005)**2)
            injected = raw+self.b*signal
            result = self.run_fit(injected)
            self.assertTrue(result.eligible, result.report)
            # Refit the complete algorithm, including window and strength selection.
            observed = injected/result.applied_background-raw/baseline.applied_background
            expected = self.b*signal/baseline.applied_background
            np.testing.assert_allclose(observed, expected, rtol=1e-7, atol=1e-10)

    def test_report_evidence_tampering_rejected(self):
        report = self.run_fit(self.data()).report
        name = report['selected_method']
        mutations = [
            lambda r: r['models'][name]['d2_guard'].__setitem__('floor', float('nan')),
            lambda r: r['models'][name]['d2_guard'].__setitem__('max_absolute_change', -1),
            lambda r: r['models'][name]['d2_guard'].__setitem__('floor_dominated_fraction', 2),
            lambda r: r['models'][name]['cv_blocks'][0].__setitem__('candidate_rms', float('inf')),
            lambda r: r['models'][name]['attempts'].pop(),
            lambda r: r['models'][name].__setitem__('strength', .6),
            lambda r: r['models'][name]['gate_stability']['changes'].__setitem__(0, .8),
            lambda r: r['automatic_background']['selected_indices'].__setitem__(0, -1),
            lambda r: r['trend_validation'].__setitem__('independent_validation', True),
        ]
        for mutate in mutations:
            changed = deepcopy(report)
            mutate(changed)
            self.assertFalse(trend_validation_passed(changed, self.s))

    def test_no_correction_invalid_input_and_cancellation(self):
        result = self.run_fit(np.broadcast_to(self.b, (len(self.g), len(self.b))))
        self.assertFalse(result.eligible)
        bad = self.data()
        bad[0, 20] = np.nan
        self.assertFalse(self.run_fit(bad).eligible)
        self.assertFalse(self.run_fit(self.data(), settings=replace(self.s, single_trend_max_d2_change=0)).eligible)
        with self.assertRaises(LoadCancelled):
            self.run_fit(self.data(), cancellation=lambda: True)


if __name__ == '__main__':
    unittest.main()
