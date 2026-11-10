import unittest
from unittest.mock import patch
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_joint import production_d2
from core.drr_bn_single_phase import fit_phase_texture, _d2_shrink_factor
from core.load_cancellation import LoadCancellation, LoadCancelled


class SingleBnPhaseTests(unittest.TestCase):
    def setUp(self):
        self.energy = np.linspace(1.4, 2., 141)
        x = np.arange(len(self.energy))
        self.fine = .01*np.sin(2*np.pi*x/20)
        self.background = np.exp(8+self.fine)
        self.sample = np.stack([self.background*np.exp(.005*np.sin(2*np.pi*x/20+.3+.1*x/len(x)) + .001*i)
                                for i in range(4)])
        self.settings = BnCalibrationSettings(single_measurement=True, single_full_range=True)

    def test_phase_candidate_and_final_guard_report(self):
        candidate, info = fit_phase_texture(self.energy, self.sample, self.background, self.fine, self.settings)
        self.assertEqual(candidate.shape, self.energy.shape)
        self.assertTrue(np.all(np.isfinite(candidate) & (candidate > 0)))
        self.assertEqual(info['node_count'], 10)
        self.assertEqual(info['basis_source'], 'measured_bn_only')
        self.assertEqual(info['d2_shrink_policy'], 'ratio_dependent_v1')
        self.assertEqual(len(info['phase_knots']), 10)
        self.assertEqual(len(info['gain_knots']), 10)
        self.assertGreater(info['checked_d2_windows'], 0)
        def independently_check(normalized):
            def curve(background):
                ratio = self.sample/background
                if normalized:
                    ratio = ratio/np.median(ratio, axis=1, keepdims=True)
                return production_d2(self.energy, ratio-1, self.settings)[0]
            before, after = curve(self.background), curve(candidate)
            span = max(self.settings.local_validation_window, info['used_window'])
            starts = sorted(set([*range(0, len(self.energy)-span+1, span//2), len(self.energy)-span]))
            return all(np.all(np.sqrt(np.mean(after[:, i:i+span]**2, axis=1)) <=
                    np.sqrt(np.mean(before[:, i:i+span]**2, axis=1))*(1+self.settings.max_local_degradation_fraction)
                    + self.settings.max_absolute_d2_degradation) for i in starts)
        self.assertEqual(info['all_gate_raw_d2_pass'], independently_check(False))
        self.assertEqual(info['all_gate_normalized_d2_pass'], independently_check(True))
        self.assertEqual(info['success'], info['fit_success'] and info['finite_positive'] and
                         info['all_gate_raw_d2_pass'] and info['all_gate_normalized_d2_pass'] and info['full_axis_drr_pass'])
        self.assertTrue(all(self.settings.single_smooth_gain_bounds[0] <= v <= self.settings.single_smooth_gain_bounds[1]
                            for v in info['gain_knots']))
        reverse, reverse_info = fit_phase_texture(self.energy[::-1], self.sample[:, ::-1], self.background[::-1],
                                                  self.fine[::-1], self.settings)
        np.testing.assert_allclose(reverse[::-1], candidate, rtol=1e-10, atol=1e-10)
        self.assertEqual(reverse_info['success'], info['success'])

    def test_near_threshold_shrink_is_proportional_not_a_fixed_step(self):
        before=np.array([10.,20.]); allowance=np.array([1.,2.])
        near=_d2_shrink_factor(before,before+allowance*1.001,allowance)
        farther=_d2_shrink_factor(before,before+allowance*1.01,allowance)
        self.assertGreater(near,.97)
        self.assertLess(abs(near-farther),.01)
        self.assertLess(farther,near)
        self.assertAlmostEqual(_d2_shrink_factor(before,before+allowance*2,allowance),.49)

    def test_cancelled_before_fitting(self):
        token = LoadCancellation()
        token.cancel()
        with self.assertRaises(LoadCancelled):
            fit_phase_texture(self.energy, self.sample, self.background, self.fine, self.settings, token)

    def test_all_gate_failures_cannot_be_masked_out(self):
        # A deliberately failing full-axis check must never certify a candidate.
        with patch('core.drr_bn_single._full_axis_drr_check', return_value=[dict(region=[1.4, 2.], passed=False)]):
            _, info = fit_phase_texture(self.energy, self.sample, self.background, self.fine, self.settings)
        self.assertFalse(info['success'])
        self.assertFalse(info['full_axis_drr_pass'])

    def test_phase_fallback_contract_and_tampering(self):
        from scipy.signal import hilbert, savgol_filter
        from core.drr_bn_single import evaluate_bn_single, single_pair_validation_passed
        e=np.linspace(1.4,2.05,700); x=np.arange(len(e))
        b=np.exp(8+.025*np.sin(.18*x)+.01*np.sin(.31*x))
        fine=np.log(b)-savgol_filter(np.log(b),101,3)
        desired=b*np.exp(np.real(hilbert(fine)*np.exp(1j*(.5+.3*x/len(x))))-fine)
        g=np.linspace(-1,1,7); z=desired*(1+.01*g[:,None])
        # Exercise routing when the simpler model families fail; the phase
        # solver and its final all-gate checks still run without mocks.
        failed=SimpleNamespace(success=False,fun=np.ones(10),x=np.array([0.,1.]))
        with patch('core.drr_bn_single.least_squares',return_value=failed), \
             patch('core.drr_bn_single_smooth.fit_smooth_texture',return_value=(None,{'success':False})):
            result=evaluate_bn_single(e,z,b,gates=g,settings=self.settings)
        self.assertTrue(result.eligible,result.reasons)
        self.assertEqual(result.report['single_pair_scope']['model'],'phase_ten_knots')
        self.assertTrue(single_pair_validation_passed(result.report,self.settings))
        self.assertFalse(np.array_equal(result.applied_background,b))
        for key,value in [('all_gate_raw_d2_pass',False),('strength_max',1.1),
                          ('d2_shrink_policy','fixed_step'),
                          ('phase_limit_radians',float('nan')),('phase_knots',[0.]*9)]:
            report=deepcopy(result.report)
            report['candidates']['single_pair']['parameters']['regions'][0]['model_parameters'][key]=value
            self.assertFalse(single_pair_validation_passed(report,self.settings),key)


if __name__ == '__main__':
    unittest.main()
