import numpy as np
import unittest
from dataclasses import replace
from copy import deepcopy
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_single import evaluate_bn_single, single_pair_validation_passed, _projected_drr_check
from core.drr_bn_joint import production_d2
from core.load_cancellation import LoadCancelled


def fixture():
    e = np.linspace(1.4, 2.05, 700)
    p = np.arange(len(e)); fine = .02*np.sin(p*.14)+.008*np.sin(p*.23)
    b = np.exp(8+fine)
    config = BnCalibrationSettings(single_measurement=True, single_full_range=False, min_improvement_fraction=.01)
    texture = np.log(b)-savgol_filter(np.log(b), config.template_window, config.template_polyorder)
    shifted = b*np.exp(CubicSpline(p, texture)(p+.5)-texture)
    gates = np.arange(5.)
    z = np.array([shifted*(1+i*.01) for i in gates])
    return e, z, b, gates, config


class TestBnSingle(unittest.TestCase):
    def test_one_scan_shift_produces_exploratory_candidate_and_preserves_protected_region(self):
        e,z,b,g,c = fixture()
        result = evaluate_bn_single(e,z,b,gates=g,settings=c)
        assert result.eligible, result.reasons
        assert result.status == 'single_pair'
        assert single_pair_validation_passed(result.report, c)
        assert result.report['independent_validation'] is False
        assert result.report['joint_validation']['passed'] is False
        assert np.array_equal(result.applied_background[(e>=1.58)&(e<=1.86)],b[(e>=1.58)&(e<=1.86)])


    def test_matched_and_scalar_scaled_background_do_not_make_candidate(self):
        e,z,b,g,c = fixture()
        matched = np.array([b*(1+i*.01) for i in g])
        for factor in (1., 1.7):
            result = evaluate_bn_single(e,matched,b*factor,gates=g,settings=c)
            assert not result.eligible
            assert result.applied_method == 'original'


    def test_invalid_and_cancellation(self):
        e,z,b,g,c = fixture()
        b[10] = 0
        assert not evaluate_bn_single(e,z,b,gates=g,settings=c).eligible
        with self.assertRaises(LoadCancelled):
            evaluate_bn_single(e,z,b,gates=g,settings=c,cancellation=lambda: True)

    def test_contract_rejects_tampering_and_overlapping_regions(self):
        e,z,b,g,c = fixture()
        result = evaluate_bn_single(e,z,b,gates=g,settings=c)
        self.assertTrue(single_pair_validation_passed(result.report, c))
        report = deepcopy(result.report)
        report['candidates']['single_pair']['parameters']['regions'][0]['shift_pixels'] = 99
        self.assertFalse(single_pair_validation_passed(report, c))
        self.assertFalse(single_pair_validation_passed(result.report, replace(c, max_shift_pixels=.1)))
        report = deepcopy(result.report)
        report['single_pair_validation']['drr_projection_check'] = False
        self.assertFalse(single_pair_validation_passed(report, c))
        report = deepcopy(result.report)
        region = next(r for r in report['candidates']['single_pair']['parameters']['regions'] if r.get('accepted'))
        region['drr_projection']['passed'] = False
        self.assertFalse(single_pair_validation_passed(report, c))
        overlap = replace(c, single_regions=((1.45, 1.55), (1.50, 1.56)))
        self.assertIn('overlapping_regions', evaluate_bn_single(e,z,b,gates=g,settings=overlap).reasons)

    def test_weak_peaks_refit_and_protected_peak_remains_exact(self):
        e,z,b,g,c = fixture()
        baseline = evaluate_bn_single(e,z,b,gates=g,settings=c)
        self.assertTrue(baseline.eligible)
        for center in (1.70, 1.505):
            peak = 2e-4*np.exp(-.5*((e-center)/.002)**2)
            injected = z*(1+peak)
            refit = evaluate_bn_single(e,injected,b,gates=g,settings=c)
            response = injected/refit.applied_background-z/baseline.applied_background
            self.assertTrue(np.isfinite(response).all())
            if center == 1.70:
                mask = (e>=1.58)&(e<=1.86)
                self.assertTrue(np.allclose(response[:, mask], (z/b*peak)[:, mask], atol=1e-12, rtol=0))
            # In the fitted band the algorithm may absorb a physical peak;
            # finite response is not a certificate of feature preservation.

    def test_invalid_configuration_falls_back(self):
        e,z,b,g,c = fixture()
        for bad in (replace(c, taper_width=float('inf')), replace(c, gate_count=0),
                    replace(c, max_shift_pixels=float('inf'))):
            self.assertFalse(evaluate_bn_single(e,z,b,gates=g,settings=bad).eligible)

    def test_bn_correlated_weak_feature_can_be_absorbed_after_refitting(self):
        e,z,b,g,c = fixture()
        baseline = evaluate_bn_single(e,z,b,gates=g,settings=c)
        fine = np.log(b)-savgol_filter(np.log(b),c.template_window,c.template_polyorder)
        feature = 1e-3*fine*np.exp(-.5*((e-1.505)/.012)**2)
        injected = z*(1+feature)
        refit = evaluate_bn_single(e,injected,b,gates=g,settings=c)
        self.assertTrue(baseline.eligible and refit.eligible)
        mask = (e>1.47)&(e<1.54)
        response = (injected/refit.applied_background-z/baseline.applied_background)[0, mask]
        expected = (z/b*feature)[0, mask]
        recovery = float(response@expected/(expected@expected))
        # A BN-correlated sample feature can be partially fitted away.
        self.assertGreater(recovery, .4)
        self.assertLess(recovery, .9)

    def test_large_bn_texture_shift_is_resolved(self):
        e,z,b,g,c = fixture()
        p = np.arange(len(e))
        fine = np.log(b)-savgol_filter(np.log(b),c.template_window,c.template_polyorder)
        shifted = b*np.exp(CubicSpline(p,fine)(p+3.0)-fine)
        z = np.array([shifted*(1+i*.01) for i in g])
        result = evaluate_bn_single(e,z,b,gates=g,settings=c)
        self.assertTrue(result.eligible, result.reasons)
        regions = result.report['candidates']['single_pair']['parameters']['regions']
        self.assertTrue(any(r.get('accepted') and abs(r['shift_pixels']) > .75 for r in regions))
        self.assertTrue(single_pair_validation_passed(result.report,c))

    def test_flat_bn_does_not_invent_a_shift(self):
        e,z,b,g,c = fixture()
        b = np.full_like(b, 1000.)
        z = np.broadcast_to(b, z.shape).copy()
        result = evaluate_bn_single(e,z,b,gates=g,settings=c)
        self.assertFalse(result.eligible)
        self.assertEqual(result.applied_method, 'original')

    def test_drr_projection_rejects_even_when_d2_improves(self):
        e = np.linspace(1.4,2.05,700)
        x = np.linspace(0,1,len(e))
        # The candidate loses high-frequency curvature while adding BN-like
        # low-frequency texture: D2 improves, DR/R projection degrades.
        bn_texture = np.sin(2*np.pi*8*x)
        old = np.array([.003*np.sin(2*np.pi*40*x)])
        new = np.array([.003*bn_texture])
        mask = np.ones(len(e),bool)
        q,_ = np.linalg.qr(np.column_stack((np.ones(len(e)),x,x*x)))
        config = BnCalibrationSettings(single_measurement=True)
        old_d2,_ = production_d2(e,old,config)
        new_d2,_ = production_d2(e,new,config)
        self.assertLess(np.linalg.norm(new_d2),np.linalg.norm(old_d2))
        check = _projected_drr_check(old,new,np.array([bn_texture]),mask,q,config)
        self.assertFalse(check['passed'])
        self.assertGreater(check['new_mean'],check['old_mean'])

    def test_full_range_changes_middle_energy_without_default_protection(self):
        e,z,b,g,c = fixture()
        c = replace(c, single_full_range=True)
        from core.drr_bn_calibration import _smoothstep
        pix = np.arange(len(e))
        fine = np.log(b)-savgol_filter(np.log(b),c.template_window,c.template_polyorder)
        support = _smoothstep((e-e[0])/c.taper_width)*_smoothstep((e[-1]-e)/c.taper_width)
        shifted = b*np.exp(support*(CubicSpline(pix,fine)(pix+.5)-fine))
        z = np.array([shifted*(1+i*.01) for i in g])
        result = evaluate_bn_single(e,z,b,gates=g,settings=c)
        self.assertTrue(result.eligible, result.reasons)
        scope = result.report['single_pair_scope']
        self.assertEqual(scope['mode'], 'full_range')
        self.assertEqual(scope['protected_regions'], [])
        self.assertTrue(np.any(np.abs(result.applied_background[(e>1.65)&(e<1.75)]-b[(e>1.65)&(e<1.75)]) > 1e-9))
        self.assertEqual(result.report['single_pair_validation']['protected_pixels'],0)
