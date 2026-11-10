import unittest
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from core.drr_bn_single_smooth import fit_smooth_texture, energy_balanced_bn_d2_score
from core.drr_bn_settings import BnCalibrationSettings

class TestSmoothBn(unittest.TestCase):
    def test_recovers_smooth_global_texture_on_synthetic_scan(self):
        e=np.linspace(1.4,2.05,700); pixel=np.arange(len(e))
        b=np.exp(8+.02*np.sin(pixel*.14)+.008*np.sin(pixel*.23))
        c=BnCalibrationSettings(single_measurement=True)
        fine=np.log(b)-savgol_filter(np.log(b),c.template_window,c.template_polyorder)
        shifted=b*np.exp(CubicSpline(pixel,fine,bc_type='natural')(pixel+2)-fine)
        z=np.stack([shifted*(1+.01*i) for i in range(5)])
        candidate,info=fit_smooth_texture(e,z,b,fine,c)
        self.assertTrue(info['success'])
        self.assertTrue(np.isfinite(candidate).all())
        self.assertLess(np.linalg.norm(candidate-shifted),np.linalg.norm(b-shifted))
        self.assertEqual(len(info['shift_knots']),6)
        self.assertEqual(len(info['gain_knots']),6)
        self.assertTrue(all(abs(v)<=10 for v in info['shift_knots']))

    def test_flat_bn_has_no_fit_basis(self):
        e=np.linspace(1.4,2.05,700); b=np.full(len(e),1000.)
        c=BnCalibrationSettings(single_measurement=True)
        candidate,info=fit_smooth_texture(e,np.stack([b,b]),b,np.zeros_like(b),c)
        self.assertIsNone(candidate)
        self.assertEqual(info['reason'],'no_resolvable_bn_basis')

    def test_refit_weak_peak_is_measured_not_assumed_preserved(self):
        e=np.linspace(1.4,2.05,700); pixel=np.arange(len(e))
        b=np.exp(8+.02*np.sin(pixel*.14)+.008*np.sin(pixel*.23))
        c=BnCalibrationSettings(single_measurement=True)
        fine=np.log(b)-savgol_filter(np.log(b),c.template_window,c.template_polyorder)
        shifted=b*np.exp(CubicSpline(pixel,fine,bc_type='natural')(pixel+.5)-fine)
        z=np.stack([shifted*(1+.01*i) for i in range(5)])
        peak=2e-4*np.exp(-.5*((e-1.7)/.002)**2)
        original,_=fit_smooth_texture(e,z,b,fine,c)
        injected,_=fit_smooth_texture(e,z*(1+peak),b,fine,c)
        i=np.argmin(abs(e-1.7))
        recovery=((z*(1+peak)/injected-z/original)[0,i]/(z/b*peak)[0,i])
        self.assertTrue(np.isfinite(recovery))
        # This tests one synthetic peak, not universal physical preservation.
        self.assertGreater(recovery,.5)
        self.assertLess(recovery,1.5)

    def test_energy_balanced_score_improves_for_removed_bn_texture(self):
        e=np.linspace(1.4,2.05,700); pixel=np.arange(len(e))
        b=np.exp(8+.02*np.sin(pixel*.14)+.008*np.sin(pixel*.23))
        c=BnCalibrationSettings(single_measurement=True)
        fine=np.log(b)-savgol_filter(np.log(b),c.template_window,c.template_polyorder)
        from core.drr_bn_joint import production_d2
        old,_=production_d2(e,np.stack([fine,1.1*fine]),c)
        score=energy_balanced_bn_d2_score(e,fine,old,.5*old,c)
        self.assertEqual(score['metric'],'energy_balanced_bn_projected_d2_rms')
        self.assertEqual(len(score['windows']),6)
        self.assertAlmostEqual(score['candidate']/score['original'],.5,places=6)

    def test_energy_score_is_invariant_to_window_amplitude(self):
        e=np.linspace(1.4,2.05,720); pixel=np.arange(len(e))
        fine=.02*np.sin(pixel*.14)+.008*np.sin(pixel*.23)
        c=BnCalibrationSettings(single_measurement=True)
        from core.drr_bn_joint import production_d2
        old,_=production_d2(e,np.stack([fine,1.1*fine]),c)
        ratios=[.3,.5,.7,.9,.8,.4]
        candidate=old.copy()
        for ids,ratio in zip(np.array_split(np.arange(len(e)),6),ratios):
            candidate[:,ids]*=ratio
        first=energy_balanced_bn_d2_score(e,fine,old,candidate,c)
        altered=old.copy(); altered_candidate=candidate.copy()
        for ids,scale in zip(np.array_split(np.arange(len(e)),6),[1,7,3,100,2,10]):
            altered[:,ids]*=scale; altered_candidate[:,ids]*=scale
        second=energy_balanced_bn_d2_score(e,fine,altered,altered_candidate,c)
        self.assertAlmostEqual(first['original'],1.,places=12)
        self.assertAlmostEqual(first['candidate'],np.mean(ratios),places=12)
        self.assertAlmostEqual(first['candidate'],second['candidate'],places=12)

    def test_zero_reference_window_is_not_a_valid_relative_score(self):
        e=np.linspace(1.4,2.05,720); x=np.arange(len(e)); fine=.02*np.sin(x*.14)
        c=BnCalibrationSettings(single_measurement=True)
        old=np.tile(fine,(2,1)); old[:,np.array_split(np.arange(len(e)),6)[2]]=0
        self.assertIsNone(energy_balanced_bn_d2_score(e,fine,old,old,c))
