import unittest
from dataclasses import replace
from copy import deepcopy
import numpy as np
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_trend import evaluate_bn_trend, trend_validation_passed
from core.load_cancellation import LoadCancelled

class TrendTests(unittest.TestCase):
 def setUp(self):
  self.e=np.linspace(1.4,2.1,301);self.g=np.linspace(-1,1,5);self.b=1000*(1+.01*np.sin(np.arange(301)*.2));x=(self.e-1.75)/.35
  self.z=self.b[None,:]*np.exp(.12+.015*x)[None,:]*(1+self.g[:,None]*.0001)
  self.s=BnCalibrationSettings(single_measurement=True,single_trend_enabled=True,single_trend_adaptive=False)
 def test_linear_auto_shared_and_checker(self):
  e,z,b=self.e.copy(),self.z.copy(),self.b.copy();r=evaluate_bn_trend(e,z,b,gates=self.g,settings=self.s)
  self.assertTrue(r.eligible,r.report);self.assertEqual(r.selected_method,'trend_linear');self.assertEqual(r.status,'single_pair');self.assertTrue(trend_validation_passed(r.report,self.s))
  a=r.applied_background;np.testing.assert_allclose((z/a)[0]/(z/a)[-1],z[0]/z[-1],rtol=1e-12)
  for n in ('trend_linear','trend_quadratic'):
   p=r.report['candidates'][n]['parameters'];v=np.polynomial.polynomial.polyval(e-p['energy_center_ev'],p['coefficients']);np.testing.assert_allclose(r.candidate_backgrounds[n],b*np.exp(v))
  np.testing.assert_array_equal(self.z,z);np.testing.assert_array_equal(self.b,b)
  t=deepcopy(r.report);t['models']['trend_linear']['max_sensitivity']=.4;self.assertFalse(trend_validation_passed(t,self.s))
  t=deepcopy(r.report);t['candidates']['trend_linear']['parameters']['coefficients'][0]+=.01;self.assertFalse(trend_validation_passed(t,self.s))
  t=deepcopy(r.report);t['background_regions_ev'][0][0]+=.01;self.assertFalse(trend_validation_passed(t,self.s))
  self.assertFalse(trend_validation_passed(r.report,replace(self.s,single_trend_order='quadratic')))
 def test_quadratic_auto_and_reverse(self):
  x=(self.e-1.75)/.35;z=self.b[None,:]*np.exp(.1+.01*x+.035*x*x)[None,:]*np.ones((len(self.g),1))
  r=evaluate_bn_trend(self.e,z,self.b,gates=self.g,settings=self.s);self.assertTrue(r.eligible,r.report);self.assertEqual(r.selected_method,'trend_quadratic')
  q=evaluate_bn_trend(self.e[::-1],z[:,::-1],self.b[::-1],gates=self.g,settings=self.s)
  np.testing.assert_allclose(r.applied_background,q.applied_background[::-1],rtol=1e-10)
 def test_no_change_is_not_calibration(self):
  z=np.broadcast_to(self.b,(len(self.g),len(self.b))).copy()
  r=evaluate_bn_trend(self.e,z,self.b,gates=self.g,settings=self.s)
  self.assertFalse(r.eligible);self.assertEqual(r.status,'preview_only');self.assertFalse(trend_validation_passed(r.report,self.s))
 def test_local_peak_not_fit(self):
  r=evaluate_bn_trend(self.e,self.z,self.b,gates=self.g,settings=self.s);peak=np.exp(-.5*((self.e-1.75)/.01)**2);q=evaluate_bn_trend(self.e,self.z*(1+.2*peak),self.b,gates=self.g,settings=self.s)
  np.testing.assert_allclose(r.applied_background,q.applied_background,rtol=1e-10)
 def test_unstable_preview_and_invalid(self):
  z=self.z.copy();z[:,self.e<1.51]*=np.exp(.15*np.sin((self.e[self.e<1.51]-1.4)*80))[None,:]
  r=evaluate_bn_trend(self.e,z,self.b,gates=self.g,settings=replace(self.s,single_trend_max_sensitivity=1e-5));self.assertFalse(r.eligible);self.assertEqual(r.status,'preview_only');self.assertIn(r.selected_method,r.candidate_backgrounds);np.testing.assert_array_equal(r.applied_background,self.b);self.assertFalse(trend_validation_passed(r.report,self.s))
  bad=evaluate_bn_trend(self.e,self.z[:,:-1],self.b,gates=self.g,settings=self.s);self.assertFalse(bad.eligible);self.assertFalse(bad.report['candidate_generated'])
  tiny=evaluate_bn_trend(self.e[:30],self.z[:,:30],self.b[:30],gates=self.g,settings=self.s);self.assertFalse(tiny.eligible)
  with self.assertRaises(LoadCancelled):evaluate_bn_trend(self.e,self.z,self.b,gates=self.g,settings=self.s,cancellation=lambda:True)

if __name__=='__main__':unittest.main()
