import unittest
import numpy as np
from core.drr_bn_gap_prediction import evaluate_bn_gap, fit_energy_gap, GapPredictionSettings
from core.load_cancellation import LoadCancelled

class GapPredictionTests(unittest.TestCase):
    def setUp(self):
        self.e=np.linspace(1.45,2.05,141);self.g=np.linspace(-1,1,5);x=np.arange(len(self.e))
        self.b=1000*np.exp(.01*np.sin(x*.24)+.005*np.sin(x*.5))
        self.z=self.b[None,:]*np.exp(.003*np.sin(x[None,:]*.24+.3)+self.g[:,None]*.001)
    def test_preview_only_preserves_input_and_reverse(self):
        z=self.z.copy();b=self.b.copy();r=evaluate_bn_gap(self.e,z,b,gates=self.g)
        self.assertFalse(r.eligible);self.assertEqual(r.status,'preview_only');self.assertEqual(r.applied_method,'original')
        np.testing.assert_array_equal(r.applied_background,b);np.testing.assert_array_equal(self.z,z);np.testing.assert_array_equal(self.b,b)
        rr=evaluate_bn_gap(self.e[::-1],z[:,::-1],b[::-1],gates=self.g)
        np.testing.assert_allclose(r.candidate_backgrounds['gap_preview'],rr.candidate_backgrounds['gap_preview'][::-1],rtol=1e-10)
        self.assertEqual(r.report['optimization_policy'],'gap_prediction_experimental_v1')
    def test_gap_footprint_and_full_refit_peaks(self):
        opt=GapPredictionSettings(gap_ev=.035,shoulder_ev=.004,continuum_nodes=8,hop=3,include_high_frequency=False)
        base,info=fit_energy_gap(self.e,self.z,self.b,gap_settings=opt)
        self.assertGreaterEqual(info['anchor_exclusion_ev']-info['max_anchor_spacing_ev'],opt.gap_ev-1e-12)
        for moving in (False,True):
            centers=1.75+(self.g*.004 if moving else np.zeros_like(self.g));feature=.0002*np.exp(-.5*((self.e[None,:]-centers[:,None])/.002)**2)
            after,_=fit_energy_gap(self.e,self.z*(1+feature),self.b,gap_settings=opt)
            recovered=(self.z*(1+feature)/after-self.z/base)/(self.z/base*feature+1e-200)
            vals=[recovered[i,np.argmin(abs(self.e-c))] for i,c in enumerate(centers)]
            self.assertTrue(np.all(np.abs(np.asarray(vals)-1)<.05),vals)
    def test_interpolated_query_excludes_full_energy_gap(self):
        opt=GapPredictionSettings(gap_ev=.035,shoulder_ev=.004,continuum_nodes=8,hop=3,include_high_frequency=False)
        query,pulse,left_anchor=70,78,69
        self.assertLess(self.e[pulse]-self.e[query],opt.gap_ev)
        # Without an anchor-spacing margin this pulse would enter the left
        # anchor fit, whose coefficients contribute to the query.
        self.assertGreater(self.e[pulse]-self.e[left_anchor],opt.gap_ev)
        baseline,_=fit_energy_gap(self.e,self.z,self.b,gap_settings=opt)
        perturbed=self.z.copy();perturbed[:,pulse]*=1.1
        changed,_=fit_energy_gap(self.e,perturbed,self.b,gap_settings=opt)
        response=np.log(changed/baseline)
        self.assertLess(abs(response[query]),1e-8)
        self.assertGreater(np.max(abs(response)),1e-5)
    def test_mapping_and_too_few_gates_for_hf(self):
        r=evaluate_bn_gap(self.e,self.z[:2],self.b,gates=self.g[:2],settings={})
        self.assertTrue(r.report['candidate_generated'])
        self.assertEqual(r.report['fit']['high_frequency_source'],'insufficient_gates')
        self.assertEqual(r.report['measurement_count'],1)
        self.assertIn('bn_filter_bands_cycles_per_pixel',r.report['fit_contract'])
    def test_invalid_flat_and_cancelled(self):
        invalid=evaluate_bn_gap(self.e,self.z,self.b,gates=self.g[:-1]);self.assertIn('invalid_input',invalid.reasons)
        flat=evaluate_bn_gap(self.e,self.z,np.ones_like(self.b),gates=self.g);self.assertIn('flat_bn_texture',flat.reasons);self.assertFalse(flat.report['candidate_generated'])
        with self.assertRaises(LoadCancelled):evaluate_bn_gap(self.e,self.z,self.b,gates=self.g,cancellation=lambda:True)

if __name__=='__main__':unittest.main()
