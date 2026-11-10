import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from scipy.signal import savgol_filter
from core.drr_bn_single_high_frequency import refine_shared_high_frequency
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_joint import production_d2
from core.load_cancellation import LoadCancelled

class SharedHighFrequencyTests(unittest.TestCase):
    def fixture(self):
        n=1340;e=np.linspace(1.43,2.10,n);x=np.arange(n);g=np.linspace(-1,1,7)
        b=np.exp(8+.015*np.sin(2*np.pi*.05*x));texture=.004*np.sin(2*np.pi*.28*x)
        z=b*np.exp(texture)*(1+.03*g[:,None]);fine=np.log(b)-savgol_filter(np.log(b),101,3)
        return e,g,b,z,fine,BnCalibrationSettings(single_measurement=True)
    def test_dense_hf_reduces_response_with_every_original_guard(self):
        e,g,b,z,f,c=self.fixture();candidate,info=refine_shared_high_frequency(e,z,b,b,f,c)
        self.assertTrue(info['passed'],info.get('reason'))
        self.assertEqual(info['source'],'sample_shared_high_frequency')
        self.assertEqual(info['schema_version'],1)
        self.assertEqual(info['estimator'],'gate_log_mean_v1')
        self.assertEqual(info['shrink_policy'],'monotone_smoothed_factor_v1')
        self.assertEqual(info['baseline_source'],'phase_candidate')
        self.assertEqual(info['cutoff_cycles_per_pixel'],.20)
        self.assertLess(info['candidate_shared_hf_rms'],info['original_shared_hf_rms']*.9)
        self.assertTrue(info['all_gate_raw_d2_pass'] and info['all_gate_normalized_d2_pass'])
        self.assertTrue(info['full_axis_drr_pass'] and info['whole_axis_drr_pass'])
        self.assertEqual(info['failed_d2_windows'],[])
        old,_=production_d2(e,z/b-1,c);new,_=production_d2(e,z/candidate-1,c)
        span=c.local_validation_window
        for i in sorted(set([*range(0,len(e)-span+1,span//2),len(e)-span])):
            x=np.sqrt(np.mean(old[:,i:i+span]**2,axis=1));y=np.sqrt(np.mean(new[:,i:i+span]**2,axis=1))
            self.assertTrue(np.all(y<=x*(1+c.max_local_degradation_fraction)+c.max_absolute_d2_degradation))
    def test_fixed_and_moving_2mev_peaks_full_refit(self):
        e,g,b,z,f,c=self.fixture();baseline,info=refine_shared_high_frequency(e,z,b,b,f,c)
        self.assertTrue(info['passed'])
        for moving in (False,True):
            centers=1.8+g*.0025 if moving else np.full(len(g),1.8)
            peak=.0002*np.exp(-.5*((e[None,:]-centers[:,None])/.002)**2);modified=z*(1+peak)
            candidate,check=refine_shared_high_frequency(e,modified,b,b,f,c)
            self.assertTrue(check['passed'])
            recovered=[]
            for j,center in enumerate(centers):
                i=np.argmin(abs(e-center));recovered.append((modified[j,i]/candidate[i]-z[j,i]/baseline[i])/(z[j,i]/b[i]*peak[j,i]))
            # One synthetic peak only; this is not a universal fidelity guarantee.
            self.assertTrue(np.all((np.array(recovered)>.95)&(np.array(recovered)<1.05)))
    def test_low_frequency_is_not_removed_and_fewer_than_three_gates_fall_back(self):
        e,g,b,z,f,c=self.fixture(); x=np.arange(len(e)); low=.002*np.sin(2*np.pi*.15*x)
        z=z*np.exp(low)
        candidate,info=refine_shared_high_frequency(e,z,b,b,f,c)
        self.assertTrue(info['passed'],info.get('reason'))
        # The added denominator must not carry away the 0.15-cycle/pixel mode.
        correction=np.log(candidate/b)
        recovered=2*np.dot(correction,np.sin(2*np.pi*.15*x))/len(x)
        self.assertLess(abs(recovered),.0002)
        for count in (1,2):
            candidate,info=refine_shared_high_frequency(e,z[:count],b,b,f,c)
            self.assertFalse(info['passed'])
            self.assertEqual(info['reason'],'insufficient_gates_for_shared_response')
            np.testing.assert_array_equal(candidate,b)

    def test_flat_input_and_failed_guards_keep_phase(self):
        e,g,b,z,f,c=self.fixture();flat=np.broadcast_to(b,(len(g),len(b))).copy()
        candidate,info=refine_shared_high_frequency(e,flat,b,b,f,c)
        self.assertFalse(info['passed']);np.testing.assert_array_equal(candidate,b)
        with patch('core.drr_bn_single._full_axis_drr_check',return_value=[{'region':[e[0],e[-1]],'passed':False}]):
            candidate,info=refine_shared_high_frequency(e,z,b,b,f,c)
        self.assertFalse(info['passed']);np.testing.assert_array_equal(candidate,b)
    def test_cancel_invalid_and_reverse(self):
        e,g,b,z,f,c=self.fixture()
        with self.assertRaises(LoadCancelled):refine_shared_high_frequency(e,z,b,b,f,c,cancellation=lambda: True)
        candidate,info=refine_shared_high_frequency(e,z,b,b,f,replace(c,max_absolute_d2_degradation=float('nan')))
        self.assertFalse(info['passed']);np.testing.assert_array_equal(candidate,b)
        forward,fi=refine_shared_high_frequency(e,z,b,b,f,c)
        backward,bi=refine_shared_high_frequency(e[::-1],z[:,::-1],b[::-1],b[::-1],f[::-1],c)
        self.assertTrue(fi['passed'] and bi['passed'])
        np.testing.assert_allclose(forward,backward[::-1],rtol=1e-12,atol=1e-9)
