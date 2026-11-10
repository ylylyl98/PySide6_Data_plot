import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.signal import hilbert,savgol_filter
from core.drr_bn_settings import BnCalibrationSettings
from core.drr_bn_single import evaluate_bn_single,single_pair_validation_passed

class HighFrequencyIntegrationTests(unittest.TestCase):
    def setUp(self):
        n=1340;self.e=np.linspace(1.43,2.10,n);x=np.arange(n);self.g=np.linspace(-1,1,7)
        self.b=np.exp(8+.025*np.sin(.18*x)+.01*np.sin(.31*x))
        fine=np.log(self.b)-savgol_filter(np.log(self.b),101,3)
        desired=self.b*np.exp(np.real(hilbert(fine)*np.exp(1j*(.5+.3*x/n)))-fine)
        self.z=desired*np.exp(.003*np.sin(2*np.pi*.26*x))*(1+.01*self.g[:,None])
        self.c=BnCalibrationSettings(single_measurement=True)
    def run_path(self,*,fail_hf=False):
        failed=SimpleNamespace(success=False,fun=np.ones(10),x=np.array([0.,1.]))
        def fallback(e,z,original,phase,fine,settings,cancellation=None):
            return phase.copy(),{'passed':False,'reason':'forced_failure'}
        with patch('core.drr_bn_single.least_squares',return_value=failed), \
             patch('core.drr_bn_single_smooth.fit_smooth_texture',return_value=(None,{'success':False})):
            if fail_hf:
                with patch('core.drr_bn_single_high_frequency.refine_shared_high_frequency',side_effect=fallback):
                    return evaluate_bn_single(self.e,self.z,self.b,gates=self.g,settings=self.c)
            return evaluate_bn_single(self.e,self.z,self.b,gates=self.g,settings=self.c)
    def test_combined_actual_path_and_final_guard_provenance(self):
        result=self.run_path();self.assertTrue(result.eligible,result.reasons)
        self.assertEqual(result.report['single_pair_scope']['model'],'phase_with_shared_high_frequency')
        self.assertTrue(single_pair_validation_passed(result.report,self.c))
        validation=result.report['single_pair_validation']
        self.assertTrue(validation['sample_shared_response'])
        region=result.report['candidates']['single_pair']['parameters']['regions'][0]
        phase=region['model_parameters'];hf=phase['shared_high_frequency']
        self.assertEqual(phase['basis_source'],'measured_bn_and_sample_shared_response')
        self.assertEqual(hf['source'],'sample_shared_high_frequency')
        self.assertTrue(hf['passed'] and hf['energy_balanced_noninferiority_pass'])
        for key in ('all_gate_raw_d2_pass','all_gate_normalized_d2_pass','full_axis_drr_pass',
                    'full_axis_drr_reports','checked_d2_windows','failed_d2_windows'):
            self.assertEqual(phase[key],hf[key])
        self.assertEqual(validation['full_axis_drr_reports'],hf['full_axis_drr_reports'])
        self.assertEqual(region['drr_projection'],hf['whole_axis_drr_projection'])
        self.assertIn('phase_only_guard_reports',phase)
        self.assertFalse(validation['independent_validation'])
    def test_failed_hf_preserves_guarded_phase(self):
        result=self.run_path(fail_hf=True);self.assertTrue(result.eligible,result.reasons)
        self.assertEqual(result.report['single_pair_scope']['model'],'phase_ten_knots')
        self.assertTrue(single_pair_validation_passed(result.report,self.c))
        self.assertFalse(result.report['single_pair_validation']['sample_shared_response'])
        self.assertEqual(result.report['phase_texture_attempt']['shared_high_frequency']['reason'],'forced_failure')
        self.assertFalse(np.array_equal(result.applied_background,self.b))
    def test_combined_contract_rejects_tampering(self):
        result=self.run_path();self.assertTrue(result.eligible)
        changes=[('source','measured_bn_only'),('passed',False),('adopted',False),('cutoff_cycles_per_pixel',.1),
                 ('estimator','masked_median'),('shrink_policy','nonmonotone'),('baseline_source','original_bn'),
                 ('whole_axis_drr_projection',{'passed':False}),('all_gate_raw_d2_pass',False),
                 ('energy_balanced_noninferiority_pass',False),('candidate_shared_hf_rms',1e10)]
        for key,value in changes:
            report=deepcopy(result.report)
            report['candidates']['single_pair']['parameters']['regions'][0]['model_parameters']['shared_high_frequency'][key]=value
            self.assertFalse(single_pair_validation_passed(report,self.c),key)
        report=deepcopy(result.report);report['single_pair_validation']['sample_shared_response']=False
        self.assertFalse(single_pair_validation_passed(report,self.c))
        report=deepcopy(result.report)
        hf=report['candidates']['single_pair']['parameters']['regions'][0]['model_parameters']['shared_high_frequency']
        hf['refined_d2_metric']['candidate']=hf['phase_only_d2_metric']['candidate']+1
        self.assertFalse(single_pair_validation_passed(report,self.c))
        report=deepcopy(result.report)
        report['candidates']['single_pair']['parameters']['regions'][0]['model_parameters']['basis_source']='measured_bn_only'
        self.assertFalse(single_pair_validation_passed(report,self.c))
