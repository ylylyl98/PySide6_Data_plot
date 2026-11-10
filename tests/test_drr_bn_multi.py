import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from core.drr_bn_multi import evaluate_bn_multi, multi_pair_validation_passed
from core.drr_bn_settings import BnCalibrationSettings
from core.load_cancellation import LoadCancelled

class MultiBnTests(unittest.TestCase):
    def fixture(self):
        e=np.linspace(1.4,2.05,700);x=np.arange(len(e));g=np.linspace(-1,1,7)
        ts=np.array([np.exp(8+.02*np.sin(.15*x+p)+.006*np.sin(.29*x+.2*p)) for p in (0.,.45,.9)])
        b=ts.mean(0);z=np.array([ts[2]*(1+.01*t) for t in g]);c=BnCalibrationSettings(single_measurement=True)
        return e,g,ts,b,z,c
    def test_group_mean_and_same_scan_checks(self):
        e,g,ts,b,z,c=self.fixture();r=evaluate_bn_multi(e,z,b,gates=g,templates=ts,settings=c)
        self.assertTrue(r.eligible,r.reasons);self.assertEqual(r.selected_method,'multi_pair')
        self.assertTrue(multi_pair_validation_passed(r.report,c))
        self.assertFalse(r.report['independent_validation']);self.assertEqual(r.report['model']['template_count'],3)
        self.assertTrue(np.isfinite(r.applied_background).all());self.assertTrue(np.all(r.applied_background>0))
        self.assertGreater(r.report['multi_pair_validation']['d2_windows_checked'],0)
    def test_wrong_scope_mean_and_no_texture_reject(self):
        e,g,ts,b,z,c=self.fixture()
        for cfg,templates,base in [(replace(c,single_full_range=False),ts,b),(c,ts[:1],b),
                                   (c,ts,b*1.1),(c,np.stack([b,b]),b)]:
            r=evaluate_bn_multi(e,z,base,gates=g,templates=templates,settings=cfg)
            self.assertFalse(r.eligible)
    def test_same_data_does_not_generate_candidate(self):
        e,g,ts,b,z,c=self.fixture();same=np.stack([b*(1+.01*t) for t in g])
        self.assertFalse(evaluate_bn_multi(e,same,b,gates=g,templates=ts,settings=c).eligible)
    def test_contract_rejects_tampering(self):
        e,g,ts,b,z,c=self.fixture();r=evaluate_bn_multi(e,z,b,gates=g,templates=ts,settings=c)
        self.assertTrue(r.eligible)
        for section,key,value in [('model','template_rank',0),('model','max_log_correction',.5),
                                  ('multi_pair_validation','d2_failed_windows',1),
                                  ('multi_pair_validation','all_gate_drr_check',False)]:
            report=deepcopy(r.report);report[section][key]=value
            self.assertFalse(multi_pair_validation_passed(report,c))
        self.assertFalse(multi_pair_validation_passed(r.report,replace(c,derivative_window=31)))
    def test_cancel_and_reverse_grid(self):
        e,g,ts,b,z,c=self.fixture()
        with self.assertRaises(LoadCancelled):evaluate_bn_multi(e,z,b,gates=g,templates=ts,settings=c,cancellation=lambda:True)
        r=evaluate_bn_multi(e[::-1],z[:,::-1],b[::-1],gates=g,templates=ts[:,::-1],settings=c)
        self.assertTrue(r.eligible,r.reasons);self.assertTrue(multi_pair_validation_passed(r.report,c))
        self.assertEqual(r.applied_background.shape,b.shape)

    def test_configured_template_settings_and_exhausted_iteration_consistent(self):
        e,g,ts,b,z,c=self.fixture(); custom=replace(c,template_window=81,template_polyorder=2)
        r=evaluate_bn_multi(e,z,b,gates=g,templates=ts,settings=custom)
        self.assertEqual(r.report['model']['template_window'],81)
        self.assertEqual(r.report['model']['template_polyorder'],2)
        self.assertTrue(multi_pair_validation_passed(r.report,custom))
        fake=[{'region':[float(e[0]),float(e[-1])],'passed':False,'gate_count':len(g)}]
        with patch('core.drr_bn_multi._full_axis_drr_check',return_value=fake):
            r=evaluate_bn_multi(e,z,b,gates=g,templates=ts,settings=c)
        self.assertFalse(r.eligible)
        self.assertIn('multi_pair',r.candidate_backgrounds)
        actual=float(np.max(np.abs(np.log(r.candidate_backgrounds['multi_pair']/b))))
        self.assertAlmostEqual(actual,r.report['model']['max_log_correction'],places=12)
