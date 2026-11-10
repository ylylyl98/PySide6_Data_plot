"""Real per-file calibration, transactional application and mean-ratio regression."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from tests import test_drr_bn_single_lifecycle as single_fixture
from ui_qt.quick.drr_bn_batch_workflow import compare_bn_batch, verify_batch
from ui_qt.quick.drr_background_workflow import (BackgroundDraft, prepare_application,
    load_with_recipe, commit_application)
from core.drr_bn_store import DrrBnStore


class BnBatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = single_fixture.SingleBnLifecycleTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.second = f.measurement.replace('_PIXIS.csv', '_PIXIS_rep02.csv')
        e = f.energy; x = np.arange(len(e))
        b = np.exp(8+.025*np.sin(.19*x)+.008*np.sin(.33*x))
        fine = np.log(b)-savgol_filter(np.log(b),101,3)
        target = b*np.exp(CubicSpline(x,fine)(x-.55)-fine)
        g = np.linspace(-1,1,9)
        z = target*(1.4+.025*g[:,None])*(1+(.08+.015*g[:,None])*np.exp(-((e-1.72)/.012)**2))
        f.write(self.second,z,g)
        self.context = replace(f.context,measurements=(f.measurement,self.second))

    def test_partial_gate_coverage_does_not_extrapolate_or_drop_whole_file(self):
        from core.loader import DataCube
        from ui_qt.quick.drr_bn_batch_workflow import _average
        e=np.linspace(1.4,1.8,5)
        first=DataCube(e,np.array([0.,1.,2.]),np.ones((3,5)),'BG','first','DR/R')
        second=DataCube(e,np.array([1.,2.]),np.full((2,5),3.),'BG','second','DR/R')
        mean,count=_average([first,second])
        np.testing.assert_allclose(mean.Z,np.array([[1.]*5,[2.]*5,[2.]*5]))
        np.testing.assert_array_equal(count,np.array([[1]*5,[2]*5,[2]*5]))
        with self.assertRaisesRegex(ValueError,'no finite overlap'):
            _average([first,replace(second,gate=np.array([4.,5.]))])

    def test_real_batch_mean_of_ratios_apply_reload_and_exports(self):
        f = self.fixture
        payload = compare_bn_batch(self.context,(f.bn,),'all',settings={'single_full_range':False})
        versions = verify_batch(self.context,payload)
        self.assertEqual(len(versions),2)
        self.assertTrue(all(v.eligible for v in versions),payload['summary'])
        store=DrrBnStore(f.root)
        saved=[store.load_result(v) for v in versions]
        expected=np.mean([v['mean_raw']/v['candidate_background']-1 for v in saved],axis=0)
        wrong=np.mean([v['mean_raw'] for v in saved],axis=0)/np.mean([v['candidate_background'] for v in saved],axis=0)-1
        self.assertGreater(np.max(np.abs(expected-wrong)),1e-6)
        np.testing.assert_allclose(payload['candidate_drr'],expected,atol=1e-12)
        from core.processing import apply_sg_derivative_energy
        from core.loader import DataCube
        d2,_=apply_sg_derivative_energy(DataCube(payload['energy'],payload['gates'],expected,'BG','mean','DR/R'),
            derivative=2,window_length=21,polyorder=2)
        np.testing.assert_allclose(payload['candidate_d2'],d2.Z,atol=1e-9)
        self.assertTrue(np.all(payload['contribution_count']==2))
        self.assertIsNone(store.find_applied((f.measurement,)))
        draft=BackgroundDraft(kind='bn',files=(f.bn,),which='all',calibration={'payload':payload,'original':False})
        prepared=prepare_application(self.context,draft)
        trial=load_with_recipe(self.context,prepared.recipe,prepared=prepared).cube
        np.testing.assert_allclose(trial.Z,expected,atol=1e-12)
        self.assertIsNone(store.find_applied((f.measurement,)))
        recipe=commit_application(prepared,current_context=self.context,settings=f.settings)
        loaded=load_with_recipe(self.context,recipe).cube
        np.testing.assert_allclose(loaded.Z,expected,atol=1e-12)
        for name in self.context.measurements:
            self.assertIsNotNone(store.find_applied((name,)))
        from ui_qt.quick.drr_workflow import DrrRequest,load_drr
        request=DrrRequest(tuple(str(f.root/n) for n in self.context.measurements),recipe.files,'external',
            experiment_folder=str(f.root),y_axis='BG',background_recipe=recipe)
        np.testing.assert_allclose(load_drr(request).source.Z,expected,atol=1e-12)
        table=np.loadtxt(payload['exports']['paths']['candidate_drr_dat'],skiprows=1)
        np.testing.assert_allclose(table[:,1:].T,expected,atol=1e-11)
        meta=json.loads(Path(payload['exports']['paths']['metadata_json']).read_text(encoding='utf-8'))
        self.assertEqual(meta['report']['measurement_count'],2)
        cached=compare_bn_batch(self.context,(f.bn,),'all',settings={'single_full_range':False})
        self.assertTrue(cached['exports']['cached'])
        original=prepare_application(self.context,replace(draft,calibration={'payload':payload,'original':True}))
        commit_application(original,current_context=self.context,settings=f.settings)
        self.assertIsNone(store.find_applied((f.measurement,)))
        self.assertIsNone(store.find_applied((self.second,)))
        with self.assertRaisesRegex(ValueError,'selection'):
            verify_batch(replace(self.context,measurements=(self.second,f.measurement)),payload)
        with self.assertRaisesRegex(ValueError,'SG settings'):
            verify_batch(replace(self.context,derivative_window=31),payload)

    def test_failed_file_remains_in_average_and_source_change_rejects_batch(self):
        f=self.fixture
        e=f.energy; x=np.arange(len(e))
        b=np.exp(8+.025*np.sin(.19*x)+.008*np.sin(.33*x))
        f.write(self.second,np.tile(1.7*b,(9,1)),np.linspace(-1,1,9))
        payload=compare_bn_batch(self.context,(f.bn,),'all',settings={'single_full_range':False})
        self.assertEqual(payload['items'][1]['status'],'original_retained')
        store=DrrBnStore(f.root)
        saved=[store.load_result(v) for v in verify_batch(self.context,payload)]
        first=saved[0]['mean_raw']/saved[0]['candidate_background']-1
        second=saved[1]['mean_raw']/saved[1]['original_background']-1
        np.testing.assert_allclose(payload['candidate_drr'],(first+second)/2,atol=1e-12)
        self.assertFalse(np.allclose(payload['candidate_drr'],first))
        draft=BackgroundDraft(kind='bn',files=(f.bn,),calibration={'payload':payload,'original':False})
        prepared=prepare_application(self.context,draft)
        with (f.root/self.second).open('a',encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError,'changed'):
            commit_application(prepared,current_context=self.context,settings=f.settings)
        self.assertIsNone(store.find_applied((f.measurement,)))
        with self.assertRaisesRegex(ValueError,'changed'):
            verify_batch(self.context,payload)


if __name__=='__main__':
    unittest.main()
