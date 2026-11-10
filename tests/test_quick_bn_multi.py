import unittest
from dataclasses import replace
import numpy as np
from tests import test_drr_bn_single_lifecycle as single_fixture
from tests import test_drr_bn_multi as multi_fixture
from ui_qt.quick.drr_background_workflow import (compare_bn, BackgroundDraft,
    prepare_application, load_with_recipe, commit_application)
from core.drr_bn_store import DrrBnStore, calibration_usable


class MultiBnLifecycleTests(unittest.TestCase):
    def test_multiple_references_generate_export_apply_and_reload(self):
        f=single_fixture.SingleBnLifecycleTests();f.setUp();self.addCleanup(f.doCleanups)
        e,g,ts,b,z,c=multi_fixture.MultiBnTests().fixture()
        f.energy=e;f.write(f.measurement,z,g)
        names=tuple(f.bn.replace('_p1a_',f'_p1a{i}_') for i in range(3))
        for name,template in zip(names,ts):
            f.write(name,np.tile(template,(4,1)),np.zeros(4))
        p=compare_bn(f.context,names,'all')
        self.assertTrue(p['eligible'],p['summary'])
        self.assertEqual(p['report']['optimization_policy'],'multi_bn_single_v1')
        self.assertEqual(p['selected_candidate'],'multi_pair')
        self.assertTrue(calibration_usable(p['report'],p['settings']))
        np.testing.assert_allclose(p['original_background'],b)
        exported=np.loadtxt(p['exports']['paths']['calibrated_drr_dat'],skiprows=1)
        expected=z/p['candidate_background']-1
        np.testing.assert_allclose(exported[:,1:].T,expected,atol=1e-10)
        draft=BackgroundDraft(kind='bn',files=names,which='all',calibration={'payload':p,'original':False})
        prepared=prepare_application(f.context,draft)
        np.testing.assert_allclose(load_with_recipe(f.context,prepared.recipe,prepared=prepared).cube.Z,expected,atol=1e-12)
        recipe=commit_application(prepared,current_context=f.context,settings=f.settings)
        np.testing.assert_allclose(load_with_recipe(f.context,recipe).cube.Z,expected,atol=1e-12)
        original=prepare_application(f.context,replace(draft,calibration={'payload':p,'original':True}))
        np.testing.assert_allclose(load_with_recipe(f.context,original.recipe,prepared=original).cube.Z,z/b-1,atol=1e-12)
        self.assertTrue(DrrBnStore(f.root).get(p['version_id']).fresh)


if __name__=='__main__':unittest.main()
