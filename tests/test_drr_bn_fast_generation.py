"""Fast BN generation retains immutable versions and explicit apply behavior."""
import unittest
from unittest.mock import patch

import numpy as np

import tests.test_drr_bn_trend_lifecycle as trend_fixture
from ui_qt.quick.drr_background_workflow import (
    BackgroundDraft, compare_bn, prepare_application, load_with_recipe,
)


class FastGenerationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = trend_fixture.TrendLifecycleTests('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_fast_generation_cache_trial_and_default_export(self):
        f = self.fixture
        with patch('core.drr_bn_single_exports.save_single_pair_products') as export:
            payload = compare_bn(f.context, (f.bn,), 'all',
                settings={'single_trend_enabled': True}, export_products=False)
            self.assertFalse(payload['cached'])
            self.assertIsNone(payload['exports'])
            self.assertTrue(payload['eligible'])
            self.assertIn('maps_drr', payload['preview'])
            cached = compare_bn(f.context, (f.bn,), 'all',
                settings={'single_trend_enabled': True}, export_products=False)
            self.assertTrue(cached['cached'])
            self.assertEqual(cached['version_id'], payload['version_id'])
            export.assert_not_called()
        draft = BackgroundDraft(kind='bn', files=(f.bn,), which='all',
                                calibration={'payload': cached, 'original': False})
        prepared = prepare_application(f.context, draft)
        cube = load_with_recipe(f.context, prepared.recipe, prepared=prepared).cube
        np.testing.assert_allclose(cube.Z, cached['mean_raw']/cached['candidate_background']-1, atol=1e-12)
        self.assertIsNone(f.store.selection_state((f.measurement,)))
        with patch('core.drr_bn_single_exports.save_single_pair_products', return_value={'status': 'exported'}) as export:
            default = compare_bn(f.context, (f.bn,), 'all', settings={'single_trend_enabled': True})
            export.assert_called_once()
            self.assertEqual(default['exports']['status'], 'exported')


if __name__ == '__main__':
    unittest.main()
