"""Native BN comparison selects the correct protocol for the selected group."""
import tempfile
import sys
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

import numpy as np

import core.drr_bn_service
import core.drr_bn_store
from ui_qt.quick.drr_background_workflow import BackgroundContext, compare_bn


class QuickBnSingleTest(unittest.TestCase):
    def test_multi_pair_exploratory_preview_uses_selected_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            energy = np.linspace(1.4, 2.0, 31)
            original = np.ones(len(energy))
            candidate = original*1.2
            payload = dict(energy=energy, mean_raw=np.ones((2, len(energy)))*2,
                original_background=original, candidate_background=original,
                candidate_backgrounds={"original": original, "multi_pair": candidate},
                selected_candidate="multi_pair", eligible=False, gates=np.array([-1., 1.]))
            context = BackgroundContext(folder, ("sample.csv",))
            with patch("core.drr_bn_service.compare_bn_group", return_value=payload), \
                 patch("core.drr_bn_store.DrrBnStore") as store, \
                 patch("core.drr_bn_single_exports.save_single_pair_products", return_value={}):
                store.return_value.save_result.return_value = SimpleNamespace(version_id="id", csv_path="result.csv")
                result = compare_bn(context, ("BNback.csv",), "all")
            self.assertTrue(result["preview"]["candidate_available"])
            np.testing.assert_allclose(result["preview"]["maps_drr"][1], np.broadcast_to(2/candidate-1, (2, len(energy))))
            self.assertFalse(result["eligible"])

    def test_measurement_count_selects_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            energy = np.linspace(1.4, 2.0, 31)
            payload = dict(energy=energy, mean_raw=np.ones((1, len(energy))),
                           original_background=np.ones(len(energy)), candidate_background=np.ones(len(energy)), eligible=True)
            for measurements, expected in ((('sample.csv',), True), (('sample.csv', 'repeat.csv'), False)):
                context = BackgroundContext(folder, measurements)
                exports = ModuleType("core.drr_bn_single_exports")
                exports.save_single_pair_products = lambda payload, version: {"paths": {"comparison_png": "plot.png"}, "status": "single_pair"}
                with patch.dict(sys.modules, {"core.drr_bn_single_exports": exports}), \
                     patch('core.drr_bn_service.compare_bn_group', return_value=payload) as compare, \
                     patch('core.drr_bn_store.DrrBnStore') as store:
                    store.return_value.save_result.return_value = SimpleNamespace(version_id='id', csv_path='result.csv')
                    result = compare_bn(context, ('BNback.csv',), 'all', settings={'single_measurement': not expected})
                    self.assertEqual(bool(result['exports']), expected)
                    self.assertGreaterEqual(result['elapsed_seconds'], 0)
                    self.assertEqual(compare.call_args.args[4]['single_measurement'], expected)


if __name__ == '__main__':
    unittest.main()
