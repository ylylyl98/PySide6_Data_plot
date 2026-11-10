"""One real CSV scan can publish, apply and reload its own BN correction."""
from dataclasses import replace
from pathlib import Path
import csv
import tempfile
import unittest

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from core.drr_bn_service import compare_bn_group
from core.drr_bn_store import DrrBnStore, calibration_usable, joint_validation_passed
from core.drr_sources import portable_source_name
from core.load_cancellation import LoadCancellation
from ui_qt.quick.drr_background_workflow import (
    BackgroundContext, BackgroundDraft, compare_bn, prepare_application,
    load_with_recipe, commit_application,
)
from ui_qt.quick.drr_bn_automatic import check_background
from ui_qt.quick.drr_workflow import DrrRequest, load_drr


class SingleBnLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.measurement = 'YZ387_p3b_9T_1.67KREF_730nmc_0p1sx20_RotIn84deg_TG-1BG=14_PIXIS.csv'
        self.bn = 'YZ387_p1a_9T_1.67KREF_730nmc_0p1sx20_RotIn84deg_BNbackground_PIXIS.csv'
        self.energy = np.linspace(1.4, 2.1, 1001)
        x = np.arange(len(self.energy), dtype=float)
        b = np.exp(8 + .025*np.sin(.19*x) + .008*np.sin(.33*x))
        fine = np.log(b) - savgol_filter(np.log(b), 101, 3)
        shifted = b*np.exp(CubicSpline(x, fine)(x+.5)-fine)
        gates = np.linspace(-1, 1, 9)
        sample = shifted*(1 + .02*gates[:, None])
        sample *= 1 + (.12 + .02*gates[:, None])*np.exp(-((self.energy-1.72)/.012)**2)
        self.write(self.measurement, sample, gates)
        self.write(self.bn, np.stack([b, b, b, b]), np.zeros(4))
        self.context = BackgroundContext(str(self.root), (self.measurement,), y_axis='BG')
        self.store = DrrBnStore(self.root)
        from PySide6.QtCore import QSettings
        self.settings = QSettings(str(self.root/'settings.ini'), QSettings.Format.IniFormat)

    def write(self, name, intensity, gates):
        with (self.root/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Vbg', 'Vtg', *(1240/self.energy)])
            for gate, row in zip(gates, intensity):
                writer.writerow([gate, gate, *row])

    def compare(self):
        payload = compare_bn(self.context, (self.bn,), 'all')
        self.assertTrue(payload['eligible'], payload['summary'])
        self.assertEqual(payload['status'], 'single_pair')
        self.assertTrue(calibration_usable(payload['report'], payload['settings']))
        self.assertFalse(joint_validation_passed(payload['report'], payload['settings']))
        self.assertIn('position', payload['report']['acquisition_condition_check']['differences'])
        return payload

    def test_apply_reload_original_and_group_scope(self):
        payload = self.compare()
        draft = BackgroundDraft(kind='bn', files=(self.bn,), which='all',
            calibration={'payload': payload, 'original': False})
        prepared = prepare_application(self.context, draft)
        cube = load_with_recipe(self.context, prepared.recipe, prepared=prepared).cube
        np.testing.assert_allclose(cube.Z, payload['mean_raw']/payload['candidate_background']-1, atol=1e-12)
        commit_application(prepared, current_context=self.context, settings=self.settings)
        version = self.store.find_applied((self.measurement,))
        self.assertEqual(version.validation_status, 'single_pair')
        name = portable_source_name(self.root, version.csv_path)
        request = DrrRequest((str(self.root/self.measurement),), (name,), 'external', y_axis='BG',
                             experiment_folder=str(self.root), background_recipe=prepared.recipe)
        products = load_drr(request)
        np.testing.assert_allclose(products.source.Z, cube.Z, atol=1e-12)
        outcome = check_background(request, automatic=False, reload=True, cancellation=LoadCancellation())
        self.assertIsNotNone(outcome.products)
        self.assertIn('independent repeats were not used', outcome.message)
        with self.assertRaisesRegex(ValueError, 'exact measurement group'):
            self.store.validate_usage(('another.csv',), name, y_axis='BG')
        with self.assertRaisesRegex(ValueError, 'SG settings'):
            load_drr(replace(request, window=31))
        original = prepare_application(self.context, replace(draft, calibration={'payload': payload, 'original': True}))
        commit_application(original, current_context=self.context, settings=self.settings)
        self.assertIsNone(self.store.find_applied((self.measurement,)))
        self.assertEqual(self.store.original_recipe((self.measurement,))['baseline_files'], (self.bn,))

    def test_exports_roundtrip_and_modified_output_is_not_overwritten(self):
        payload = self.compare()
        from core.drr_bn_single_exports import save_single_pair_products
        from core.drr_bn_joint import production_d2
        from core.drr_bn_settings import BnCalibrationSettings
        paths = payload['exports']['paths']
        energy = np.asarray(payload['energy'])
        expected = payload['mean_raw']/payload['candidate_background']-1
        table = np.loadtxt(paths['calibrated_drr_dat'], delimiter='\t', skiprows=1)
        np.testing.assert_allclose(table[:, 0], energy, rtol=1e-11)
        np.testing.assert_allclose(table[:, 1:].T, expected, rtol=1e-10, atol=1e-11)
        d2, _ = production_d2(energy, expected, BnCalibrationSettings(**payload['settings']))
        table = np.loadtxt(paths['calibrated_d2_dat'], delimiter='\t', skiprows=1)
        np.testing.assert_allclose(table[:, 1:].T, d2, rtol=1e-10, atol=1e-9)
        from PIL import Image
        for key in ('comparison_png', 'diagnostics_png'):
            with Image.open(paths[key]) as im:
                self.assertGreater(im.width, 1000)
                self.assertGreater(im.height, 500)
        from core.drr_bn_store import is_bn_calibration_source
        self.assertTrue(is_bn_calibration_source(self.root, paths['background_csv']))
        version = self.store.get(payload['version_id'])
        self.assertTrue(save_single_pair_products(payload, version)['cached'])
        path = Path(paths['calibrated_drr_dat'])
        path.write_text('user change', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed'):
            save_single_pair_products(payload, version)
        self.assertEqual(path.read_text(encoding='utf-8'), 'user change')

    def test_pair_scope_settings_and_source_freshness_cannot_be_bypassed(self):
        payload = self.compare()
        bad = {**payload, 'settings': {**payload['settings'], 'single_measurement': False}, 'cached': False}
        with self.assertRaisesRegex(ValueError, 'single-measurement settings'):
            self.store.save_result(bad)
        bad = {**payload, 'settings': {**payload['settings'], 'derivative_window': 31}, 'cached': False}
        with self.assertRaisesRegex(ValueError, 'evidence'):
            self.store.save_result(bad)
        version = self.store.get(payload['version_id'])
        with (self.root/self.measurement).open('a', encoding='utf-8') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.store.activate(version)

    def test_repeat_mode_still_rejects_metadata_conflict(self):
        with self.assertRaisesRegex(ValueError, 'condition differs'):
            compare_bn_group(self.root, (self.measurement,), (self.bn,), settings={'single_measurement': False})


if __name__ == '__main__':
    unittest.main()
