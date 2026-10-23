"""End-to-end region recipes through UI loading, export and restart."""
import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import QSettings

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_sources import DrrBackgroundResolution, DrrSourceCache, discover_drr_sources
from core.drr_analysis_workspace import create_dataset
from core.drr_workspace_session import save_session, load_session
from core.plotting import HeatmapParams
from core.processing import apply_sg_derivative_energy
from ui_qt.common import ExportOptions
from ui_qt.main_window import MainWindow
from ui_qt.controllers_drr import DrrController
from ui_qt.drr_region_backgrounds import edit_regions


class RegionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for filename, values in [('sample.csv', [100, 120]), ('default_back.csv', [50, 50]),
                                 ('region_back.csv', [25, 30])]:
            (self.root / filename).write_text(
                'Vbg,Vtg,' + ','.join(str(v) for v in np.linspace(740, 780, 81)) + '\n'
                + '\n'.join(f'{i},0,' + ','.join([str(v)] * 81) for i, v in enumerate(values)) + '\n')
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.IniFormat)
        with patch.object(MainWindow, '_restore_last_folder', lambda _self: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda _self: None), patch(
                'ui_qt.main_window.QSettings', return_value=settings):
            self.window = MainWindow()
        self.window._suspend_drr_autoplot = True
        self.window.current_folder = str(self.root)
        self.window.drr_available_sources = discover_drr_sources(self.root)
        self.window.drr_selected_files = ['sample.csv']
        self.window.drr_baseline_files_manual = ['default_back.csv']
        self.window.drr_baseline_combo.blockSignals(True)
        self.window.drr_baseline_combo.setCurrentText('External')
        self.window.drr_baseline_combo.blockSignals(False)
        self.window.drr_baseline_combine_combo.blockSignals(True)
        self.window.drr_baseline_combine_combo.setCurrentIndex(0)
        self.window.drr_baseline_combine_combo.blockSignals(False)
        self.window._drr_baseline_user_selected = True
        self.region = DrrBackgroundRegion(1.61, 1.64, ('region_back.csv',), 'first')
        self.window._drr_background_regions = (self.region,)
        self.signal = SimpleNamespace(emit=lambda *_: None)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def options(self):
        with patch.object(self.window, '_launch_load_options') as launch:
            self.window._start_load('DRR')
        self.assertEqual(launch.call_count, 1)
        return launch.call_args.args[0]

    def load(self):
        options = self.options()
        with patch.object(self.window, '_provenance_records_for_load', return_value=()):
            return self.window._load_task_data(options, progress=self.signal, log=self.signal)

    def test_ui_worker_uses_region_ratio_and_tracks_every_background(self):
        options = self.options()
        self.assertEqual(options.baseline_files, ['default_back.csv', 'region_back.csv'])
        self.assertEqual(options.drr_assignments[0].regions, (self.region,))
        loaded = self.load()
        mask = (loaded.cube.energy >= 1.61) & (loaded.cube.energy < 1.64)
        np.testing.assert_allclose(loaded.cube.Z[:, mask], [[3.] * mask.sum(), [3.8] * mask.sum()])
        np.testing.assert_allclose(loaded.cube.Z[:, ~mask], [[1.] * (~mask).sum(), [1.4] * (~mask).sum()])
        self.assertEqual(loaded.cube.drr_background_boundaries, (1.61, 1.64))
        self.window.loaded = loaded
        with patch.object(self.window, '_start_load') as reload:
            self.window._ensure_loaded_matches_drr_params()
        reload.assert_not_called()

    def test_missing_region_file_prevents_worker_launch(self):
        (self.root / 'region_back.csv').unlink()
        with patch.object(self.window, '_launch_load_options') as launch, patch.object(
                self.window, '_invalidate_drr_for_background_selection') as invalid:
            self.window._start_load('DRR')
        launch.assert_not_called()
        self.assertIn('region_back.csv', invalid.call_args.args[0])

    def test_restore_keeps_default_separate_from_override(self):
        loaded = self.load()
        self.window._drr_background_regions = ()
        self.window.drr_baseline_files_manual = []
        self.window.drr_controller._restore_saved_drr_recipe(
            DrrBackgroundResolution(assignments=loaded.drr_assignments, numerical_path='common'))
        self.assertEqual(self.window._drr_background_regions, (self.region,))
        self.assertEqual(self.window.drr_baseline_files_manual, ['default_back.csv'])
        self.assertIn('1 X range override', self.window.drr_baseline_summary.text())
        self.assertIn('region_back.csv', self.window.drr_baseline_summary.toolTip())

    def test_cancel_editor_preserves_recipe(self):
        self.options()
        before = self.window._drr_assignments
        with patch('ui_qt.drr_background_regions_dialog.DrrBackgroundRegionsDialog') as factory, patch.object(
                self.window, '_start_load') as load:
            factory.return_value.exec.return_value = QDialog.Rejected
            edit_regions(self.window)
        load.assert_not_called()
        self.assertEqual(self.window._drr_assignments, before)
        self.assertEqual(self.window._drr_background_regions, (self.region,))

    def test_catalog_refresh_keeps_region_files_out_of_default_selection(self):
        loaded = self.load()
        result = DrrBackgroundResolution(assignments=loaded.drr_assignments, numerical_path='common')
        self.window._drr_assignments_automatic = True
        self.window._drr_refresh_generation = 1
        cache = DrrSourceCache(load_on_init=False)
        with patch('ui_qt.main_window.resolve_drr_background_assignments', return_value=result), patch.object(
                DrrController, '_refresh_auto_external', return_value=False), patch.object(self.window, '_start_load'):
            self.window._on_drr_catalog_refresh_result(
                (str(self.root), self.window.drr_available_sources, cache), 1, True, set())
        self.assertEqual(self.window.drr_baseline_files_manual, ['default_back.csv'])
        self.assertEqual(self.window._drr_background_regions, (self.region,))

    def test_apply_editor_changes_recipe_and_starts_load(self):
        changed = replace(self.region, xmin_ev=1.62, baseline_which='last')
        with patch('ui_qt.drr_background_regions_dialog.DrrBackgroundRegionsDialog') as factory, patch.object(
                self.window, '_start_load') as load:
            factory.return_value.exec.return_value = QDialog.Accepted
            factory.return_value.regions.return_value = (changed,)
            edit_regions(self.window)
        load.assert_called_once_with('DRR')
        self.assertEqual(self.window._drr_assignments[0].regions, (changed,))
        self.assertFalse(self.window._drr_assignments_automatic)
        self.assertEqual(self.window.drr_baseline_files_manual, ['default_back.csv'])

    def test_editor_callbacks_select_files_and_build_actual_preview(self):
        with patch('ui_qt.drr_background_regions_dialog.DrrBackgroundRegionsDialog') as factory:
            factory.return_value.exec.return_value = QDialog.Rejected
            edit_regions(self.window)
        callbacks = factory.call_args.kwargs
        with patch.object(DrrController, '_open_drr_source_dialog', return_value=['region_back.csv']) as choose:
            self.assertEqual(callbacks['select_files']([]), ['region_back.csv'])
        self.assertTrue(choose.call_args.kwargs['baseline_mode'])
        energy, background = callbacks['preview']((self.region,))
        mask = (energy >= 1.61) & (energy < 1.64)
        np.testing.assert_allclose(background[mask], 25.)
        np.testing.assert_allclose(background[~mask], 50.)

    def test_measurement_switch_obeys_keep_background(self):
        self.window.drr_pin_baseline_chk.blockSignals(True)
        self.window.drr_pin_baseline_chk.setChecked(True)
        self.window.drr_pin_baseline_chk.blockSignals(False)
        with patch.object(DrrController, '_open_drr_source_dialog', return_value=['second.csv']), patch.object(
                self.window, '_start_load'):
            self.window.drr_controller._edit_drr_measurements()
        self.assertEqual(self.window._drr_background_regions, (self.region,))
        self.window.drr_pin_baseline_chk.setChecked(False)
        with patch.object(DrrController, '_open_drr_source_dialog', return_value=['third.csv']), patch.object(
                DrrController, '_start_auto_external'):
            self.window.drr_controller._edit_drr_measurements()
        self.assertEqual(self.window._drr_background_regions, ())

    def test_default_frame_edit_keeps_region_and_reloads(self):
        self.window.loaded = self.load()
        self.window.drr_baseline_combine_combo.setCurrentIndex(1)
        self.assertEqual(self.window._drr_background_regions, (self.region,))
        with patch.object(self.window, '_start_load') as load:
            self.window._ensure_loaded_matches_drr_params()
        load.assert_called_once_with('DRR')

    def test_workspace_roundtrip_preserves_derivative_boundaries(self):
        loaded = self.load()
        dataset = create_dataset(loaded.cube, 'sample', {'assignments': [a.to_dict() for a in loaded.drr_assignments]})
        path = self.root / 'session.npz'
        save_session(path, [dataset])
        datasets, _, _ = load_session(path)
        restored = datasets[0]
        self.assertEqual(restored.key, dataset.key)
        self.assertEqual(tuple(restored.cube.drr_background_boundaries), (1.61, 1.64))
        original, _ = apply_sg_derivative_energy(dataset.cube, derivative=2, window_length=7)
        reopened, _ = apply_sg_derivative_energy(restored.cube, derivative=2, window_length=7)
        np.testing.assert_allclose(original.Z, reopened.Z, equal_nan=True)

    def test_export_metadata_and_dat_use_exact_region_recipe(self):
        loaded = self.load()
        raw = loaded.cube
        second, _ = apply_sg_derivative_energy(raw, derivative=2, window_length=7)
        params = HeatmapParams('sample', 'Energy', 'Gate', 'DR/R', 0, 4,
                              (float(raw.energy.min()), float(raw.energy.max())), (0, 1))
        options = ExportOptions(mode='DRR', params=params, drr_cube=raw, drr_raw_cube=raw,
                                drr_second_cube=second, drr_raw_params=params, drr_second_params=params)
        with patch('core.export._save_heatmap_png', side_effect=lambda path, *_a, **_k: Path(path).write_bytes(b'png')):
            self.window._export_task(loaded, options, progress=self.signal, log=self.signal)
        metadata = list((self.root / 'Processed Data' / 'DRR').glob('*.metadata.json'))
        self.assertTrue(metadata)
        for path in metadata:
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['processing']['drr_background_assignments'], [a.to_dict() for a in loaded.drr_assignments])
            self.assertIn('region_back.csv', json.dumps(saved['inputs']))
        from core.drr_auto_external import resolve_auto_external
        restored = resolve_auto_external(self.root, discover_drr_sources(self.root), ['sample.csv'])
        self.assertTrue(restored.resolved, restored.reason)
        self.assertEqual(restored.assignments[0].regions, (self.region,))
        (self.root / 'region_back.csv').unlink()
        missing = resolve_auto_external(self.root, discover_drr_sources(self.root), ['sample.csv'])
        self.assertFalse(missing.resolved, 'A missing region background must not become a new default recommendation.')
        self.assertIn('missing', missing.reason.lower())


if __name__ == '__main__':
    unittest.main()
