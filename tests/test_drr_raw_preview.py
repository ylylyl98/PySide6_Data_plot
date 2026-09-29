import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from core.drr_sources import DrrMeasurementAssignment


class RawPreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, factor in [('a.csv', 1), ('b.csv', 2), ('bg.csv', 3)]:
            (self.root / name).write_text('Vbg_set,Vtg_set,Vbias_set,620,1000,1240\n'
                f'0,0,0,{10*factor},{20*factor},{30*factor}\n'
                f'1,0,0,{20*factor},{40*factor},{60*factor}\n', encoding='utf-8')
        self.loaded = SimpleNamespace(mode='DRR', folder=str(self.root), selected_files=['a.csv', 'b.csv'],
            primary_file='a.csv', baseline_files=[], drr_mode_label='DR/R Self',
            drr_baseline_text='Self (last frame)', drr_baseline_which='last', y_axis_spec='bg', drr_assignments=())

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('core.drr_raw_preview'), 'Raw inspection needs a read-only data loader')
        from core import drr_raw_preview
        return drr_raw_preview

    def test_single_and_average_preserve_raw_intensity(self):
        api = self.api()
        snapshot = api.snapshot_from_loaded(self.loaded)
        single = api.load_preview(snapshot, 'measurement', 'single', 'a.csv')
        mean = api.load_preview(snapshot, 'measurement', 'average')
        np.testing.assert_allclose(single.cube.Z, [[30,20,10], [60,40,20]])
        np.testing.assert_allclose(mean.cube.Z, [[45,30,15], [90,60,30]])
        self.assertFalse(mean.spectrum_only)
        self.assertEqual(len(list(self.root.iterdir())), 3)

    def test_self_reference_is_mean_of_selected_last_frames(self):
        api = self.api()
        result = api.load_preview(api.snapshot_from_loaded(self.loaded), 'background', 'reference')
        self.assertTrue(result.spectrum_only)
        np.testing.assert_allclose(result.cube.Z[0], [90,60,30])

    def test_backgrounds_follow_frozen_per_measurement_assignments(self):
        api = self.api()
        self.loaded.drr_assignments = (DrrMeasurementAssignment('a.csv', 'External', ('bg.csv',), 'first'),
            DrrMeasurementAssignment('b.csv', 'Self (last frame)'))
        snapshot = api.snapshot_from_loaded(self.loaded)
        self.loaded.selected_files.clear()
        self.assertEqual(snapshot.measurements, ('a.csv','b.csv'))
        self.assertEqual(set(snapshot.backgrounds), {'bg.csv','b.csv'})
        result = api.load_preview(snapshot, 'background', 'reference')
        np.testing.assert_allclose(result.cube.Z[0], [105,70,35])

    def test_precomputed_map_is_not_presented_as_raw(self):
        api = self.api()
        self.loaded.drr_mode_label = 'DR/R Map'
        with self.assertRaisesRegex(ValueError, 'precomputed'):
            api.snapshot_from_loaded(self.loaded)

    def test_shifted_grids_average_only_overlapping_samples(self):
        api = self.api()
        from core.loader import DataCube
        a = DataCube(np.array([1.,2.,3.]), np.array([0.,1.]), np.array([[1.,2.,3.],[4.,5.,6.]]), 'Gate', 'a', 'R')
        b = DataCube(np.array([2.,3.,4.]), np.array([0.,1.]), np.array([[20.,30.,40.],[50.,60.,70.]]), 'Gate', 'b', 'R')
        result = api.average_raw_cubes([a,b])
        np.testing.assert_allclose(result.Z, [[1,11,16.5],[4,27.5,33]])
        np.testing.assert_allclose(a.Z, [[1,2,3],[4,5,6]])

    def test_common_self_reference_selects_frame_after_gate_alignment(self):
        api = self.api()
        (self.root / 'b.csv').write_text('Vbg_set,Vtg_set,Vbias_set,620,1000,1240\n'
            '0,0,0,20,40,60\n2,0,0,60,120,180\n', encoding='utf-8')
        reference = api.load_preview(api.snapshot_from_loaded(self.loaded), 'background', 'reference')
        # First file ends at gate=1. Second file interpolates there to [120,80,40].
        np.testing.assert_allclose(reference.cube.Z[0], [90,60,30])

    def test_common_reference_uses_processing_grid_tolerance(self):
        api = self.api()
        (self.root / 'b.csv').write_text('Vbg_set,Vtg_set,Vbias_set,620,1000,1240\n'
            '0,0,0,20,40,60\n0.9999999,0,0,40,80,120\n', encoding='utf-8')
        reference = api.load_preview(api.snapshot_from_loaded(self.loaded), 'background', 'reference')
        np.testing.assert_allclose(reference.cube.Z[0], [90,60,30])


    def test_dialog_loads_average_and_gate_spectrum_without_changing_owner(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QThreadPool
        from PySide6.QtTest import QTest
        self.assertIsNotNone(importlib.util.find_spec('ui_qt.drr_raw_dialog'), 'Raw view needs an on-demand dialog')
        from ui_qt.drr_raw_dialog import DrrRawDialog
        app = QApplication.instance() or QApplication([])
        pool = QThreadPool()
        dialog = DrrRawDialog(self.api().snapshot_from_loaded(self.loaded), pool=pool, initial_gate=1)
        self.addCleanup(dialog.close)
        dialog.show()
        for _ in range(200):
            app.processEvents()
            if dialog.preview is not None:
                break
            QTest.qWait(10)
        self.assertIsNotNone(dialog.preview, dialog.status.text())
        np.testing.assert_allclose(dialog.preview.cube.Z, [[45,30,15],[90,60,30]])
        np.testing.assert_allclose(dialog.spectrum_line.get_ydata(), [90,60,30])
        dialog.gate_spin.setValue(0)
        np.testing.assert_allclose(dialog.spectrum_line.get_ydata(), [45,30,15])
        self.assertEqual(self.loaded.selected_files, ['a.csv','b.csv'])
        self.assertTrue(pool.waitForDone(5000))

    def test_latest_request_wins_and_close_ignores_late_result(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from threading import Event
        from unittest.mock import patch
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QThreadPool
        from PySide6.QtTest import QTest
        from ui_qt.drr_raw_dialog import DrrRawDialog
        app = QApplication.instance() or QApplication([])
        pool = QThreadPool()
        entered, release = Event(), Event()
        original = self.api().load_preview
        def slow_read(*args):
            entered.set()
            release.wait(3)
            return original(*args)
        dialog = DrrRawDialog(self.api().snapshot_from_loaded(self.loaded), pool=pool)
        self.addCleanup(dialog.close)
        with patch('ui_qt.drr_raw_dialog.load_preview', side_effect=slow_read):
            dialog.show()
            for _ in range(100):
                app.processEvents()
                if entered.is_set(): break
                QTest.qWait(10)
            self.assertTrue(entered.is_set())
            dialog.role_combo.setCurrentIndex(1)
            dialog.scope_combo.setCurrentIndex(2)
            release.set()
            for _ in range(200):
                app.processEvents()
                if dialog.preview is not None: break
                QTest.qWait(10)
            self.assertIsNotNone(dialog.preview)
            self.assertTrue(dialog.preview.spectrum_only)
            entered.clear(); release.clear()
            dialog.request_preview()
            for _ in range(100):
                app.processEvents()
                if entered.is_set(): break
                QTest.qWait(10)
            dialog.close()
            release.set()
            self.assertTrue(pool.waitForDone(5000))
            app.processEvents()
            self.assertIsNone(dialog.preview)

    def test_main_window_entry_is_lazy_and_preserves_loaded_result(self):
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from unittest.mock import patch
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QSettings
        from PySide6.QtTest import QTest
        from ui_qt.main_window import MainWindow
        from ui_qt.common import LoadedState
        app = QApplication.instance() or QApplication([])
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.IniFormat)
        with patch('ui_qt.main_window.QSettings', return_value=settings), patch.object(
                MainWindow, '_restore_last_folder', lambda s: None), patch.object(
                MainWindow, '_schedule_automatic_update_check', lambda s: None):
            window = MainWindow()
        self.addCleanup(window.close)
        window.loaded = LoadedState(**vars(self.loaded))
        original = window.loaded
        self.assertIsNone(getattr(window, '_drr_raw_dialog', None))
        window._open_drr_raw_preview()
        dialog = window._drr_raw_dialog
        for _ in range(200):
            app.processEvents()
            if dialog.preview is not None: break
            QTest.qWait(10)
        self.assertIsNotNone(dialog.preview, dialog.status.text())
        window._open_drr_raw_preview()
        self.assertIs(window._drr_raw_dialog, dialog)
        self.assertIs(window.loaded, original)
        dialog.close()
        self.assertIsNone(window._drr_raw_dialog)
        self.assertTrue(window.thread_pool.waitForDone(5000))
