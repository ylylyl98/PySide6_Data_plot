import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
from ui_qt.main_window import MainWindow, LoadedState
from tests.test_drr_three_regions import example


class ThreeRegionUiTests(unittest.TestCase):
    def test_unchanged_window_close_does_not_write_region_defaults(self):
        self.assertEqual(self.settings.allKeys(), [])
        self.w.close()
        self.settings.sync()
        self.assertEqual(self.settings.allKeys(), [])

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = QSettings(str(Path(self.tmp.name) / 'settings.ini'), QSettings.IniFormat)
        with patch('ui_qt.main_window.QSettings', return_value=self.settings), patch.object(
                MainWindow, '_restore_last_folder'), patch.object(MainWindow, '_schedule_automatic_update_check'):
            self.w = MainWindow()
        self.addCleanup(self.w.close)
        cube, _ = example()
        self.w.loaded = LoadedState(mode='DRR', folder=self.tmp.name, cube=cube)
        for key, value in dict(xmin=1, xmax=6, ymin=0, ymax=2, vmin=0, vmax=200).items():
            self.w._set_spin_value_silent(self.w.drr_spins[key], value)

    def test_three_regions_auto_middle_and_shared_boundaries(self):
        w = self.w
        w.drr_region_count_combo.setCurrentIndex(2)
        self.assertTrue(w.drr_split_scale_chk.isChecked())
        w._set_spin_value_silent(w.drr_split_spins['x0'], 2.5)
        w._set_spin_value_silent(w.drr_split_spins['x1'], 4.5)
        w._auto_split_vrange('drr', 'middle')
        p = w._split_scale_for_prefix('drr')
        self.assertEqual(p.split_x2, 4.5)
        self.assertLess(p.middle_vmax, 21)
        for side in ('left', 'middle', 'right'):
            w._set_spin_value_silent(w.drr_second_split_spins[side+'_vmin'], -1)
            w._set_spin_value_silent(w.drr_second_split_spins[side+'_vmax'], 1)
        second = w._split_scale_for_prefix('drr_second')
        self.assertEqual((p.split_x, p.split_x2), (second.split_x, second.split_x2))
        self.assertEqual(second.middle_vmin, -1)
        w._on_split_scale_param_changed('drr_second')
        self.assertEqual(w.drr_split_spins['x0'].value(), 2.5)
        self.assertEqual(w.drr_split_spins['x1'].value(), 4.5)
        w.drr_split_fix_checks['middle_vmax'].setChecked(True)
        w._set_spin_value_silent(w.drr_split_spins['middle_vmax'], 50)
        w._auto_split_vrange('drr', 'middle')
        self.assertEqual(w.drr_split_spins['middle_vmax'].value(), 50)

    def test_switch_back_to_one_or_two_and_json_roundtrip(self):
        from ui_qt.drr_regions import save_settings, restore_settings
        w = self.w
        w.drr_region_count_combo.setCurrentIndex(2)
        w._set_spin_value_silent(w.drr_split_spins['x0'], 2.5)
        w._set_spin_value_silent(w.drr_split_spins['x1'], 4.5)
        w.drr_split_fix_checks['x1'].setChecked(True)
        save_settings(w)
        w.drr_region_count_combo.setCurrentIndex(0)
        self.assertIsNone(w._split_scale_for_prefix('drr'))
        restore_settings(w)
        self.assertEqual(w.drr_region_count_combo.currentData(), 3)
        self.assertEqual(w.drr_split_spins['x1'].value(), 4.5)
        self.assertTrue(w.drr_split_fix_checks['x1'].isChecked())
        w.drr_region_count_combo.setCurrentIndex(1)
        self.assertIsNone(w._split_scale_for_prefix('drr').split_x2)

    def test_auto_boundary_respects_other_fixed_boundary(self):
        from ui_qt.drr_regions import initialize_boundaries
        w = self.w
        w.drr_region_count_combo.setCurrentIndex(2)
        w._set_spin_value_silent(w.drr_split_spins['x0'], 4.5)
        w.drr_split_fix_checks['x0'].setChecked(True)
        initialize_boundaries(w, center=True)
        self.assertEqual(w.drr_split_spins['x0'].value(), 4.5)
        self.assertGreater(w.drr_split_spins['x1'].value(), 4.5)
        self.assertLess(w.drr_split_spins['x1'].value(), 6)
