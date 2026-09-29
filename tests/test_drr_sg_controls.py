"""SG defaults and manual edits must not change the raw paired product."""
import os
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.loader import DataCube
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow


class DrrSgControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.settings = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.settings.cleanup()

    def setUp(self):
        settings = QSettings(os.path.join(self.settings.name, self._testMethodName + '.ini'),
                             QSettings.IniFormat)
        settings_patch = patch('ui_qt.main_window.QSettings', return_value=settings)
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        with patch.object(MainWindow, "_restore_last_folder", lambda self: None):
            self.w = MainWindow()
        self.w.tabs.setCurrentIndex(next(
            i for i in range(self.w.tabs.count()) if self.w.tabs.tabText(i) == "DRR"))
        self.w._suspend_drr_autoplot = True
        self.w._show_error = self.fail
        self.w.drr_region_count_combo.setCurrentIndex(0)
        for key, value in (("xmin", -2), ("xmax", 2), ("ymin", 0), ("ymax", 1)):
            self.w.drr_spins[key].setValue(value)

    def tearDown(self):
        self.w.thread_pool.waitForDone(5000)
        self.w.close()
        self.w.deleteLater()
        self.app.processEvents()

    def load(self, points):
        x = np.linspace(-2, 2, points)
        z = np.array([x ** 4 + .01 * np.sin(70*x), x ** 4])
        cube = DataCube(x, np.array([0., 1.]), z, "Gate", "SG test", "DR/R")
        self.w.loaded = LoadedState(mode="DRR", folder="", primary_file="map.xlsx",
                                   selected_files=["map.xlsx"], cube=cube)
        self.w.drr_selected_files = ["map.xlsx"]
        return cube

    def render(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.w._plot_mode("DRR")
            self.w.thread_pool.waitForDone(100)
            self.app.processEvents()
            compute = getattr(self.w, "_drr_display_compute", None)
            if compute is None or compute.worker is None:
                self.w._plot_mode("DRR")
                return
        self.fail("Derivative did not finish")

    def test_auto_window_tracks_energy_resolution_and_clamps_short_spectra(self):
        for points, expected in ((512, 11), (1024, 21), (1340, 21), (512, 11), (8, 7)):
            with self.subTest(points=points):
                self.load(points)
                _, _, window, poly = self.w.drr_controller._drr_cube_with_metadata(2)
                self.assertEqual(window, expected)
                self.assertEqual(poly, 2)
                self.assertEqual(self.w.drr_sg_window_spin.value(), expected)

    def test_manual_window_survives_redraw_and_new_source_until_auto_restored(self):
        self.load(512)
        self.w.drr_controller._drr_cube_with_metadata(2)
        self.w.drr_sg_window_spin.setValue(15)
        self.assertFalse(self.w.drr_sg_auto_chk.isChecked())
        self.load(8)
        self.assertEqual(self.w.drr_controller._drr_cube_with_metadata(2)[2], 7)
        self.load(1340)
        self.assertEqual(self.w.drr_controller._drr_cube_with_metadata(2)[2], 15)
        self.w.drr_sg_auto_chk.setChecked(True)
        self.assertEqual(self.w.drr_controller._drr_cube_with_metadata(2)[2], 21)

    def test_none_exposes_sg_controls_and_window_edit_changes_only_second_product(self):
        cube = self.load(512)
        self.w.drr_view_side_btn.setChecked(True)
        self.render()
        self.assertEqual(self.w.drr_derivative_combo.currentText(), "None")
        self.assertFalse(self.w.drr_sg_window_spin.isHidden())
        self.assertFalse(self.w.drr_sg_poly_spin.isHidden())
        original = self.w._drr_plot_cubes["second"].Z.copy()
        self.w.drr_sg_window_spin.setValue(21)
        self.render()
        np.testing.assert_array_equal(self.w._drr_plot_cubes["raw"].Z, cube.Z)
        self.assertFalse(np.allclose(self.w._drr_plot_cubes["second"].Z, original))
        self.w.drr_view_side_btn.setChecked(False)
        self.w.drr_view_side_btn.setChecked(True)
        self.assertEqual(self.w.drr_sg_window_spin.value(), 21)

    def test_advanced_derivative_does_not_replace_left_paired_product(self):
        cube = self.load(512)
        self.w.drr_view_side_btn.setChecked(True)
        for mode in ("None", "dE", "d2E"):
            self.w.drr_derivative_combo.setCurrentText(mode)
            self.render()
            np.testing.assert_array_equal(self.w._drr_plot_cubes["raw"].Z, cube.Z)


if __name__ == "__main__":
    unittest.main()
