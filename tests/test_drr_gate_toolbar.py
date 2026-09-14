"""Regression coverage for the compact DRR gate toolbar controls."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.loader import DataCube
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow


class DrrGateToolbarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.settings_dir = tempfile.TemporaryDirectory(prefix="drr-gate-toolbar-settings-")
        cls.old_settings_format = QSettings.defaultFormat()
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, cls.settings_dir.name)

    @classmethod
    def tearDownClass(cls):
        QSettings.setDefaultFormat(cls.old_settings_format)
        cls.settings_dir.cleanup()

    def setUp(self):
        settings = QSettings("DPTK", "PySide6_Data_Plot")
        settings.clear()
        settings.sync()
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def _load_cube(self):
        cube = DataCube(
            np.linspace(-1.0, 1.0, 9),
            np.array([2.35, 1.1, -0.25, -1.8]),
            np.ones((4, 9)),
            "Gate", "Synthetic", "DR/R",
            gate_unit="V",
        )
        self.window.loaded = LoadedState(
            mode="DRR", folder="", primary_file="map.xlsx", selected_files=["map.xlsx"],
            cube=cube, drr_mode_label="DR/R Map",
        )
        self.window.drr_selected_files = ["map.xlsx"]
        self.window.drr_spins["gate"].setValue(float(cube.gate[2]))
        self.window._plot_mode("DRR")
        return cube

    def test_toolbar_gate_input_has_voltage_suffix_and_no_arrows(self):
        spin = self.window.drr_gate_toolbar_spin
        self.assertEqual(spin.suffix(), " V")
        self.assertEqual(spin.buttonSymbols(), spin.ButtonSymbols.NoButtons)
        self.assertFalse(spin.keyboardTracking())

    def test_prev_next_follow_actual_gate_order_and_sync_sidebar(self):
        cube = self._load_cube()
        self.window.drr_gate_prev_btn.click()
        self.assertEqual(self.window.drr_gate_toolbar_spin.value(), float(cube.gate[1]))
        self.assertEqual(self.window.drr_spins["gate"].value(), float(cube.gate[1]))
        self.window.drr_gate_next_btn.click()
        self.assertEqual(self.window.drr_gate_toolbar_spin.value(), float(cube.gate[2]))
        self.assertEqual(self.window.drr_spins["gate"].value(), float(cube.gate[2]))
        self.window.drr_gate_next_btn.click()
        self.window.drr_gate_next_btn.click()
        self.assertEqual(self.window.drr_gate_toolbar_spin.value(), float(cube.gate[3]))
        self.assertFalse(self.window.drr_gate_next_btn.isEnabled())

    def test_toolbar_and_sidebar_edits_share_requested_gate_and_nearest_linecut(self):
        cube = self._load_cube()
        self.window.drr_gate_toolbar_spin.setValue(2.0)
        self.assertEqual(self.window.drr_spins["gate"].value(), 2.0)
        gate_used, _x, _y = self.window.drr_controller._current_drr_spectrum(cube)
        self.assertEqual(gate_used, float(cube.gate[0]))
        self.window.drr_spins["gate"].setValue(-1.8)
        self.assertEqual(self.window.drr_gate_toolbar_spin.value(), -1.8)
        self.assertEqual(self.window.drr_controller._current_drr_spectrum(cube)[0], -1.8)

    def test_toolbar_controls_disable_without_a_plotted_drr_cube(self):
        self.window.loaded = None
        self.window._last_plot_cube = None
        self.window._update_action_states()
        self.assertFalse(self.window.drr_gate_toolbar_spin.isEnabled())
        self.assertFalse(self.window.drr_gate_prev_btn.isEnabled())
        self.assertFalse(self.window.drr_gate_next_btn.isEnabled())

    def test_gate_step_keeps_existing_axes_and_zoom(self):
        self._load_cube()
        axis = self.window._drr_heatmap_ax
        axis.set_xlim(-0.4, 0.6)
        self.window.drr_gate_next_btn.click()
        self.window._run_scheduled_plot_redraw("DRR")
        self.assertIs(self.window._drr_heatmap_ax, axis)
        np.testing.assert_allclose(axis.get_xlim(), (-0.4, 0.6))


if __name__ == "__main__":
    unittest.main()
