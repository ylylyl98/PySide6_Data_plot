"""The linked spectrum stays usable in the supported compact workspace."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.loader import DataCube
from tests.ui_test_helpers import dispose_owned_window
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class InspectionPlotLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")

    def setUp(self):
        self.directory = TemporaryDirectory(prefix="inspection-layout-")
        self.addCleanup(self.directory.cleanup)
        settings = QSettings(str(Path(self.directory.name) / "ui.ini"), QSettings.IniFormat)
        install_theme(self.app, mode="light")
        with (patch("ui_qt.main_window.QSettings", return_value=settings),
              patch("ui_qt.presentation_widget.QSettings", return_value=settings),
              patch.object(MainWindow, "_restore_last_folder", lambda _: None),
              patch.object(MainWindow, "_schedule_automatic_update_check", lambda _: None)):
            self.window = MainWindow(drr_display_backend="widgets")
        self.addCleanup(dispose_owned_window, self.window)
        self.errors = Mock()
        self.window._show_error = self.errors
        self.window._ensure_loaded_matches_ui_params = lambda _mode: True
        energy = np.linspace(1.7, 2.7, 81)
        gate = np.linspace(-1, 1, 9)
        signal = 1 + np.exp(-((energy[None, :] - 2.2 - .02 * gate[:, None]) / .04) ** 2)
        self.cube = DataCube(energy, gate, signal, "Gate", "Inspection sample", "Intensity")

    def assert_inspection_layout(self, mode):
        window = self.window
        window.tabs.setCurrentIndex(next(
            index for index in range(window.tabs.count()) if window.tabs.tabText(index) == mode
        ))
        window.loaded = LoadedState(mode=mode, folder="", primary_file="synthetic.dat", cube=self.cube)
        if mode == "DRR":
            window.drr_selected_files = ["synthetic.dat"]
        spins = window.drr_spins if mode == "DRR" else window.pl_spins
        for key, value in dict(xmin=1.7, xmax=2.7, ymin=-1, ymax=1, gate=0, vmin=1, vmax=2).items():
            window._set_spin_value_silent(spins[key], value)
        for width, height in ((1180, 700), (1320, 900)):
            with self.subTest(mode=mode, size=(width, height)):
                window.resize(width, height)
                window.show()
                self.app.processEvents()
                window._last_plot_params_key = None
                window._plot_mode(mode)
                window.canvas.draw()
                self.errors.assert_not_called()
                renderer = window.canvas.get_renderer()
                heatmap = window._drr_heatmap_ax if mode == "DRR" else window._pl_heatmap_ax
                spectrum = window._drr_spectrum_ax if mode == "DRR" else window._pl_spectrum_ax
                self.assertEqual(heatmap.get_xlabel(), "")
                self.assertFalse(any(label.get_visible() for label in heatmap.get_xticklabels()))
                self.assertIn("Photon Energy", spectrum.get_xlabel())
                self.assertAlmostEqual(
                    heatmap.bbox.height / (heatmap.bbox.height + spectrum.bbox.height), .7,
                    places=3,
                )
                self.assertGreaterEqual(spectrum.bbox.height, 72)
                self.assertLessEqual(spectrum.title.get_window_extent(renderer).y1, spectrum.bbox.y1 + 1)
                self.assertGreater(spectrum.title.get_window_extent(renderer).y0, spectrum.bbox.y0)
                canvas_width, canvas_height = window.canvas.get_width_height()
                for label in (heatmap.title, spectrum.title, spectrum.xaxis.label, spectrum.yaxis.label):
                    bounds = label.get_window_extent(renderer)
                    self.assertGreaterEqual(bounds.x0, -1)
                    self.assertLessEqual(bounds.x1, canvas_width + 1)
                    self.assertGreaterEqual(bounds.y0, -1)
                    self.assertLessEqual(bounds.y1, canvas_height + 1)

    def test_drr_map_and_spectrum_do_not_compete_for_energy_labels(self):
        self.assert_inspection_layout("DRR")

    def test_pl_spectrum_remains_a_primary_inspection_surface(self):
        self.assert_inspection_layout("PL")
        self.assertFalse(hasattr(self.window, "pl_export_pair_chk"))

    def test_collapse_gate_update_and_expansion_keep_axes_zoom_and_data(self):
        window = self.window
        original = self.cube.Z.copy()
        for mode in ("PL", "DRR"):
            with self.subTest(mode=mode):
                self.assert_inspection_layout(mode)
                heatmap = window._drr_heatmap_ax if mode == "DRR" else window._pl_heatmap_ax
                spectrum = window._drr_spectrum_ax if mode == "DRR" else window._pl_spectrum_ax
                axes = tuple(window.figure.axes)
                heatmap.set_xlim(1.9, 2.4)
                heatmap.set_ylim(-.6, .6)
                divider = window._inspection_plot_divider
                divider.set_heatmap_fraction(.60)
                QTest.mouseClick(divider.toggle_button, Qt.LeftButton)
                window.canvas.draw()
                self.assertFalse(spectrum.get_visible())
                self.assertEqual(heatmap.get_xlabel(), "Photon Energy (eV)")
                if mode == "DRR":
                    window._set_spin_value_silent(window.drr_spins["gate"], .5)
                    window.drr_controller._update_drr_spectrum_and_gate_line(self.cube)
                else:
                    window._set_spin_value_silent(window.pl_spins["gate"], .5)
                    window.pl_controller._update_pl_spectrum_and_gate_line(self.cube)
                window.canvas.draw()
                self.assertFalse(spectrum.get_visible())
                QTest.mouseClick(divider.toggle_button, Qt.LeftButton)
                window.canvas.draw()
                self.assertTrue(spectrum.get_visible())
                self.assertAlmostEqual(divider.heatmap_fraction, .60)
                self.assertEqual(tuple(window.figure.axes), axes)
                self.assertEqual(heatmap.get_xlim(), (1.9, 2.4))
                self.assertEqual(heatmap.get_ylim(), (-.6, .6))
                self.assertEqual(spectrum.get_title(), "Spectrum @ 0.5 V")
                np.testing.assert_array_equal(self.cube.Z, original)


if __name__ == "__main__":
    unittest.main()
