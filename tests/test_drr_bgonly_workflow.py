from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui_qt.controllers_drr import DrrController
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class DrrBgOnlyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self) -> None:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_editing_measurement_clears_stale_external_recipe_and_loads_self(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Initial Data" / "BGonly_Rev.csv"
            source.parent.mkdir()
            source.write_text("Vbg,Vtg,700,701\n0,0,10,11\n1,0,12,13\n", encoding="utf-8")
            w = self.window
            w.current_folder = str(root)
            w.drr_selected_files = ["Initial Data/old.csv"]
            w.drr_baseline_files_manual = ["Initial Data/old_bg.csv"]
            w.drr_baseline_combo.setCurrentText("External")
            captured = []
            with patch.object(DrrController, "_open_drr_source_dialog", return_value=["Initial Data/BGonly_Rev.csv"]), \
                 patch.object(w.thread_pool, "start", captured.append):
                w.drr_controller._edit_drr_measurements()
            self.assertEqual(w.drr_baseline_combo.currentText(), "Self (last frame)")
            self.assertEqual(len(captured), 1, w.statusBar().currentMessage())
            self.assertEqual(captured[0].args[0].drr_baseline_text, "Self (last frame)")

    def test_load_clear_reload_rebuilds_drr_axes_for_full_gate_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Initial Data" / "BGonly_Rev.csv"
            source.parent.mkdir()
            energy = [700 + 0.5 * index for index in range(20)]
            rows = [-2.0, 6.0, 18.0, 30.0]
            source.write_text(
                "Vbg,Vtg," + ",".join(str(value) for value in energy) + "\n"
                + "\n".join(
                    f"{gate},0," + ",".join(str(10.0 + gate + index) for index in range(20))
                    for gate in rows
                ) + "\n",
                encoding="utf-8",
            )
            w = self.window
            w.current_folder = str(root)
            w.tabs.setCurrentIndex(next(i for i in range(w.tabs.count()) if w.tabs.tabText(i) == "DRR"))
            self.app.processEvents()
            w.drr_selected_files = ["Initial Data/BGonly_Rev.csv"]
            w.drr_baseline_combo.setCurrentText("Self (last frame)")
            w._drr_baseline_user_selected = True
            for check in w.drr_fix_checks.values():
                check.setChecked(False)
            w._drr_view_limits = None
            for iteration in range(2):
                captured = []
                with patch.object(w.thread_pool, "start", captured.append):
                    w._start_load("DRR")
                self.assertEqual(len(captured), 1, w.statusBar().currentMessage())
                options = captured[0].args[0]
                loaded = w._load_task(
                    options,
                    progress=SimpleNamespace(emit=lambda *_args: None),
                    log=SimpleNamespace(emit=lambda *_args: None),
                )
                w._on_loaded(loaded, request_token=w._active_load_token)
                w._on_load_finished(request_token=w._active_load_token)
                self.assertEqual(w.last_plotted_mode, "DRR")
                self.assertIsNotNone(w._drr_heatmap_ax)
                self.assertTrue(w._drr_heatmap_ax.collections)
                self.assertTrue(np.isfinite(w._last_plot_cube.Z).all())
                ylim = tuple(float(value) for value in w._drr_heatmap_ax.get_ylim())
                self.assertLessEqual(ylim[0], -2.0, f"iteration={iteration}; ylim={ylim}; controls={(w.drr_spins['ymin'].value(), w.drr_spins['ymax'].value())}; cube={w._last_plot_cube.gate}")
                self.assertGreaterEqual(ylim[1], 30.0)
                if iteration == 0:
                    w._clear_loaded_drr_view()

    def test_side_by_side_uses_compact_titles_and_separate_spectrum_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Initial Data" / "very_long_BGonly_Rev_measurement_filename.csv"
            source.parent.mkdir()
            energy = [700 + 0.5 * index for index in range(20)]
            source.write_text(
                "Vbg,Vtg," + ",".join(str(value) for value in energy) + "\n"
                + "\n".join(
                    f"{gate},0," + ",".join(str(10.0 + gate + index) for index in range(20))
                    for gate in (-2.0, 6.0, 18.0, 30.0)
                ) + "\n",
                encoding="utf-8",
            )
            w = self.window
            w.current_folder = str(root)
            w.tabs.setCurrentIndex(next(i for i in range(w.tabs.count()) if w.tabs.tabText(i) == "DRR"))
            self.app.processEvents()
            w.drr_selected_files = ["Initial Data/very_long_BGonly_Rev_measurement_filename.csv"]
            w.drr_baseline_combo.setCurrentText("Self (last frame)")
            w._drr_baseline_user_selected = True
            w._drr_side_by_side = True
            for check in w.drr_fix_checks.values():
                check.setChecked(False)
            captured = []
            with patch.object(w.thread_pool, "start", captured.append):
                w._start_load("DRR")
            options = captured[0].args[0]
            loaded = w._load_task(
                options,
                progress=SimpleNamespace(emit=lambda *_args: None),
                log=SimpleNamespace(emit=lambda *_args: None),
            )
            w._on_loaded(loaded, request_token=w._active_load_token)
            w._on_load_finished(request_token=w._active_load_token)
            w.figure.canvas.draw()
            raw_map = w._drr_heatmap_axes["raw"]
            second_map = w._drr_heatmap_axes["second"]
            raw_spectrum = w._drr_spectrum_axes["raw"]
            second_spectrum = w._drr_spectrum_axes["second"]
            self.assertEqual(raw_map.get_title(), "ΔR/R")
            self.assertEqual(second_map.get_title(), "Second derivative")
            self.assertEqual(raw_map.get_xlabel(), "")
            self.assertEqual(second_map.get_xlabel(), "")
            self.assertIn("Photon Energy", raw_spectrum.get_xlabel())
            self.assertIn("Photon Energy", second_spectrum.get_xlabel())
            renderer = w.figure.canvas.get_renderer()
            self.assertFalse(
                raw_map.title.get_window_extent(renderer).overlaps(
                    raw_spectrum.title.get_window_extent(renderer)
                )
            )
            canvas_width, canvas_height = w.figure.canvas.get_width_height()
            for axis in (raw_map, second_map, raw_spectrum, second_spectrum):
                for label in (axis.title, axis.xaxis.label, axis.yaxis.label):
                    bbox = label.get_window_extent(renderer)
                    self.assertGreaterEqual(bbox.x0, -1.0)
                    self.assertLessEqual(bbox.x1, canvas_width + 1.0)
                    self.assertGreaterEqual(bbox.y0, -1.0)
                    self.assertLessEqual(bbox.y1, canvas_height + 1.0)


if __name__ == "__main__":
    unittest.main()
