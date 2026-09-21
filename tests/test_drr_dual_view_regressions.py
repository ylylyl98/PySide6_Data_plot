"""Reviewer reproductions for the DRR dual-view interaction contract."""

from __future__ import annotations

import os
import tempfile
import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
import numpy as np

from core.loader import DataCube
from core.processing import apply_sg_derivative_energy, compute_auto_limits
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow


class DrrDualViewRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.settings_dir = tempfile.TemporaryDirectory(prefix="drr-review-settings-")
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
        self.errors = Mock()
        self.window._show_error = self.errors
        index = next(
            i for i in range(self.window.tabs.count())
            if self.window.tabs.tabText(i) == "DRR"
        )
        self.window.tabs.setCurrentIndex(index)
        energy = np.linspace(-2.0, 2.0, 41)
        gate = np.array([-1.0, 0.0, 1.0])
        values = np.array([(2 + g) * energy ** 2 + g * energy ** 3 for g in gate])
        self.cube = DataCube(energy, gate, values, "Gate", "Synthetic DRR", "DR/R")
        self.window.loaded = LoadedState(
            mode="DRR", folder="", primary_file="map.xlsx", selected_files=["map.xlsx"],
            cube=self.cube, drr_mode_label="DR/R Map",
        )
        self.window.drr_selected_files = ["map.xlsx"]
        for key, value in (
            ("xmin", -2), ("xmax", 2), ("ymin", -1), ("ymax", 1),
            ("gate", 0), ("vmin", -5), ("vmax", 20),
        ):
            self._set_silently(self.window.drr_spins[key], value)
        self._set_silently(self.window.drr_sg_window_spin, 9)
        self._set_silently(self.window.drr_sg_poly_spin, 2)
        self.window._drr_view_limits = None
        self.window._plot_mode("DRR")
        self.errors.assert_not_called()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    @staticmethod
    def _set_silently(spin, value):
        blocked = spin.blockSignals(True)
        try:
            spin.setValue(value)
        finally:
            spin.blockSignals(blocked)

    def _assert_visible_limits(self, xlim, ylim):
        self._wait_derivative()
        for axis in self.window._drr_heatmap_axes.values():
            np.testing.assert_allclose(axis.get_xlim(), xlim)
            np.testing.assert_allclose(axis.get_ylim(), ylim)
        for axis in self.window._drr_spectrum_axes.values():
            np.testing.assert_allclose(axis.get_xlim(), xlim)

    def _wait_derivative(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.window.thread_pool.waitForDone(100)
            self.app.processEvents()
            self.window._run_scheduled_plot_redraw('DRR')
            compute = getattr(self.window, '_drr_display_compute', None)
            if compute is None or compute.worker is None:
                self.app.processEvents()
                return
            time.sleep(.005)
        self.fail('Derivative did not finish')

    def test_layout_and_product_switches_retain_shared_zoom(self):
        w = self.window
        w.drr_view_side_btn.setChecked(True)
        xlim, ylim = (-0.7, 0.9), (-0.4, 0.6)
        w._drr_heatmap_axes["raw"].set_xlim(xlim)
        w._drr_heatmap_axes["raw"].set_ylim(ylim)
        self._assert_visible_limits(xlim, ylim)
        w.drr_view_side_btn.setChecked(False)
        self._assert_visible_limits(xlim, ylim)
        w._on_drr_plot_view_changed("second")
        self._assert_visible_limits(xlim, ylim)
        w._on_drr_plot_view_changed("raw")
        self._assert_visible_limits(xlim, ylim)
        w.drr_view_side_btn.setChecked(True)
        self._assert_visible_limits(xlim, ylim)
        self.errors.assert_not_called()

    def test_manual_xy_overrides_prior_zoom_for_both_products(self):
        w = self.window
        w.drr_view_side_btn.setChecked(True)
        w._drr_heatmap_axes["raw"].set_xlim(-0.7, 0.9)
        w._drr_heatmap_axes["raw"].set_ylim(-0.4, 0.6)
        for key, value in (("xmin", -0.5), ("xmax", 0.5), ("ymin", -0.2), ("ymax", 0.4)):
            w.drr_spins[key].setValue(value)
        w._run_scheduled_plot_redraw("DRR")
        self._assert_visible_limits((-0.5, 0.5), (-0.2, 0.4))
        w.drr_view_side_btn.setChecked(False)
        w._on_drr_plot_view_changed("second")
        self._assert_visible_limits((-0.5, 0.5), (-0.2, 0.4))
        self.errors.assert_not_called()

    def test_save_captures_current_pan_zoom_without_a_view_switch(self):
        w = self.window
        xlim, ylim = (-0.7, 0.9), (-0.4, 0.6)
        w._drr_heatmap_ax.set_xlim(xlim)
        w._drr_heatmap_ax.set_ylim(ylim)
        original_pool = w.thread_pool
        jobs = []
        try:
            w.thread_pool = SimpleNamespace(start=jobs.append)
            w._start_export("DRR")
            self.assertEqual(len(jobs), 1)
            options = jobs[0].args[1]
            for params in (options.drr_raw_params, options.drr_second_params):
                np.testing.assert_allclose(params.xlim, xlim)
                np.testing.assert_allclose(params.ylim, ylim)
            self.errors.assert_not_called()
        finally:
            w.thread_pool = original_pool
            w._export_in_progress = False

    def test_clear_loaded_source_clears_canvas_without_name_error(self):
        self.window._clear_loaded_drr_view()
        self.assertIsNone(self.window.loaded)
        self.assertIsNone(self.window.last_plotted_mode)
        self.assertEqual(self.window.figure.axes, [])
        self.errors.assert_not_called()

    def test_raw_split_scale_does_not_override_second_color_limits(self):
        w = self.window
        w.drr_split_scale_chk.setChecked(True)
        self._wait_derivative()
        second, *_ = w.drr_controller._drr_cube_with_metadata(2)
        w.drr_second_vmin_spin.setValue(-3)
        w.drr_second_vmax_spin.setValue(3)
        raw_params = w._make_drr_params(self.cube, None)
        second_params = w._make_drr_params(second, 2)
        self.assertIsNotNone(raw_params.split_scale)
        # Current UI shares the split boundary; each product retains its own limits.
        self.assertIsNotNone(second_params.split_scale)
        self.assertEqual(second_params.split_scale.split_x, raw_params.split_scale.split_x)
        self.assertNotEqual(second_params.split_scale.left_vmax, raw_params.split_scale.left_vmax)
        self.assertEqual((second_params.vmin, second_params.vmax), (-3, 3))
        self.errors.assert_not_called()

    def test_auto_v_restores_auto_for_active_second_product(self):
        w = self.window
        w._on_drr_plot_view_changed("second")
        w.drr_second_vmin_spin.setValue(-3)
        w.drr_second_vmax_spin.setValue(3)
        self.assertFalse(w.drr_second_auto_scale)
        w.drr_auto_v_btn.click()
        w._run_scheduled_plot_redraw("DRR")
        self.assertTrue(w.drr_second_auto_scale)
        self._wait_derivative()
        limits = compute_auto_limits(w._drr_plot_cubes["second"], log_scale=False)
        np.testing.assert_allclose(
            (w.drr_second_vmin_spin.value(), w.drr_second_vmax_spin.value()),
            (limits.vmin, limits.vmax), atol=1e-5,
        )
        self.errors.assert_not_called()

    def test_advanced_first_derivative_changes_data_and_label(self):
        w = self.window
        w.drr_derivative_combo.setCurrentText("dE")
        w._run_scheduled_plot_redraw("DRR")
        self._wait_derivative()
        expected, _ = apply_sg_derivative_energy(
            self.cube, derivative=1, window_length=9, polyorder=2,
        )
        np.testing.assert_allclose(w._last_plot_cube.Z, expected.Z)
        label = str(w.loaded.drr_derivative_label)
        self.assertTrue("dE" in label or "first derivative" in label.casefold(), label)
        self.errors.assert_not_called()

    def test_product_switch_invalidates_previous_fit_and_peak_state(self):
        w = self.window
        w._drr_fit_gate = 0.0
        w._drr_fit_x = self.cube.energy.copy()
        w._drr_fit_y = np.full(self.cube.energy.shape, 12345.0)
        w._drr_fit_centers = np.array([0.0])
        w._drr_peak_gate = 0.0
        w._drr_peak_indices = np.array([10])
        w._on_drr_plot_view_changed("second")
        self.assertIsNone(w._drr_fit_y)
        self.assertTrue(w._drr_peak_indices is None or w._drr_peak_indices.size == 0)
        self.errors.assert_not_called()

    def test_pending_auto_does_not_apply_to_replacement_source(self):
        from dataclasses import replace
        w = self.window
        w._drr_pending_second_auto = self.cube
        w.loaded.cube = replace(self.cube)
        with patch.object(w, '_auto_drr_second_vrange') as auto:
            w._on_drr_display_ready()
            auto.assert_not_called()
        self.errors.assert_not_called()


if __name__ == "__main__":
    unittest.main()
