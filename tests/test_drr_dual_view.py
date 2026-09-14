from __future__ import annotations

import unittest
import os
import shutil
import tempfile
from unittest.mock import patch

import numpy as np

from core.loader import DataCube
from ui_qt.controllers_drr import DrrController


class _Value:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _Combo:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class _Owner:
    def __init__(self, cube):
        self.loaded = type("Loaded", (), {"mode": "DRR", "cube": cube})()
        self.drr_derivative_combo = _Combo("d2E")
        self.drr_sg_window_spin = _Value(9)
        self.drr_sg_poly_spin = _Value(2)
        self._drr_derivative_cache = {}
        self.statuses = []

    def _status(self, message):
        self.statuses.append(message)


class DrrDualViewNumericsTests(unittest.TestCase):
    def test_explicit_products_keep_raw_data_and_reuse_derivative_cache(self):
        energy = np.linspace(-2.0, 2.0, 41)
        cube = DataCube(
            energy=energy,
            gate=np.array([0.0]),
            Z=np.array([energy ** 2]),
            gate_label="Gate",
            title="quadratic",
            cbar_label="DR/R",
            gate_unit="V",
            y_axis_semantic="gate_voltage",
        )
        owner = _Owner(cube)
        controller = DrrController(owner)

        raw, raw_derivative, raw_window, raw_poly = controller._drr_cube_with_metadata(derivative=None)
        second, second_derivative, second_window, second_poly = controller._drr_cube_with_metadata(2)
        second_again, *_ = controller._drr_cube_with_metadata(2)

        self.assertIs(raw, cube)
        self.assertIsNone(raw_derivative)
        self.assertEqual((raw_window, raw_poly), (9, 2))
        self.assertEqual(second_derivative, 2)
        self.assertEqual((second_window, second_poly), (9, 2))
        self.assertIs(second_again, second)
        np.testing.assert_allclose(second.Z, 2.0, atol=1e-12)
        np.testing.assert_allclose(raw.Z, cube.Z)
        self.assertEqual(second.gate_unit, "V")
        self.assertEqual(second.y_axis_semantic, "gate_voltage")

        owner.drr_sg_window_spin._value = 11
        changed_sg, *_ = controller._drr_cube_with_metadata(2)
        self.assertIsNot(changed_sg, second)
        replacement = DataCube(
            energy=energy.copy(), gate=np.array([0.0]), Z=np.array([energy ** 2]),
            gate_label="Gate", title="quadratic", cbar_label="DR/R",
        )
        owner.loaded.cube = replacement
        changed_source, *_ = controller._drr_cube_with_metadata(2)
        self.assertIsNot(changed_source, changed_sg)

    def test_selected_derivative_is_used_when_override_is_omitted(self):
        energy = np.linspace(-1.0, 1.0, 21)
        cube = DataCube(energy, np.array([0.0]), np.array([energy ** 2]), "Gate", "q", "DR/R")
        owner = _Owner(cube)
        owner.drr_derivative_combo = _Combo("dE")
        controller = DrrController(owner)

        _result, derivative, _window, _poly = controller._drr_cube_with_metadata()

        self.assertEqual(derivative, 1)


class DrrDualViewWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QSettings
        cls.app = QApplication.instance() or QApplication([])
        cls._settings_tmp = tempfile.mkdtemp(prefix="drr-dual-settings-")
        QSettings.setDefaultFormat(QSettings.IniFormat)
        QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, cls._settings_tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._settings_tmp, ignore_errors=True)

    def setUp(self):
        from ui_qt.main_window import MainWindow
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            self.window = MainWindow()
        self.cube = DataCube(
            energy=np.linspace(-2.0, 2.0, 21), gate=np.array([-1.0, 0.0, 1.0]),
            Z=np.ones((3, 21)), gate_label="Gate", title="map", cbar_label="DR/R",
        )
        from ui_qt.common import LoadedState
        self.window.loaded = LoadedState(
            mode="DRR", folder="", primary_file="map.xlsx", selected_files=["map.xlsx"],
            cube=self.cube, drr_mode_label="DR/R Map",
        )
        self.window.drr_selected_files = ["map.xlsx"]

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_default_tab_is_raw_and_side_by_side_shares_products(self):
        self.window._plot_mode("DRR")
        self.assertEqual(self.window._drr_plot_view, "raw")
        self.assertEqual(set(self.window._drr_plot_cubes), {"raw"})
        self.assertTrue(self.window.drr_view_raw_btn.isChecked())
        self.assertFalse(self.window.drr_view_second_btn.isChecked())

        self.window._on_drr_plot_view_changed("second")
        self.assertEqual(set(self.window._drr_plot_cubes), {"second"})
        self.assertTrue(self.window.drr_view_second_btn.isChecked())
        self.window.drr_view_side_btn.setChecked(True)
        self.assertEqual(set(self.window._drr_plot_cubes), {"raw", "second"})
        self.assertIs(self.window._drr_heatmap_ax, self.window._drr_heatmap_axes["second"])
        self.window.drr_second_vmin_spin.setValue(-4.0)
        self.window.drr_second_vmax_spin.setValue(4.0)
        self.assertEqual(self.window._make_drr_params(self.window._drr_plot_cubes["second"], 2).vmin, -4.0)
        self.assertEqual(self.window.drr_second_cmap.currentData(), "dptk_rdbu_r_p0p60")

    def test_second_product_uses_independent_split_color_scale(self):
        self.window._plot_mode("DRR")
        self.window.drr_spins["xmin"].setValue(-2.0)
        self.window.drr_spins["xmax"].setValue(2.0)
        self.window.drr_split_scale_chk.setChecked(True)
        self.window.drr_split_spins["x0"].setValue(0.0)
        self.window.drr_second_split_spins["left_vmin"].setValue(-3.0)
        self.window.drr_second_split_spins["left_vmax"].setValue(-1.0)
        self.window.drr_second_split_spins["right_vmin"].setValue(1.0)
        self.window.drr_second_split_spins["right_vmax"].setValue(3.0)

        second = self.window._drr_plot_cubes["raw"]
        params = self.window._make_drr_params(second, 2)

        self.assertIsNotNone(params.split_scale)
        self.assertEqual(params.split_scale.split_x, 0.0)
        self.assertEqual(params.split_scale.left_vmin, -3.0)
        self.assertEqual(params.split_scale.right_vmax, 3.0)

if __name__ == "__main__":
    unittest.main()
