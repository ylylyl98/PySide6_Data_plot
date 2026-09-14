import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox
from matplotlib.figure import Figure
from ui_qt.main_window import MainWindow


def _result(source, angles):
    n, m = len(angles), 4
    data = SimpleNamespace(source_file=source, wavelength_nm=np.arange(m, dtype=float),
                           spectra=np.ones((n, m)), source_rows=np.arange(n), detected_columns={})
    settings = SimpleNamespace(peak_center_nm=1.5, gate_min_nm=1., gate_max_nm=2., left_min_nm=0., left_max_nm=.5,
                               right_min_nm=2.5, right_max_nm=3.)
    return SimpleNamespace(data=data, settings=settings, measured_angle_deg=np.asarray(angles, float),
                           cleaned_spectra=np.ones((n, m)), cosmic_ray_mask=np.zeros((n, m), bool),
                           baseline=np.zeros((n, m)), corrected=np.ones((n, m)),
                           integrated_area=np.arange(n, dtype=float)+1,
                           area_uncertainty=np.ones(n), included=np.ones(n, bool),
                           quality_flags=np.zeros(n, dtype=bool), cosmic_pixels_removed=np.zeros(n, dtype=int))


class ShgDisplayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_single_final_plot_selects_nearest_angle_and_updates_title(self):
        owner = MainWindow.__new__(MainWindow); owner.figure = Figure()
        owner.shg_angle_cursor_spin = QDoubleSpinBox(); owner.shg_angle_cursor_spin.setValue(14.)
        owner.shg_spectrum_view_combo = QComboBox(); owner.shg_spectrum_view_combo.addItem("Raw + cleaned")
        owner.loaded = SimpleNamespace(shg_fit=None)
        MainWindow._plot_shg_result(owner, _result("single.csv", [0., 15., 30.]))
        self.assertEqual(owner._shg_selected_index, 1)
        self.assertIn("15°", owner._shg_raw_ax.get_title())

    def test_cursor_change_is_view_only_while_pending_payload_is_retained(self):
        owner = SimpleNamespace(_shg_reprocess_pending_payload="pending", _shg_reprocess_workers=[object()])
        from ui_qt.controllers_shg import ShgController
        owner.loaded = SimpleNamespace(mode="SHG Processing")
        owner._schedule_plot_redraw = lambda *_args, **_kwargs: None
        controller = ShgController(owner)
        controller._shg_reprocess_pending_payload = "pending"
        controller._on_shg_angle_cursor_changed()
        self.assertEqual(controller._shg_reprocess_pending_payload, "pending")

    def test_single_enabled_fit_is_rendered_with_latest_nearest_angle(self):
        owner = MainWindow.__new__(MainWindow); owner.figure = Figure()
        owner.shg_angle_cursor_spin = QDoubleSpinBox(); owner.shg_angle_cursor_spin.setValue(29.)
        owner.shg_spectrum_view_combo = QComboBox(); owner.shg_spectrum_view_combo.addItem("Raw + cleaned")
        fit = SimpleNamespace(fit_mask=np.array([True, True, True]), residual=np.zeros(3),
                              i0=1., amplitude=2., x_center_deg=15.)
        owner.loaded = SimpleNamespace(shg_fit=fit)
        result = _result("fit.csv", [0., 15., 30.])
        MainWindow._plot_shg_result(owner, result)
        self.assertEqual(owner._shg_selected_index, 2)
        self.assertIn("30°", owner._shg_raw_ax.get_title())
        self.assertTrue(any("Fit xc" in line.get_label() for line in owner._shg_angle_ax.lines))

    def test_compare_plot_renders_both_latest_results(self):
        owner = MainWindow.__new__(MainWindow); owner.figure = Figure()
        owner.shg_compare_display_combo = QComboBox(); owner.shg_compare_display_combo.addItem("Absolute")
        a, b = _result("A.csv", [0., 15., 30.]), _result("B.csv", [5., 20., 35.])
        owner.loaded = SimpleNamespace(shg_fit=None, shg_fit_b=None, shg_twist=None)
        MainWindow._plot_shg_comparison(owner, a, b)
        self.assertGreaterEqual(len(owner._shg_angle_ax.lines), 2)
        self.assertGreaterEqual(len(owner._shg_angle_ax.collections), 2)
        self.assertIn("measured angle", owner._shg_angle_ax.get_title().lower())

    def test_real_reprocess_publication_redraws_latest_spectrum_without_new_request(self):
        from ui_qt.common import LoadedState
        from unittest.mock import Mock
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        try:
            old = _result("old.csv", [0., 10., 20.])
            latest = _result("latest.csv", [0., 15., 30.])
            latest.data.spectra[:] = np.array([[1., 1., 1., 1.], [2., 3., 4., 5.], [9., 9., 9., 9.]])
            window.loaded = LoadedState(mode="SHG Processing", folder="", selected_files=[],
                                        shg_data=old.data, shg_result=old)
            shg_tab = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "SHG")
            window.tabs.setCurrentIndex(shg_tab)
            controller = window.shg_controller
            controller._shg_reprocess_generation = 4
            controller._shg_reprocess_key = (4, latest.settings, None, False)
            controller._shg_reprocess_source_key = None
            controller._shg_reprocess_processing_key = None
            requests = Mock(wraps=controller._request_shg_reprocess)
            controller._request_shg_reprocess = requests
            window.shg_angle_cursor_spin.setValue(29.)
            controller._on_shg_reprocessed(4, (latest, None, None, None, None))
            for _ in range(30):
                self.app.processEvents()
            self.assertIs(window.loaded.shg_result, latest)
            self.assertEqual(window._shg_selected_index, 2)
            self.assertIn("30°", window._shg_raw_ax.get_title())
            np.testing.assert_allclose(window._shg_raw_ax.lines[0].get_ydata(), [9., 9., 9., 9.])
            requests.assert_not_called()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
