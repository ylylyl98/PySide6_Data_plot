"""DRR sidebar navigation must preserve processing state and product ownership."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolButton

from core.loader import DataCube
from ui_qt.common import LoadedState
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme
from tests.ui_test_helpers import dispose_owned_window


class DrrSidebarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = QSettings(str(Path(self.tmp.name) / "ui.ini"), QSettings.IniFormat)
        with (patch("ui_qt.main_window.QSettings", return_value=self.settings),
              patch.object(MainWindow, "_restore_last_folder", lambda self: None),
              patch.object(MainWindow, "_schedule_automatic_update_check", lambda self: None)):
            self.w = MainWindow()
        self.w.setAttribute(Qt.WA_DontShowOnScreen)
        self.w._suspend_drr_autoplot = True
        self.w.resize(1900, 1000)
        self.w.show()
        self.w.tabs.setCurrentIndex(1)
        self.app.processEvents()

    def tearDown(self):
        dispose_owned_window(self.w)
        self.tmp.cleanup()

    def page(self, index):
        self.w.drr_sidebar_tabs.setCurrentIndex(index)
        self.app.processEvents()

    def test_background_mode_reveals_only_applicable_controls(self):
        self.page(0)
        w = self.w
        for mode, external, fitted in (("Self (last frame)", False, False),
                                       ("External", True, False),
                                       ("Fitted (sample)", False, True),
                                       ("Self (first frame)", False, False)):
            with self.subTest(mode=mode):
                w.drr_baseline_combo.setCurrentText(mode)
                self.app.processEvents()
                self.assertEqual(w.drr_edit_baselines_btn.isVisible(), external)
                self.assertEqual(w.drr_baseline_combine_combo.isVisible(), external)
                self.assertEqual(w.drr_background_fit_btn.isVisible(), fitted)
                self.assertTrue(w.drr_baseline_summary.isVisible())
                self.assertTrue(w.drr_data_controls.isVisible())

    def test_mixed_only_parameters_hide_without_losing_their_values(self):
        self.page(1)
        w = self.w
        self.assertFalse(w.drr_sg_y_window_spin.isVisible())
        w.drr_second_kind_combo.setCurrentIndex(1)
        self.assertTrue(w.drr_sg_y_window_spin.isVisible())
        self.assertTrue(w.drr_mixed_controls_form.labelForField(w.drr_sg_y_window_spin).isVisible())
        w.drr_sg_y_window_spin.setValue(9)
        w.drr_second_kind_combo.setCurrentIndex(0)
        self.assertFalse(w.drr_sg_y_window_spin.isVisible())
        self.assertFalse(w.drr_mixed_controls_form.labelForField(w.drr_sg_y_window_spin).isVisible())
        w.drr_second_kind_combo.setCurrentIndex(1)
        self.assertEqual(w.drr_sg_y_window_spin.value(), 9)
        self.assertFalse(w.drr_sg_y_auto_chk.isChecked())

    def test_page_and_color_navigation_do_not_mutate_parameters_or_request_plots(self):
        w = self.w
        w.drr_spins["vmin"].setValue(-.25)
        w.drr_second_vmin_spin.setValue(-4)
        w.drr_fix_checks["vmin"].setChecked(True)
        w.drr_sg_window_spin.setValue(15)
        from ui_qt.drr_display_memory import _controls, _value
        before = {key: _value(widget) for key, widget in _controls(w).items()}
        with patch.object(w, "_schedule_plot_redraw") as redraw:
            for index in (0, 1, 0, 1, 0):
                self.page(index)
                w.drr_color_tabs.setCurrentIndex(index % 2)
            w.drr_display_tab_scroll.ensureWidgetVisible(w.drr_color_tabs)
            self.app.processEvents()
            QTest.mouseClick(w.drr_color_tabs, Qt.LeftButton,
                             pos=w.drr_color_tabs.tabRect(1).center())
            self.assertTrue(w.drr_second_vmin_spin.isVisible())
            self.assertFalse(w.drr_spins["vmin"].isVisible())
            w.drr_color_tabs.setFocus()
            QTest.keyClick(w.drr_color_tabs, Qt.Key_Left)
            self.assertTrue(w.drr_spins["vmin"].isVisible())
            self.assertFalse(w.drr_second_vmin_spin.isVisible())
            redraw.assert_not_called()
        self.assertEqual(before, {key: _value(widget) for key, widget in _controls(w).items()})
        self.assertEqual(w.drr_sg_window_spin.value(), 15)
        self.assertEqual(w._drr_plot_view, "raw")
        self.assertTrue(w.drr_spins["gate"].isHidden())
        self.assertTrue(w.drr_gate_toolbar_spin.isVisible())

    def test_shared_boundaries_are_editable_from_either_product(self):
        self.page(0)
        w = self.w
        w.drr_spins["xmin"].setValue(1)
        w.drr_spins["xmax"].setValue(4)
        w.drr_region_count_combo.setCurrentIndex(2)
        w.drr_split_spins["x0"].setValue(2)
        w.drr_split_spins["x1"].setValue(3)
        w.drr_split_spins["middle_vmax"].setValue(.7)
        w.drr_second_split_spins["middle_vmax"].setValue(8)
        for target in (0, 1, 0):
            w.drr_color_tabs.setCurrentIndex(target)
            self.app.processEvents()
            self.assertTrue(w.drr_split_spins["x0"].isVisible())
            self.assertTrue(w.drr_split_spins["x1"].isVisible())
            self.assertFalse(w.drr_spins["vmin"].isVisible())
            self.assertFalse(w.drr_second_vmin_spin.isVisible())
        self.assertEqual(w.drr_split_spins["middle_vmax"].value(), .7)
        self.assertEqual(w.drr_second_split_spins["middle_vmax"].value(), 8)
        w.drr_region_count_combo.setCurrentIndex(1)
        self.assertFalse(w.drr_split_spins["x1"].isVisible())
        w.drr_region_count_combo.setCurrentIndex(0)
        self.assertFalse(w.drr_region_boundaries.isVisible())
        self.assertTrue(w.drr_spins["vmin"].isVisible())

    def test_raw_auto_only_updates_raw_colors_even_when_derivative_plot_is_active(self):
        w = self.w
        cube = DataCube(np.linspace(1, 2, 21), np.array([0., 1.]),
                        np.linspace(-.5, .5, 42).reshape(2, 21), "Gate", "Test", "DR/R")
        w.loaded = LoadedState(mode="DRR", folder="", primary_file="synthetic.xlsx",
                              selected_files=["synthetic.xlsx"], cube=cube)
        w._drr_plot_view = "second"
        w.drr_second_vmin_spin.setValue(-4)
        w.drr_second_vmax_spin.setValue(4)
        w._drr_memory_protected = {"second_min", "second_max"}
        w._update_action_states()
        w.drr_color_tabs.setCurrentIndex(0)
        with patch.object(w, "_auto_drr_second_vrange") as second_auto:
            w.drr_auto_v_btn.click()
            second_auto.assert_not_called()
        self.assertLess(w.drr_spins["vmin"].value(), w.drr_spins["vmax"].value())
        self.assertEqual(w.drr_second_vmin_spin.value(), -4)
        self.assertEqual(w.drr_second_vmax_spin.value(), 4)
        self.assertTrue({"second_min", "second_max"} <= w._drr_memory_protected)

    def test_data_selection_stays_visible_while_pages_scroll_and_keyboard_switches_pages(self):
        w = self.w
        self.page(0)
        self.assertEqual([w.drr_sidebar_tabs.tabText(i) for i in range(w.drr_sidebar_tabs.count())],
                         ["Display", "Processing"])
        for button in w.drr_display_tab_scroll.findChildren(QToolButton):
            if button.text() == "More display settings":
                button.setChecked(True)
        self.app.processEvents()
        button = w.drr_edit_measurements_btn
        point = button.mapTo(w.left_panel, button.rect().topLeft())
        scroll = w.drr_display_tab_scroll
        scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
        self.app.processEvents()
        self.assertEqual(point, button.mapTo(w.left_panel, button.rect().topLeft()))
        self.assertEqual(w.drr_tab_scroll.verticalScrollBar().maximum(), 0)
        w.drr_sidebar_tabs.tabBar().setFocus()
        QTest.keyClick(w.drr_sidebar_tabs.tabBar(), Qt.Key_Right)
        self.assertEqual(w.drr_sidebar_tabs.currentIndex(), 1)
        self.assertTrue(button.isVisible())
        self.assertTrue(w.drr_baseline_combo.isVisible())
        self.assertTrue(w.drr_baseline_summary.isVisible())
        self.assertEqual(point, button.mapTo(w.left_panel, button.rect().topLeft()))
        QTest.keyClick(w.drr_sidebar_tabs.tabBar(), Qt.Key_Left)
        self.assertEqual(w.drr_sidebar_tabs.currentIndex(), 0)
        w.resize(1180, 820)
        w.drr_baseline_combo.setCurrentText("External")
        for index in (0, 1):
            self.page(index)
            self.app.processEvents()
            self.assertEqual(w.drr_tab_scroll.verticalScrollBar().maximum(), 0)
            self.assertTrue(w.drr_tab_scroll.viewport().rect().contains(
                button.mapTo(w.drr_tab_scroll.viewport(), button.rect().center())))

    def test_pending_raw_auto_completes_while_the_other_product_is_visible(self):
        from ui_qt.drr_color_limits import _context, roi_request
        w = self.w
        cube = DataCube(np.linspace(1, 2, 21), np.array([0., 1.]),
                        np.ones((2, 21)), "Gate", "Test", "DR/R")
        w.loaded = LoadedState(mode="DRR", folder="", primary_file="synthetic.xlsx",
                              selected_files=["synthetic.xlsx"], cube=cube)
        w._drr_plot_view = "second"
        request = roi_request(w)
        context = _context(w, request)
        w._drr_pending_raw_auto = (cube, context, request)
        with (patch("ui_qt.controllers_drr.DrrController._auto_drr_vrange") as apply_auto,
              patch.object(w, "_schedule_plot_redraw")):
            w._on_drr_display_ready()
            apply_auto.assert_called_once_with(follow_view=False)
            apply_auto.reset_mock()
            w._drr_pending_raw_auto = (cube, context, request)
            w.drr_spins["xmin"].setValue(1.25)
            w._on_drr_display_ready()
            apply_auto.assert_not_called()

    def test_range_editors_fit_at_supported_sidebar_widths(self):
        w = self.w
        self.page(0)
        for width in (320, 380, 560):
            total = sum(w.workspace_splitter.sizes())
            w.workspace_splitter.setSizes([width, total - width])
            self.app.processEvents()
            for minimum, maximum in (("xmin", "xmax"), ("ymin", "ymax"), ("vmin", "vmax")):
                first, second = w.drr_spins[minimum], w.drr_spins[maximum]
                w.drr_display_tab_scroll.ensureWidgetVisible(first)
                self.app.processEvents()
                row = first.parentWidget()
                for widget in (first, second, w.drr_fix_checks[minimum], w.drr_fix_checks[maximum]):
                    self.assertTrue(row.contentsRect().contains(widget.geometry()), (width, minimum))
                    self.assertTrue(widget.accessibleName())
                self.assertLess(first.geometry().right(), second.geometry().left())
                self.assertEqual(first.geometry().top(), second.geometry().top())

    def test_color_arrow_steps_are_independent_for_products_and_regions(self):
        w = self.w
        pairs = [(w.drr_spins['vmin'], w.drr_spins['vmax'], 100., 1.),
                 (w.drr_second_vmin_spin, w.drr_second_vmax_spin, .01, .0001)]
        for prefix, magnitude, step in (('drr', 1., .01), ('drr_second', .001, .00001)):
            spins = getattr(w, prefix + '_split_spins')
            for side in ('left', 'middle', 'right'):
                pairs.append((spins[side + '_vmin'], spins[side + '_vmax'], magnitude, step))
        for lower, upper, magnitude, _ in pairs:
            for spin, value in ((lower, -magnitude), (upper, magnitude)):
                blocked = spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(blocked)
        axes = {key: spin.value() for key, spin in w.drr_spins.items() if key not in ('vmin', 'vmax')}
        for lower, upper, magnitude, step in pairs:
            before = [(lo.value(), hi.value()) for lo, hi, _, _ in pairs if hi is not upper]
            upper.stepUp()
            self.assertAlmostEqual(upper.value(), magnitude + step, places=8)
            self.assertEqual(lower.value(), -magnitude)
            self.assertEqual(before, [(lo.value(), hi.value()) for lo, hi, _, _ in pairs if hi is not upper])
        self.assertEqual(axes, {key: w.drr_spins[key].value() for key in axes})
        self.assertEqual(w.drr_spins['xmin'].singleStep(), .1)
        self.assertEqual(w.drr_split_spins['x0'].singleStep(), .01)


if __name__ == "__main__":
    unittest.main()
