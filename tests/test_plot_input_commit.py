"""Input commits must not repaint plots while a number is still being typed."""
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractSpinBox, QLineEdit, QVBoxLayout, QWidget

from tests.ui_test_helpers import dispose_owned_window
from ui_qt.common import LoadedState, QDoubleSpinBox, QSpinBox
from ui_qt.compact_spinbox import CompactDoubleSpinBox
from ui_qt.main_window import MainWindow


class NumericCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_typing_waits_for_return_or_focus_loss_and_arrows_still_commit(self):
        for spin_type in (QDoubleSpinBox, QSpinBox, CompactDoubleSpinBox):
            for commit_key in (Qt.Key_Return, Qt.Key_Tab):
                with self.subTest(spin=spin_type.__name__, commit=commit_key):
                    host = QWidget()
                    layout = QVBoxLayout(host)
                    spin = spin_type()
                    spin.setRange(-1000, 1000)
                    spin.setValue(1)
                    layout.addWidget(spin)
                    layout.addWidget(QLineEdit())
                    values = []
                    spin.valueChanged.connect(values.append)
                    try:
                        host.show()
                        spin.setFocus()
                        self.app.processEvents()
                        spin.selectAll()
                        QTest.keyClicks(spin, "600")
                        self.assertEqual(values, [])
                        self.assertEqual(spin.value(), 1)
                        QTest.keyClick(spin, commit_key)
                        self.app.processEvents()
                        self.assertEqual(values, [600])
                        spin.stepUp()
                        self.assertEqual(values[-1], 601)
                    finally:
                        dispose_owned_window(host)


class PlotInputCommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.settings_dir = TemporaryDirectory()
        self.addCleanup(self.settings_dir.cleanup)
        settings = QSettings(str(Path(self.settings_dir.name) / "ui.ini"), QSettings.IniFormat)
        for item in (
            patch("ui_qt.main_window.QSettings", lambda *args, **kwargs: settings),
            patch.object(MainWindow, "_restore_last_folder", autospec=True),
            patch.object(MainWindow, "_schedule_automatic_update_check", autospec=True),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.window = MainWindow()
        self.addCleanup(dispose_owned_window, self.window)

    def activate(self, mode):
        w = self.window
        w.tabs.setCurrentIndex(next(i for i in range(w.tabs.count()) if w.tabs.tabText(i) == mode))
        w.show()
        self.app.processEvents()

    def test_all_main_window_numbers_wait_for_commit(self):
        tracking = [spin.accessibleName() or spin.objectName() or type(spin).__name__
                    for spin in self.window.findChildren(QAbstractSpinBox) if spin.keyboardTracking()]
        self.assertEqual(tracking, [])

    def test_drr_range_typing_calls_controller_only_when_committed(self):
        self.activate("DRR")
        spin = self.window.drr_spins["xmin"]
        spin.setFocus()
        self.app.processEvents()
        controller = self.window.drr_controller
        with patch.object(type(controller), "_on_drr_plot_param_changed", autospec=True) as changed:
            spin.selectAll()
            QTest.keyClicks(spin, "1.5")
            changed.assert_not_called()
            QTest.keyClick(spin, Qt.Key_Return)
            changed.assert_called_once_with(controller, spin)

    def test_plot_label_waits_for_commit(self):
        edit = self.window.pl_dat_yaxis_label_edit
        edit.show()
        self.window.show()
        edit.setFocus()
        self.app.processEvents()
        controller = self.window.pl_controller
        with patch.object(type(controller), "_on_pl_plot_param_changed", autospec=True) as changed:
            edit.selectAll()
            QTest.keyClicks(edit, "Voltage")
            changed.assert_not_called()
            QTest.keyClick(edit, Qt.Key_Return)
            changed.assert_called_once_with(controller)

    def test_split_scale_arrow_change_submits_once(self):
        with patch.object(self.window, "_on_split_scale_param_changed") as changed:
            self.window.drr_split_spins["left_vmin"].stepUp()
            changed.assert_called_once_with("drr")

    def test_shg_arrows_submit_without_waiting_for_focus_loss(self):
        controller = self.window.shg_controller
        with patch.object(type(controller), "_on_shg_param_changed", autospec=True) as changed:
            self.window.shg_peak_center_spin.stepUp()
            changed.assert_called_once_with(controller)

    def test_scheduled_redraw_waits_for_pending_numeric_input(self):
        self.activate("PL")
        w = self.window
        w.loaded = LoadedState(mode="PL", folder="")
        spin = w.pl_spins["xmin"]
        spin.setFocus()
        self.app.processEvents()
        with patch.object(w, "_plot_mode") as plot:
            w._schedule_plot_redraw("PL", delay_ms=0)
            spin.selectAll()
            QTest.keyClicks(spin, "1.5")
            QTest.qWait(30)
            plot.assert_not_called()
            self.assertIn("PL", w._plot_redraw_pending)
            QTest.keyClick(spin, Qt.Key_Return)
            QTest.qWait(250)
            plot.assert_called_once_with("PL")

    def test_repeated_numeric_steps_render_only_the_latest_value(self):
        self.activate("PL")
        w = self.window
        w.loaded = LoadedState(mode="PL", folder="")
        spin = w.pl_spins["vmin"]
        spin.setFocus()
        self.app.processEvents()
        rendered = []
        with patch.object(w, "_plot_mode", side_effect=lambda mode: rendered.append(spin.value())):
            for _ in range(3):
                QTest.keyClick(spin, Qt.Key_Up)
                QTest.qWait(110)
            self.assertEqual(rendered, [])
            QTest.qWait(160)
            self.assertEqual(rendered, [spin.value()])

    def test_mcd_window_timer_waits_for_the_next_number_to_commit(self):
        self.activate("MCD")
        w = self.window
        w.loaded = LoadedState(mode="MCD", folder="")
        spin = w.mcd_window_center_spin
        spin.setFocus()
        self.app.processEvents()
        controller = w.mcd_controller
        with patch.object(type(controller), "_refresh_mcd_center_trace", autospec=True,
                          return_value=True) as refresh:
            spin.selectAll()
            QTest.keyClicks(spin, "1.6")
            controller._apply_pending_mcd_center_refresh()
            refresh.assert_not_called()
            QTest.keyClick(spin, Qt.Key_Return)
            QTest.qWait(100)
            refresh.assert_called_once_with(controller)

    def test_mcd_recalculation_waits_for_processing_input_to_commit(self):
        self.activate("MCD")
        w = self.window
        w.loaded = LoadedState(mode="MCD", folder="")
        blocked = w.mcd_files.blockSignals(True)
        w.mcd_files.addItem("source.csv")
        w.mcd_files.item(0).setSelected(True)
        w.mcd_files.blockSignals(blocked)
        spin = w.mcd_gap_spin
        spin.setFocus()
        self.app.processEvents()
        controller = w.mcd_controller
        with patch.object(w, "_start_load") as start, patch.object(
            type(controller), "_mcd_settings_from_ui", autospec=True
        ):
            spin.selectAll()
            QTest.keyClicks(spin, "5")
            controller._apply_pending_mcd_settings()
            start.assert_not_called()
            QTest.keyClick(spin, Qt.Key_Return)
            controller._apply_pending_mcd_settings()
            start.assert_called_once_with("MCD")

    def test_mcd_pending_edit_retry_cannot_be_a_zero_interval_loop(self):
        self.activate("MCD")
        w = self.window
        w.loaded = LoadedState(mode="MCD", folder="")
        spin = w.mcd_gap_spin
        spin.setFocus()
        self.app.processEvents()
        spin.selectAll()
        QTest.keyClicks(spin, "5")
        timer = w.mcd_controller._mcd_auto_apply_timer
        timer.setInterval(0)  # Source refresh paths request immediate updates.
        retries = []
        timer.timeout.connect(lambda: retries.append(True))
        w.mcd_controller._apply_pending_mcd_settings()
        QTest.qWait(30)
        self.assertEqual(retries, [])
        self.assertGreater(timer.interval(), 0)

    def test_repeated_shg_processing_steps_recalculate_only_once(self):
        self.activate("SHG")
        w = self.window
        w.loaded = LoadedState(mode="SHG Processing", folder="")
        spin = w.shg_peak_center_spin
        spin.setFocus()
        self.app.processEvents()
        controller = w.shg_controller
        with patch.object(type(controller), "_start_shg_reprocess", autospec=True) as start:
            for _ in range(3):
                QTest.keyClick(spin, Qt.Key_Up)
                QTest.qWait(110)
            start.assert_not_called()
            QTest.qWait(160)
            start.assert_called_once_with(controller)

    def test_shg_reprocessing_waits_for_the_next_number_to_commit(self):
        self.activate("SHG")
        w = self.window
        w.loaded = LoadedState(mode="SHG Processing", folder="", shg_data=object())
        spin = w.shg_peak_center_spin
        spin.setFocus()
        self.app.processEvents()
        controller = w.shg_controller
        with patch.object(type(controller), "_shg_settings_from_ui", autospec=True,
                          side_effect=ValueError("Invalid settings")) as settings:
            spin.selectAll()
            QTest.keyClicks(spin, "600")
            controller._start_shg_reprocess()
            settings.assert_not_called()
            QTest.keyClick(spin, Qt.Key_Return)
            controller._start_shg_reprocess()
            settings.assert_called_once_with(controller)

    def test_second_drr_color_steps_render_only_once(self):
        self.activate("DRR")
        w = self.window
        w.loaded = LoadedState(mode="DRR", folder="")
        w.last_plotted_mode = "DRR"
        spin = w.drr_second_vmax_spin
        spin.setFocus()
        self.app.processEvents()
        with patch.object(w, "_plot_mode") as plot, patch.object(
            w, "_prepare_drr_display_derivative", return_value=True
        ):
            for _ in range(3):
                QTest.keyClick(spin, Qt.Key_Up)
                QTest.qWait(110)
            plot.assert_not_called()
            QTest.qWait(160)
            plot.assert_called_once_with("DRR")

    def test_derivative_completion_preserves_uncommitted_range_text(self):
        self.activate("DRR")
        w = self.window
        source = object()
        w.loaded = LoadedState(mode="DRR", folder="", cube=source)
        spin = w.drr_spins["vmax"]
        spin.setFocus()
        self.app.processEvents()
        pending = (source, {"refresh_split": True})
        w._drr_pending_ranges = pending
        with patch.object(w, "_refresh_automatic_ranges", side_effect=lambda *args, **kwargs:
                          w._set_spin_value_silent(spin, 42)) as refresh, patch.object(
            w, "_prepare_drr_display_derivative", return_value=True
        ), patch.object(w, "_plot_mode") as plot:
            spin.selectAll()
            QTest.keyClicks(spin, "600")
            w._on_drr_display_ready()
            refresh.assert_not_called()
            self.assertIs(w._drr_pending_ranges, pending)
            self.assertTrue(spin.lineEdit().isModified())
            self.assertIn("600", spin.text())
            QTest.keyClick(spin, Qt.Key_Return)
            QTest.qWait(250)
            refresh.assert_called_once_with("DRR", refresh_split=True)
            plot.assert_called_once_with("DRR")

    def test_previous_redraw_waits_for_label_commit_too(self):
        self.activate("PL")
        w = self.window
        w.loaded = LoadedState(mode="PL", folder="")
        edit = w.pl_dat_yaxis_label_edit
        edit.show()
        edit.setFocus()
        self.app.processEvents()
        with patch.object(w, "_plot_mode") as plot:
            w._schedule_plot_redraw("PL", delay_ms=0)
            edit.selectAll()
            QTest.keyClicks(edit, "Voltage")
            QTest.qWait(30)
            plot.assert_not_called()
            QTest.keyClick(edit, Qt.Key_Return)
            QTest.qWait(160)
            plot.assert_called_once_with("PL")


if __name__ == "__main__":
    unittest.main()
