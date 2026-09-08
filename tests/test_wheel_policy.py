import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDateTimeEdit, QDial, QDoubleSpinBox,
    QListWidget, QScrollArea, QSlider, QSpinBox, QTabBar, QVBoxLayout, QWidget,
)

from ui_qt.wheel_policy import install_wheel_value_guard


class WheelPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.guard = install_wheel_value_guard(cls.app)

    def wheel(self, widget, delta=-120, modifiers=Qt.KeyboardModifier.NoModifier):
        local = QPointF(5, 5)
        event = QWheelEvent(local, QPointF(widget.mapToGlobal(QPoint(5, 5))),
                            QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                            modifiers, Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(widget, event)

    def test_all_value_controls_and_inner_editors_remain_unchanged(self):
        controls = [QComboBox(), QSpinBox(), QDoubleSpinBox(), QDateTimeEdit(), QSlider(), QDial(), QTabBar()]
        controls[0].addItems(["First", "Second", "Third"])
        for title in ("First", "Second", "Third"):
            controls[-1].addTab(title)
        for control in controls:
            try:
                if isinstance(control, (QComboBox, QTabBar)):
                    control.setCurrentIndex(1)
                    current = control.currentIndex
                elif isinstance(control, QDateTimeEdit):
                    current = control.dateTime
                else:
                    control.setValue(5)
                    current = control.value
                before = current()
                control.show()
                for focused in (False, True):
                    control.setFocus() if focused else control.clearFocus()
                    for delta in (-120, 120):
                        for modifier in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ControlModifier):
                            self.wheel(control, delta, modifier)
                            self.assertEqual(current(), before, type(control).__name__)
                            if isinstance(control, (QSpinBox, QDoubleSpinBox, QDateTimeEdit)):
                                self.wheel(control.lineEdit(), delta, modifier)
                                self.assertEqual(current(), before)
            finally:
                control.close()
                control.deleteLater()

    def test_wheel_over_control_scrolls_panel_and_keyboard_still_edits(self):
        area = QScrollArea()
        body = QWidget()
        layout = QVBoxLayout(body)
        combo = QComboBox()
        combo.addItems(["First", "Second", "Third"])
        combo.setCurrentIndex(1)
        layout.addWidget(combo)
        layout.addStretch()
        body.setMinimumHeight(1500)
        area.setWidget(body)
        area.setWidgetResizable(True)
        area.resize(300, 200)
        area.show()
        self.app.processEvents()
        try:
            self.wheel(combo)
            self.assertEqual(combo.currentIndex(), 1)
            self.assertGreater(area.verticalScrollBar().value(), 0)
            combo.setFocus()
            QTest.keyClick(combo, Qt.Key.Key_Down)
            self.assertEqual(combo.currentIndex(), 2)
            before = area.verticalScrollBar().value()
            self.wheel(area.verticalScrollBar())
            self.assertGreater(area.verticalScrollBar().value(), before)
        finally:
            area.close()
            area.deleteLater()

    def test_open_popup_and_file_list_can_scroll_without_selecting(self):
        combo = QComboBox()
        combo.addItems([str(i) for i in range(100)])
        combo.show()
        combo.showPopup()
        self.app.processEvents()
        files = QListWidget()
        files.addItems([str(i) for i in range(100)])
        files.resize(200, 150)
        files.setCurrentRow(0)
        files.show()
        self.app.processEvents()
        try:
            self.wheel(combo.view().viewport())
            self.assertEqual(combo.currentIndex(), 0)
            self.assertGreater(combo.view().verticalScrollBar().value(), 0)
            self.wheel(files.viewport())
            self.assertEqual(files.currentRow(), 0)
            self.assertGreater(files.verticalScrollBar().value(), 0)
        finally:
            combo.hidePopup()
            combo.close()
            files.close()
            combo.deleteLater()
            files.deleteLater()

    def test_power_peak_controls_are_covered_and_install_is_idempotent(self):
        from ui_qt.main_window import MainWindow
        with patch.object(MainWindow, "_restore_last_folder", lambda self: None):
            window = MainWindow()
        try:
            self.assertIs(install_wheel_value_guard(), self.guard)
            peaks = window.power_peak_controller
            from core.power_multi_peaks import PeakSeed
            peaks.set_peaks((PeakSeed(1.6, .02),))
            self.wheel(peaks.peak_table.viewport())
            self.wheel(peaks.model)
            self.wheel(peaks.metric)
            self.assertEqual(peaks.settings().peaks[0].center_ev, 1.6)
            self.assertEqual(peaks.settings().peaks[0].fwhm_ev, .02)
            self.assertEqual(peaks.model.currentText(), "Lorentzian")
            self.assertEqual(peaks.metric.currentText(), "Integrated area")
        finally:
            window.close()
            window.deleteLater()
