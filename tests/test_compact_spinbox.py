import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QLineEdit
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from ui_qt.compact_spinbox import CompactDoubleSpinBox


class CompactSpinTests(unittest.TestCase):
    def test_enter_commits_typed_number_without_keyboard_tracking(self):
        app=QApplication.instance() or QApplication([])
        spin=CompactDoubleSpinBox();spin.setDecimals(9);spin.setKeyboardTracking(False);spin.setValue(1.)
        try:
            spin.show();spin.setFocus();app.processEvents();spin.selectAll()
            QTest.keyClicks(spin,'1.02');QTest.keyClick(spin,Qt.Key_Return)
            self.assertEqual(spin.value(),1.02)
        finally:spin.close()

    def test_fixed_display_keeps_trailing_zeros_and_internal_precision(self):
        app=QApplication.instance() or QApplication([])
        spin=CompactDoubleSpinBox();spin.setDecimals(9)
        spin.display_precision=4;spin.keep_trailing_zeros=True
        spin.setSingleStep(.0001);spin.setValue(1.820000031)
        self.assertEqual(spin.textFromValue(spin.value()),'1.8200')
        self.assertEqual(spin.value(),1.820000031)
        spin.stepUp();self.assertAlmostEqual(spin.value(),1.820100031,places=9)

    def test_focus_keeps_precision_but_typing_changes_value(self):
        app=QApplication.instance() or QApplication([])
        host=QWidget();layout=QVBoxLayout(host)
        spin=CompactDoubleSpinBox();spin.setDecimals(9);spin.setValue(1.234567891)
        other=QLineEdit();layout.addWidget(spin);layout.addWidget(other)
        try:
            host.show();spin.setFocus();app.processEvents();other.setFocus();app.processEvents()
            self.assertEqual(spin.value(),1.234567891)
            self.assertEqual(spin.textFromValue(spin.value()),'1.235')
            spin.setFocus();spin.selectAll();QTest.keyClicks(spin,'1.235');QTest.keyClick(spin,Qt.Key_Tab)
            app.processEvents();self.assertEqual(spin.value(),1.235)
        finally:host.close()
