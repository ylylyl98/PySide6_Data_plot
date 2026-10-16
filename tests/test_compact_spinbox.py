import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QLineEdit,QStyle,QStyleOptionSpinBox
from PySide6.QtTest import QTest
from PySide6.QtCore import QLocale, Qt
from ui_qt.compact_spinbox import CompactDoubleSpinBox


class CompactSpinTests(unittest.TestCase):
    def test_adaptive_arrows_use_latest_pair_even_after_silent_updates(self):
        app=QApplication.instance() or QApplication([])
        lower=CompactDoubleSpinBox(significant_digits=4)
        upper=CompactDoubleSpinBox(significant_digits=4)
        for spin in (lower,upper):
            spin.setDecimals(8);spin.setRange(-1e12,1e12)
        lower.set_adaptive_step_peer(upper);upper.set_adaptive_step_peer(lower)
        changes=[];upper.valueChanged.connect(changes.append)
        try:
            for lo,hi,step in ((-100.,100.,1.),(-1.,1.,.01),(-.01,.01,.0001),
                               (.008,.012,.00002),(-200.,-100.,.5),
                               (1000.,1000.,10.),(0.,0.,.01),(-1e-8,1e-8,1e-8)):
                with self.subTest(bounds=(lo,hi)):
                    for spin,value in ((lower,lo),(upper,hi)):
                        spin.blockSignals(True);spin.setValue(value);spin.blockSignals(False)
                    changes.clear()
                    upper.stepUp()
                    self.assertAlmostEqual(upper.value(),hi+step,places=10)
                    self.assertEqual(lower.value(),lo)
                    self.assertEqual(len(changes),1)
            lower.setDecimals(4);upper.setDecimals(4)
            lower.setValue(-.0001);upper.setValue(.0001)
            upper.stepUp()
            self.assertEqual(upper.value(),.0002)
            self.assertEqual(upper.decimals(),4)
        finally:lower.close();upper.close()

    def test_adaptive_arrow_commits_typed_bound_once_before_stepping(self):
        app=QApplication.instance() or QApplication([])
        host=QWidget();layout=QVBoxLayout(host)
        lower=CompactDoubleSpinBox(significant_digits=4)
        upper=CompactDoubleSpinBox(significant_digits=4)
        for spin in (lower,upper):
            spin.setDecimals(8);spin.setRange(-1e12,1e12);layout.addWidget(spin)
        lower.set_adaptive_step_peer(upper);upper.set_adaptive_step_peer(lower)
        changes=[];upper.valueChanged.connect(changes.append)
        try:
            host.show();upper.setFocus();app.processEvents()
            for action in ('keyboard','mouse'):
                with self.subTest(action=action):
                    upper.setValue(200.);changes.clear();upper.selectAll()
                    QTest.keyClicks(upper,'2')
                    self.assertEqual(upper.value(),200.)
                    self.assertEqual(changes,[])
                    if action == 'keyboard':
                        QTest.keyClick(upper,Qt.Key_Up)
                    else:
                        option=QStyleOptionSpinBox();upper.initStyleOption(option)
                        arrow=upper.style().subControlRect(QStyle.CC_SpinBox,option,QStyle.SC_SpinBoxUp,upper)
                        QTest.mouseClick(upper,Qt.LeftButton,pos=arrow.center())
                    self.assertEqual(upper.value(),2.01)
                    self.assertEqual(changes,[2.01])
                    QTest.keyClick(upper,Qt.Key_Down)
                    self.assertEqual(upper.value(),2.)
        finally:host.close()

    def test_significant_display_preserves_small_values_through_focus_and_steps(self):
        app=QApplication.instance() or QApplication([])
        host=QWidget();layout=QVBoxLayout(host)
        spin=CompactDoubleSpinBox(significant_digits=4)
        spin.setLocale(QLocale.c());spin.setDecimals(8);spin.setRange(-1e12,1e12)
        spin.setSingleStep(.00000001)
        other=QLineEdit();layout.addWidget(spin);layout.addWidget(other)
        changes=[];spin.valueChanged.connect(changes.append)
        try:
            host.show();other.setFocus();app.processEvents()
            for value,text in ((0.,'0'),(-.01,'-0.01'),(1.23000000,'1.23'),
                               (1.23456789,'1.235'),(.00000012,'1.2e-07')):
                with self.subTest(value=value):
                    spin.setValue(value);changes.clear()
                    self.assertEqual(spin.text(),text)
                    spin.setFocus();app.processEvents()
                    self.assertEqual(spin.text(),f'{value:.8f}')
                    other.setFocus();app.processEvents()
                    self.assertEqual(spin.value(),value)
                    self.assertEqual(spin.text(),text)
                    self.assertEqual(changes,[])
            spin.stepUp();self.assertEqual(spin.value(),.00000013)
            spin.setLocale(QLocale(QLocale.German));spin.setValue(.01234567)
            self.assertEqual(spin.text(),'0,01235')
            spin.setFocus();spin.selectAll();QTest.keyClicks(spin,'0,12345678')
            QTest.keyClick(spin,Qt.Key_Return);other.setFocus();app.processEvents()
            self.assertEqual(spin.value(),.12345678)
            self.assertEqual(spin.text(),'0,1235')
        finally:host.close()

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
