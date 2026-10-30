import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest

from PySide6.QtCore import QLocale, Qt, QTimer
from PySide6.QtGui import QValidator
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLineEdit, QVBoxLayout, QWidget

from ui_qt.common import QDoubleSpinBox, QSpinBox
from ui_qt.compact_spinbox import CompactDoubleSpinBox


class NumericInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = QWidget()
        self.layout = QVBoxLayout(self.host)
        self.other = QLineEdit()
        self.layout.addWidget(self.other)

    def tearDown(self):
        self.host.close()

    def add_spin(self, spin):
        self.layout.insertWidget(0, spin)
        self.host.show()
        self.other.setFocus()
        self.app.processEvents()
        return spin

    def type_value(self, spin, text, key=Qt.Key_Return):
        spin.setFocus()
        self.app.processEvents()
        spin.selectAll()
        QTest.keyClicks(spin, text)
        self.assertEqual(spin.cleanText(), text)
        QTest.keyClick(spin, key)
        self.app.processEvents()

    def test_scientific_double_keys_locale_affixes_and_commits(self):
        for kind in (QDoubleSpinBox, CompactDoubleSpinBox):
            spin = kind()
            spin.setDecimals(8)
            spin.setRange(-1e12, 1e12)
            spin.setPrefix('E: ')
            spin.setSuffix(' eV')
            self.add_spin(spin)
            for locale, examples in ((QLocale.c(), (('1e3', 1000.), ('1e-6', .000001), ('-2.5e4', -25000.))),
                                     (QLocale(QLocale.German), (('-2,5e4', -25000.), ('1e-6', .000001)))):
                spin.setLocale(locale)
                for text, expected in examples:
                    with self.subTest(kind=kind, locale=locale.name(), text=text):
                        previous = spin.value()
                        changes = []
                        spin.valueChanged.connect(changes.append)
                        self.type_value(spin, text)
                        self.assertEqual(spin.value(), expected)
                        self.other.setFocus()
                        self.app.processEvents()
                        self.assertEqual(spin.value(), expected)
                        self.assertEqual(changes, [] if previous == expected else [expected])
                        spin.valueChanged.disconnect(changes.append)
            spin.close()

    def test_idle_label_focus_enter_blur_and_steps_preserve_value(self):
        spin = QDoubleSpinBox()
        spin.setLocale(QLocale.c())
        spin.setDecimals(9)
        spin.setRange(-1e12, 1e12)
        spin.setSingleStep(.000000001)
        spin.setToolTip('Energy boundary')
        self.add_spin(spin)
        for value, label in ((1.234567891, '1.2346'), (.000001237, '1.237e-06'),
                             (.0001, '1e-04'), (-.00005, '-5e-05'), (0., '0'),
                             (1.2e8, '1.2e+08')):
            with self.subTest(value=value):
                spin.setValue(value)
                self.assertEqual(spin.text(), label)
                changes = []
                spin.valueChanged.connect(changes.append)
                spin.setFocus()
                self.app.processEvents()
                self.assertEqual(spin.text(), label)
                QTest.keyClick(spin, Qt.Key_Return)
                self.assertEqual(spin.text(), label)
                self.other.setFocus()
                self.app.processEvents()
                self.assertEqual(spin.value(), value)
                self.assertEqual(spin.text(), label)
                self.assertEqual(changes, [])
                spin.valueChanged.disconnect(changes.append)
        spin.setValue(1.234567891)
        spin.stepUp()
        self.assertEqual(spin.value(), 1.234567892)
        self.assertIn('Energy boundary', spin.toolTip())
        self.assertIn('1.234567892', spin.toolTip())
        spin.display_significant_digits = 5
        self.assertEqual(spin.textFromValue(.0001), '1e-04')

    def test_range_validation_and_storage_precision_remain_native(self):
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(-10., 10.)
        self.add_spin(spin)
        for text in ('1e', '1e-', '1e3'):
            self.assertEqual(spin.validate(text, len(text))[0], QValidator.Intermediate)
        for text in ('NaN', '1e9999', '1e--2'):
            self.assertEqual(spin.validate(text, len(text))[0], QValidator.Invalid)
        spin.setValue(2.)
        self.type_value(spin, '1.234567e0')
        self.assertEqual(spin.value(), 1.2346)
        self.assertEqual(spin.decimals(), 4)
        self.assertEqual((spin.minimum(), spin.maximum()), (-10., 10.))

    def test_scientific_integer_requires_exact_integral_result(self):
        spin = QSpinBox()
        spin.setRange(-2147483648, 2147483647)
        spin.setKeyboardTracking(False)
        self.add_spin(spin)
        for text, expected in (('1e3', 1000), ('-2.5e4', -25000), ('2.147483647e9', 2147483647)):
            self.type_value(spin, text)
            self.assertEqual(spin.value(), expected)
        for text in ('1e-6', '2.0000000000000001e0', '1e9999'):
            self.assertEqual(spin.validate(text, len(text))[0], QValidator.Invalid)
        self.assertEqual(spin.validate('2.147483648e9', 14)[0], QValidator.Intermediate)
        spin.setLocale(QLocale(QLocale.German))
        self.type_value(spin, '-2,5e4')
        self.assertEqual(spin.value(), -25000)
        spin.stepUp()
        self.assertEqual(spin.value(), -24999)

    def test_manual_mcd_center_popup_accepts_scientific_and_preserves_cancel(self):
        from ui_qt.mcd_unified_page import McdUnifiedView
        view = McdUnifiedView()
        self.layout.addWidget(view)
        captured = []
        view.manual_candidate_added.connect(captured.append)
        for accepted in (True, None, False):
            failures = []

            def edit_popup():
                dialog = QApplication.activeModalWidget()
                try:
                    spin = dialog.findChild(QDoubleSpinBox, 'mcd_manual_center')
                    self.assertIsNotNone(spin)
                    self.assertEqual((spin.minimum(), spin.maximum(), spin.decimals()), (-1e6, 1e6, 6))
                    buttons = dialog.findChild(QDialogButtonBox)
                    if accepted is None:
                        spin.setFocus()
                        spin.selectAll()
                        QTest.keyClicks(spin, '1e')
                        self.assertFalse(spin.hasAcceptableInput())
                        self.assertFalse(buttons.button(QDialogButtonBox.Ok).isEnabled())
                        buttons.button(QDialogButtonBox.Ok).click()
                        self.assertTrue(dialog.isVisible())
                        buttons.button(QDialogButtonBox.Cancel).click()
                    else:
                        self.type_value(spin, '1.643e0')
                        buttons.button(QDialogButtonBox.Ok if accepted else QDialogButtonBox.Cancel).click()
                except Exception as error:
                    failures.append(error)
                    dialog.reject()

            QTimer.singleShot(0, edit_popup)
            view._add_manual_candidate_dialog()
            self.assertEqual(failures, [])
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]['center_ev'], 1.643)


if __name__ == '__main__':
    unittest.main()
