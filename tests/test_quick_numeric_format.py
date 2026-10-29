"""Numeric presentation policy and real Quick input without production settings."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from pathlib import Path
import unittest

from PySide6.QtCore import QObject, QUrl, Qt, Slot
from PySide6.QtQml import QJSEngine, QQmlComponent, QQmlEngine
from PySide6.QtQuick import QQuickItem, QQuickWindow
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

QML = Path(__file__).resolve().parents[1] / "ui_qt/quick/qml"


class _Bridge(QObject):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.editor = None
        self.descriptor = {}

    @Slot(str, "QVariant")
    def setParameter(self, key, value):
        self.calls.append((key, value))
        self.descriptor = {**self.descriptor, "value": value}
        self.editor.setProperty("descriptor", self.descriptor)


class QuickNumericFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.engine = QQmlEngine()
        self.js = QJSEngine()
        source = (QML / "NumericFormat.js").read_text(encoding="utf-8")
        result = self.js.evaluate(source.replace(".pragma library", "", 1))
        self.assertFalse(result.isError(), result.toString())
        self.addCleanup(self.engine.deleteLater)
        self.addCleanup(self.js.deleteLater)

    def call(self, name, *arguments):
        function = self.js.globalObject().property(name)
        result = function.call([self.js.toScriptValue(value) for value in arguments])
        self.assertFalse(result.isError(), result.toString())
        return result

    def test_semantic_precision_units_and_explicit_overrides(self):
        cases = [
            ({"key": "xmin", "label": "Energy minimum (eV)"}, "1.2346"),
            ({"key": "refit.peak.0.center", "label": "Center (eV)"}, "1.234568"),
            ({"key": "refit.peak.0.lower", "label": "Center min (eV)"}, "1.234568"),
            ({"key": "gate", "label": "Gate voltage (V)"}, "1.235"),
            ({"key": "gate", "label": "Temperature (K)"}, "1.23"),
            ({"key": "gate", "label": "Cursor power (uW)"}, "1.2346"),
            ({"key": "xmin", "label": "Power minimum"}, "1.2346"),
            ({"key": "xmax", "label": "Intensity maximum"}, "1.2346"),
            ({"key": "bCursor", "label": "Magnetic field (T)"}, "1.235"),
            ({"key": "max_delta_b", "label": "Maximum field difference (T)"}, "1.2346"),
            ({"key": "peakCenter", "label": "Peak center (nm)"}, "1.23"),
            ({"key": "angleCursor", "label": "Angle (deg)"}, "1.23"),
            ({"key": "width", "label": "FWHM (meV)"}, "1.235"),
            ({"key": "p2p.style.figure_width", "label": "Figure width"}, "1.2346"),
            ({"key": "factor", "label": "Correction multiplier"}, "1.2346"),
            ({"key": "xmin", "label": "Energy", "displayDecimals": 2}, "1.23"),
            ({"key": "xmin", "label": "Energy", "displayDigits": 3}, "1.23"),
            ({"key": "count", "decimals": 0}, "1"),
        ]
        for descriptor, expected in cases:
            with self.subTest(descriptor=descriptor):
                self.assertEqual(self.call("display", 1.23456789, descriptor).toString(), expected)
        self.assertEqual(self.call("display", 1.00000001, {"label": "Intensity count"}).toString(), "1")
        self.assertFalse(self.call("policy", {"label": "Intensity count"}).toVariant()["integer"])

    def test_scientific_presentation_and_strict_scientific_input(self):
        for value, expected in [(0.000000123456789, "1.2346e-7"), (123456789, "1.2346e8"),
                                (1e-3, "0.001"), (-0., "0"), (1.2, "1.2")]:
            with self.subTest(value=value):
                self.assertEqual(self.call("display", value, {"displayDecimals": 2}).toString(), expected)
        for text, expected in [("1e3", 1000), ("1e-6", 1e-6), ("-2.5e4", -25000),
                               (" +.5E+2 ", 50), ("1.", 1)]:
            self.assertEqual(self.call("parse", text).toNumber(), expected)
        for text in ["0xff", "0b10", "0o10", "NaN", "Infinity", "", " ", "1e", "1e999", "1e-999", "1,2", "2_000"]:
            with self.subTest(text=text):
                self.assertTrue(self.call("parse", text).isUndefined())
        self.assertEqual(self.call("exact", 1.23456789012345).toString(), "1.23456789012345")

    def make_editor(self, **descriptor):
        self.bridge = _Bridge()
        self.bridge.descriptor = {"key": "number", "value": 1.23456789012345, **descriptor}
        component = QQmlComponent(self.engine, QUrl.fromLocalFile(str(QML / "AppNumberField.qml")))
        self.editor = component.createWithInitialProperties({
            "descriptor": self.bridge.descriptor, "bridge": self.bridge,
            "colors": dict(control="#ffffff", text="#111111", muted="#666666", disabled="#888888",
                           error="#cc0000", focus="#0055aa", border="#cccccc", accent="#0055aa",
                           selectedText="#ffffff", pressed="#dddddd", hover="#eeeeee"),
        })
        self.assertIsNotNone(self.editor, "\n".join(error.toString() for error in component.errors()))
        self.bridge.editor = self.editor
        self.window = QQuickWindow()
        self.window.resize(240, 100)
        self.blur = QQuickItem(self.window.contentItem())
        self.editor.setParentItem(self.window.contentItem())
        self.editor.setParent(self.window.contentItem())
        self.editor.setWidth(180)
        self.input = self.editor.findChild(QObject, "app_parameter_number")
        self.assertIsNotNone(self.input)
        self.window.show()
        self.window.requestActivate()
        QTest.qWait(10)

        def dispose():
            self.window.close()
            self.editor.setParentItem(None)
            self.editor.deleteLater()
            self.window.deleteLater()
            self.app.processEvents()
        self.addCleanup(dispose)
        return self.input

    def type_text(self, text):
        self.input.forceActiveFocus()
        QTest.keyClick(self.window, Qt.Key_A, Qt.ControlModifier)
        for character in text:
            QTest.keyClick(self.window, ord(character), Qt.NoModifier)

    def test_native_focus_commit_and_precision_are_independent_of_display(self):
        self.make_editor(label="Energy minimum (eV)")
        self.assertEqual(self.input.property("text"), "1.2346")
        self.input.forceActiveFocus()
        self.assertEqual(self.input.property("text"), "1.23456789012345")
        QTest.keyClick(self.window, Qt.Key_Return)
        self.blur.forceActiveFocus()
        self.assertEqual(self.bridge.calls, [])
        self.assertEqual(self.bridge.descriptor["value"], 1.23456789012345)
        self.type_text("1.23456789012346")
        self.editor.setProperty("descriptor", {**self.bridge.descriptor, "description": "Refreshed"})
        self.assertEqual(self.input.property("text"), "1.23456789012346")
        QTest.keyClick(self.window, Qt.Key_Return)
        self.blur.forceActiveFocus()
        self.assertEqual(self.bridge.calls, [("number", 1.23456789012346)])
        self.assertEqual(self.input.property("text"), "1.2346")

    def test_native_integer_accepts_scientific_notation_and_rejects_fractional_results(self):
        self.make_editor(value=1, decimals=0, min=0, max=10000)
        self.type_text("1e3")
        QTest.keyClick(self.window, Qt.Key_Return)
        self.assertEqual(self.bridge.calls, [("number", 1000)])
        for text in ["1e-3", "0xff", "0b10", "Infinity"]:
            self.type_text(text)
            QTest.keyClick(self.window, Qt.Key_Return)
            self.assertTrue(self.editor.property("invalid"))
            self.assertEqual(self.bridge.calls, [("number", 1000)])
        QTest.keyClick(self.window, Qt.Key_Escape)
        self.assertFalse(self.editor.property("invalid"))
        self.assertEqual(self.input.property("text"), "1000")

    def test_native_presentation_updates_when_units_change_without_a_numeric_change(self):
        self.make_editor(label="Power (uW)")
        self.assertEqual(self.input.property("text"), "1.2346")
        self.editor.setProperty("descriptor", {**self.bridge.descriptor, "label": "Temperature (K)"})
        self.assertEqual(self.input.property("text"), "1.23")
        self.assertEqual(self.bridge.calls, [])

    def test_native_scientific_input_and_steps_do_not_round_storage(self):
        self.make_editor(value=.1, step=.2)
        self.input.forceActiveFocus()
        QTest.keyClick(self.window, Qt.Key_Up)
        self.assertEqual(self.input.property("text"), "0.3")
        self.assertEqual(self.bridge.calls, [("number", .1 + .2)])
        QTest.keyClick(self.window, Qt.Key_Return)
        self.blur.forceActiveFocus()
        self.assertEqual(len(self.bridge.calls), 1)
        self.type_text("1e-6")
        QTest.keyClick(self.window, Qt.Key_Return)
        self.blur.forceActiveFocus()
        self.assertEqual(self.bridge.calls[-1], ("number", 1e-6))
        self.assertEqual(self.input.property("text"), "1e-6")


if __name__ == "__main__":
    unittest.main()
