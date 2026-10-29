"""Native PL display controls exercised with a Widgets-free QGuiApplication.

The subprocess isolates Qt application type, configuration and synthetic data.
Numeric drafts below are entered through real key events, not QML properties.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class QuickPlDisplayTests(unittest.TestCase):
    def test_native_pl_display_in_isolated_process(self):
        with TemporaryDirectory(prefix="quick-pl-display-process-") as temporary:
            environment = dict(
                os.environ, QT_QPA_PLATFORM="offscreen", QT_QUICK_BACKEND="software",
                MPLBACKEND="agg", MPLCONFIGDIR=temporary, XDG_CONFIG_HOME=temporary,
                XDG_CACHE_HOME=temporary, LOCALAPPDATA=temporary, APPDATA=temporary,
                PYTHONIOENCODING="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--isolated"],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=240,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("QUICK_PL_DISPLAY_TESTS=", result.stdout)


def _run_isolated():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib.abc
    import json
    import time
    import traceback

    class NoWidgets(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {
                "PySide6.QtWidgets", "PySide6.QtQuickWidgets", "ui_qt.common",
                "ui_qt.main_window", "matplotlib.backends.backend_qt",
                "matplotlib.backends.backend_qtagg",
            }:
                raise AssertionError("Native PL display imported Widgets: " + fullname)

    sys.meta_path.insert(0, NoWidgets())
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPointF, QSettings, Qt, QUrl, qInstallMessageHandler
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication, QKeyEvent, QMouseEvent
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtQuickControls2 import QQuickStyle
    from ui_qt.quick.application_bridge import QuickApplicationBridge

    initial_imports = set(sys.modules)
    QQuickStyle.setStyle("Basic")
    app = QGuiApplication([])
    if sys.platform == "win32":
        QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        app.setFont(QFont("Segoe UI", 9))
    python_errors, qml_errors = [], []

    def uncaught(kind, value, trace):
        python_errors.append("".join(traceback.format_exception(kind, value, trace)))
        traceback.print_exception(kind, value, trace)

    sys.excepthook = uncaught

    def message(kind, context, text):
        if any(marker in text for marker in (
            "ReferenceError", "TypeError", "Cannot assign", "Unable to assign", "Binding loop", "Error:",
        )):
            qml_errors.append(text)
            print(text, file=sys.stderr)

    previous_handler = qInstallMessageHandler(message)

    def settle():
        for _ in range(4):
            app.processEvents()
            time.sleep(.005)

    def wait(predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                settle()
                return
            time.sleep(.005)
        raise AssertionError("Timed out waiting for native PL display")

    class Cases(unittest.TestCase):
        def setUp(self):
            pstart, qstart = len(python_errors), len(qml_errors)
            self.addCleanup(lambda: self.assertEqual(python_errors[pstart:], []))
            self.addCleanup(lambda: self.assertEqual(qml_errors[qstart:], []))
            self.temporary = TemporaryDirectory(prefix="quick-pl-display-case-")
            self.addCleanup(self.temporary.cleanup)
            self.root = Path(self.temporary.name).resolve()
            self.bridge = QuickApplicationBridge(settings=QSettings(str(self.root / "settings.ini"), QSettings.IniFormat))
            self.addCleanup(self.bridge.close)
            self.state = self.bridge.plDisplayBridge
            self.bridge.setParameter("pl.autoNext", False)

        def idle(self):
            wait(lambda: not self.bridge._pending and not self.bridge._busy
                 and not self.bridge._plot_timer.isActive()
                 and not self.bridge._pl_range_timer.isActive())

        def load(self):
            import numpy as np
            energy, gate = np.linspace(1., 3., 21), np.array([-1., 0., 1.])
            values = 1 + np.arange(63).reshape(3, 21) / 10
            rows = ["X\t" + "\t".join(map(str, gate))]
            rows.extend("\t".join(map(str, [x, *values[:, index]])) for index, x in enumerate(energy))
            path = self.root / "synthetic.dat"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            self.bridge.openFile(QUrl.fromLocalFile(str(path)))
            self.idle()
            self.assertIsNotNone(self.bridge._cube, self.bridge.snapshot["status"])

        def window(self, *, full=False, width=300):
            engine = QQmlApplicationEngine()
            qml_path = Path(__file__).resolve().parents[1] / "ui_qt/quick/qml"
            if full:
                from ui_qt.quick.plot_surface import register_plot_surface
                register_plot_surface()
                engine.setInitialProperties({"appBridge": self.bridge})
                engine.load(QUrl.fromLocalFile(str(qml_path / "Application.qml")))
            else:
                engine.setInitialProperties({"displayBridge": self.state, "displayColors": self.bridge.colors})
                engine.loadData(b'''import QtQuick
import QtQuick.Window
Window {
    required property var displayBridge
    required property var displayColors
    width: 300; height: 1800; visible: true
    PlDisplay { anchors.fill: parent; bridge: displayBridge; colors: displayColors }
}''', QUrl.fromLocalFile(str(qml_path / "PlDisplayHarness.qml")))
            self.assertTrue(engine.rootObjects(), "\n".join(qml_errors))
            window = engine.rootObjects()[0]
            window.setWidth(1320 if full else width)
            window.setHeight(1000 if full else 1800)

            def dispose():
                window.close()
                engine.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                settle()

            self.addCleanup(dispose)
            settle()
            return window

        def item(self, window, name):
            found = window.findChild(QObject, name)
            if found is None:
                def visit(parent):
                    if parent.objectName() == name:
                        return parent
                    for child in parent.childItems():
                        result = visit(child)
                        if result is not None:
                            return result
                    return None
                found = visit(window.contentItem())
            self.assertIsNotNone(found, name)
            return found

        def click(self, window, item):
            self.assertTrue(item.isVisible(), item.objectName())
            self.assertTrue(item.isEnabled(), item.objectName())
            point = item.mapToScene(item.boundingRect().center())
            for kind, buttons in ((QEvent.MouseButtonPress, Qt.LeftButton), (QEvent.MouseButtonRelease, Qt.NoButton)):
                QCoreApplication.sendEvent(window, QMouseEvent(kind, point, point, Qt.LeftButton, buttons, Qt.NoModifier))
            settle()

        def key(self, window, key, text="", modifiers=Qt.NoModifier):
            for kind in (QEvent.KeyPress, QEvent.KeyRelease):
                QCoreApplication.sendEvent(window, QKeyEvent(kind, key, modifiers, text))
            settle()

        def edit(self, window, key, text):
            field = self.item(window, "drr_input_" + key)
            field.forceActiveFocus()
            settle()
            self.key(window, Qt.Key_A, "a", Qt.ControlModifier)
            for character in text:
                self.key(window, 0, character)
            self.assertEqual(field.property("text"), text)
            return field

        def value(self, key):
            return self.state.snapshot["numbers"][key]["value"]

        def capture_review_images(self, window):
            destination = os.environ.get("PL_DISPLAY_SCREENSHOT_DIR")
            if not destination:
                return
            folder = Path(destination)
            folder.mkdir(parents=True, exist_ok=True)
            wait(lambda: not window.grabWindow().isNull())
            self.assertTrue(window.grabWindow().save(str(folder / "normal.png")))
            pane = self.item(window, "app_parameter_pane")
            start = pane.mapToScene(QPointF(pane.width() + 2, pane.height() / 2))
            end = start - QPointF(pane.width() - 220, 0)
            for kind, point, button, buttons in (
                (QEvent.MouseButtonPress, start, Qt.LeftButton, Qt.LeftButton),
                (QEvent.MouseMove, end, Qt.NoButton, Qt.LeftButton),
                (QEvent.MouseButtonRelease, end, Qt.LeftButton, Qt.NoButton),
            ):
                QCoreApplication.sendEvent(window, QMouseEvent(kind, point, point, button, buttons, Qt.NoModifier))
                settle()
            self.assertLessEqual(pane.width(), 225, "Review screenshot must show the narrow sidebar")
            self.assertTrue(window.grabWindow().save(str(folder / "narrow.png")))

        def test_native_application_integration_and_initial_state(self):
            self.assertIs(type(app), QGuiApplication)
            for module in ("PySide6.QtWidgets", "ui_qt.main_window", "core.loader", "numpy", "matplotlib"):
                self.assertNotIn(module, initial_imports)
            self.assertIs(self.state, self.bridge.plDisplayBridge)
            self.load()
            window = self.window(full=True)
            self.assertTrue(self.item(window, "pl_native_parameters").isVisible())
            self.assertTrue(self.item(window, "pl_display").isVisible())
            self.assertEqual(self.state.snapshot["scale"], "linear")
            cmap = self.state.snapshot["choices"]["cmap"]
            options = cmap.get("options", cmap.get("items"))
            self.assertEqual(options[cmap["index"]]["text"], "turbo")
            for name in ("drr_range_axis.xmin", "drr_range_axis.ymin", "drr_choice_regionCount", "pl_tab_linear", "pl_tab_log"):
                self.assertIsNotNone(self.item(window, name))
            self.capture_review_images(window)

        def test_keyboard_enter_tab_escape_and_arrows(self):
            self.load()
            window = self.window()
            before = self.value("axis.xmin")
            self.edit(window, "axis.xmin", "1.25")
            self.assertEqual(self.value("axis.xmin"), before)
            self.key(window, Qt.Key_Return, "\r")
            self.assertAlmostEqual(self.value("axis.xmin"), 1.25)
            self.edit(window, "axis.xmin", "1.3")
            self.key(window, Qt.Key_Tab, "\t")
            self.assertAlmostEqual(self.value("axis.xmin"), 1.3)
            self.edit(window, "axis.xmin", "1.4")
            self.key(window, Qt.Key_Escape)
            self.assertAlmostEqual(self.value("axis.xmin"), 1.3)
            self.key(window, Qt.Key_Up)
            stepped = self.value("axis.xmin")
            self.assertGreater(stepped, 1.3)
            settle()
            field = self.item(window, "drr_input_axis.xmin")
            self.assertEqual(field.property("text"), self.state.snapshot["numbers"]["axis.xmin"]["text"])
            self.key(window, Qt.Key_Return, "\r")
            self.assertEqual(self.value("axis.xmin"), stepped)
            self.key(window, Qt.Key_Down)
            self.assertAlmostEqual(self.value("axis.xmin"), 1.3)

        def test_invalid_draft_blocks_scale_switch_and_survives_refresh(self):
            self.load()
            window = self.window()
            before = self.value("linear.vmin")
            field = self.edit(window, "linear.vmin", "nan")
            self.state.setFlag("clip", not self.state.snapshot["flags"]["clip"]["checked"])
            settle()
            self.assertIs(field, self.item(window, "drr_input_linear.vmin"))
            self.assertEqual(field.property("text"), "nan")
            self.click(window, self.item(window, "pl_tab_log"))
            self.assertEqual(self.state.snapshot["scale"], "linear")
            self.assertEqual(self.value("linear.vmin"), before)
            self.assertTrue(self.item(window, "drr_number_linear.vmin").property("errorText"))
            field.forceActiveFocus()
            self.key(window, Qt.Key_Escape)
            self.click(window, self.item(window, "pl_tab_log"))
            self.assertEqual(self.state.snapshot["scale"], "log")

        def test_auto_preserves_only_checked_fixed_limits(self):
            self.load()
            window = self.window()
            self.edit(window, "axis.xmin", "1.25")
            self.key(window, Qt.Key_Return, "\r")
            self.assertFalse(self.state.snapshot["flags"]["fix.axis.xmin"]["checked"])
            self.click(window, self.item(window, "drr_flag_fix.axis.xmin"))
            self.edit(window, "axis.xmax", "2.75")
            self.key(window, Qt.Key_Return, "\r")
            self.click(window, self.item(window, "drr_action_auto.x"))
            self.assertAlmostEqual(self.value("axis.xmin"), 1.25)
            self.assertAlmostEqual(self.value("axis.xmax"), 3.)
            self.assertTrue(self.state.snapshot["flags"]["fix.axis.xmin"]["checked"])
            self.click(window, self.item(window, "drr_flag_fix.axis.xmin"))
            self.click(window, self.item(window, "drr_action_auto.x"))
            self.assertAlmostEqual(self.value("axis.xmin"), 1.)
            automatic_max = self.value("linear.vmax")
            self.edit(window, "linear.vmin", ".5")
            self.key(window, Qt.Key_Return, "\r")
            self.click(window, self.item(window, "drr_flag_fix.linear.vmin"))
            self.edit(window, "linear.vmax", "6")
            self.key(window, Qt.Key_Return, "\r")
            self.click(window, self.item(window, "drr_action_auto.linear"))
            self.assertAlmostEqual(self.value("linear.vmin"), .5)
            self.assertAlmostEqual(self.value("linear.vmax"), automatic_max)
            self.assertTrue(self.state.snapshot["flags"]["fix.linear.vmin"]["checked"])

        def test_independent_scale_banks_and_three_region_controls(self):
            self.load()
            window = self.window()
            self.edit(window, "linear.vmin", ".5")
            self.key(window, Qt.Key_Return, "\r")
            self.click(window, self.item(window, "pl_tab_log"))
            self.edit(window, "log.vmin", ".25")
            self.key(window, Qt.Key_Return, "\r")
            self.click(window, self.item(window, "pl_tab_linear"))
            self.assertAlmostEqual(self.value("linear.vmin"), .5)
            self.assertAlmostEqual(self.value("log.vmin"), .25)
            choice = self.item(window, "drr_choice_regionCount")
            choice.forceActiveFocus()
            self.key(window, Qt.Key_Down)
            self.key(window, Qt.Key_Down)
            self.assertEqual(self.state.snapshot["regionCount"], 3)
            for key in ("boundary.x0", "boundary.x1", "linear.left_vmin", "linear.middle_vmin", "linear.right_vmin"):
                self.assertTrue(self.item(window, "drr_input_" + key).isVisible(), key)
            self.assertLess(self.value("boundary.x0"), self.value("boundary.x1"))
            for key, value in (("linear.left_vmin", .1), ("linear.middle_vmin", .2), ("linear.right_vmin", .3)):
                self.edit(window, key, str(value))
                self.key(window, Qt.Key_Return, "\r")
                self.assertAlmostEqual(self.value(key), value)

        def test_narrow_sidebar_keeps_min_max_auto_and_fix_usable(self):
            self.load()
            window = self.window()
            for width in (220, 260, 320):
                with self.subTest(width=width):
                    window.setWidth(width)
                    settle()
                    for name in ("drr_input_axis.xmin", "drr_input_axis.xmax", "drr_action_auto.x", "drr_flag_fix.axis.xmin", "pl_tab_linear", "pl_tab_log"):
                        item = self.item(window, name)
                        self.assertTrue(item.isVisible(), name)
                        self.assertGreater(item.width(), 12, name)
                        left = item.mapToScene(item.boundingRect().topLeft()).x()
                        right = item.mapToScene(item.boundingRect().topRight()).x()
                        self.assertGreaterEqual(left, -.5, name)
                        self.assertLessEqual(right, width + .5, name)
                    self.edit(window, "axis.xmin", "1.2")
                    self.key(window, Qt.Key_Return, "\r")
                    self.assertAlmostEqual(self.value("axis.xmin"), 1.2)
                    self.click(window, self.item(window, "drr_action_auto.x"))
                    self.assertAlmostEqual(self.value("axis.xmin"), 1.)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    qInstallMessageHandler(previous_handler)
    print("QUICK_PL_DISPLAY_TESTS=" + json.dumps({"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(_run_isolated())
    unittest.main()
