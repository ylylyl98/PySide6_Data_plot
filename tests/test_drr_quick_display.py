"""Exercise the real QML page against the production DRR controls."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from pathlib import Path
from tempfile import TemporaryDirectory
import cProfile
import pstats
import unittest
from unittest.mock import patch

from PySide6.QtCore import QObject, QSettings, Qt
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tests.ui_test_helpers import dispose_owned_window
from ui_qt.common import LoadedState, pending_numeric_input
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class DrrQuickDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = QSettings(str(Path(self.temporary.name) / "ui.ini"), QSettings.IniFormat)
        self.theme = install_theme(self.app, mode="light")
        with (patch("ui_qt.main_window.QSettings", lambda *_: self.settings),
              patch("ui_qt.presentation_widget.QSettings", lambda *_: self.settings),
              patch.object(MainWindow, "_restore_last_folder", lambda _: None),
              patch.object(MainWindow, "_schedule_automatic_update_check", lambda _: None)):
            self.window = MainWindow(drr_display_backend="quick")
        self.addCleanup(dispose_owned_window, self.window)
        self.host = self.window.drr_display_host

    def activate(self):
        index = next(i for i in range(self.window.tabs.count()) if self.window.tabs.tabText(i) == "DRR")
        self.window.tabs.setCurrentIndex(index)
        self.window.show()
        self.app.processEvents()
        self.assertTrue(self.host.ensure_loaded(), self.host.load_error)
        QTest.qWait(30)
        return self.host.bridge

    def item(self, key):
        item = self.host.quick_widget.rootObject().findChild(QQuickItem, key)
        self.assertIsNotNone(item, key)
        return item

    def edit(self, key, text):
        editor = self.item("drr_input_" + key)
        editor.forceActiveFocus()
        self.app.processEvents()
        QTest.keyClick(self.host.quick_widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(self.host.quick_widget, text)
        return editor

    def test_lazy_page_and_fallback_are_owned_by_window(self):
        self.assertIsNone(self.host.quick_widget)
        self.activate()
        self.assertTrue(self.host.ready)
        self.assertFalse(self.host.legacy_display.isVisible())
        self.assertIs(self.host.bridge.parent(), self.host)
        self.window.hide()
        self.app.processEvents()
        self.assertFalse(self.host.bridge._poll.isActive())
        self.window.show()
        self.app.processEvents()
        self.assertTrue(self.host.bridge._poll.isActive())

    def test_real_qml_draft_enter_and_tab_commit_once(self):
        bridge = self.activate()
        spin = self.window.drr_spins["xmin"]
        before = spin.value()
        changes = []
        spin.valueChanged.connect(changes.append)
        self.edit("axis.xmin", "1.234567")
        self.assertEqual(spin.value(), before)
        self.assertEqual(changes, [])
        self.assertIs(pending_numeric_input(self.window), spin)
        QTest.keyClick(self.host.quick_widget, Qt.Key_Return)
        self.assertAlmostEqual(spin.value(), 1.2346, places=4)
        self.assertEqual(len(changes), 1)
        self.assertFalse(bridge.editing)
        QTest.keyClick(self.host.quick_widget, Qt.Key_Tab)
        self.assertEqual(len(changes), 1)
        self.edit("axis.xmin", "1.5")
        QTest.keyClick(self.host.quick_widget, Qt.Key_Tab)
        self.assertEqual(spin.value(), 1.5)
        self.assertEqual(len(changes), 2)

    def test_idle_focus_does_not_defer_updates_and_draft_survives_refresh(self):
        bridge = self.activate()
        editor = self.item("drr_input_axis.xmin")
        editor.forceActiveFocus()
        self.app.processEvents()
        self.assertIsNone(pending_numeric_input(self.window))
        self.edit("axis.xmin", "1.75")
        other = self.window.drr_spins["xmax"]
        blocked = other.blockSignals(True)
        other.setValue(3.123456)
        other.blockSignals(blocked)
        bridge.refresh()
        self.app.processEvents()
        self.assertEqual(editor.property("text"), "1.75")
        self.assertIs(editor, self.item("drr_input_axis.xmin"))
        self.assertEqual(bridge.snapshot["numbers"]["axis.xmax"]["value"], other.value())

    def test_invalid_text_does_not_mutate_and_escape_cancels(self):
        self.activate()
        spin = self.window.drr_spins["xmin"]
        before = spin.value()
        self.edit("axis.xmin", "nan")
        QTest.keyClick(self.host.quick_widget, Qt.Key_Return)
        self.assertEqual(spin.value(), before)
        self.assertTrue(self.host.bridge.error)
        QTest.keyClick(self.host.quick_widget, Qt.Key_Escape)
        self.assertIsNone(pending_numeric_input(self.window))

    def test_draft_defers_existing_redraw_until_commit(self):
        self.activate()
        self.window.loaded = LoadedState(mode="DRR", folder="")
        self.edit("axis.xmin", "1.9")
        with patch.object(self.window, "_plot_mode") as plot:
            self.window._schedule_plot_redraw("DRR", delay_ms=0)
            QTest.qWait(40)
            plot.assert_not_called()
            self.assertIn("DRR", self.window._plot_redraw_pending)
        QTest.keyClick(self.host.quick_widget, Qt.Key_Return)
        self.assertIsNone(pending_numeric_input(self.window))

    def test_quick_arrows_retain_trailing_debounce(self):
        self.activate()
        self.window.loaded = LoadedState(mode="DRR", folder="")
        editor = self.item("drr_input_axis.xmin")
        editor.forceActiveFocus()
        with patch.object(self.window, "_run_scheduled_plot_redraw") as redraw:
            for _ in range(3):
                QTest.keyClick(self.host.quick_widget, Qt.Key_Up)
                self.assertEqual(self.window._plot_redraw_timers["DRR"].interval(), 180)
                QTest.qWait(100)
            redraw.assert_not_called()
            QTest.qWait(120)
            redraw.assert_called_once_with("DRR")

    def test_live_theme_and_framebuffer(self):
        self.activate()
        root = self.host.quick_widget.rootObject()
        light = root.property("color")
        self.theme.set_mode("dark")
        self.app.processEvents()
        self.assertNotEqual(root.property("color"), light)
        image = self.host.quick_widget.grabFramebuffer()
        self.assertFalse(image.isNull())
        self.assertGreater(image.width(), 250)

    def test_switching_parameter_page_commits_the_pending_number(self):
        self.activate()
        self.edit("axis.xmin", "1.75")
        self.window.drr_sidebar_tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(self.window.drr_spins["xmin"].value(), 1.75)
        self.assertIsNone(pending_numeric_input(self.window))
        self.assertFalse(self.host.bridge._poll.isActive())

    def test_invalid_draft_is_reverted_when_leaving_display(self):
        bridge = self.activate()
        before = self.window.drr_spins["xmin"].value()
        self.edit("axis.xmin", "nan")
        self.window.drr_sidebar_tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(self.window.drr_spins["xmin"].value(), before)
        self.assertFalse(bridge.editing)
        self.assertFalse(bridge.error)
        self.window.drr_spins["xmin"].setValue(2.5)
        self.window.drr_sidebar_tabs.setCurrentIndex(0)
        self.app.processEvents()
        self.assertIn("2.5", self.item("drr_input_axis.xmin").property("text"))

    def test_one_change_does_not_convert_the_full_snapshot_for_every_control(self):
        self.activate()
        self.item("drr_input_axis.xmin").forceActiveFocus()
        self.app.processEvents()
        profile = cProfile.Profile()
        profile.enable()
        QTest.keyClick(self.host.quick_widget, Qt.Key_Up)
        profile.disable()
        # Regression: per-control Python property reads converted the full
        # map 340 times and blocked one arrow event for over a second. Bound
        # that expensive interface crossing, without a machine-speed limit.
        calls = sum(value[1] for (filename, _line, name), value in pstats.Stats(profile).stats.items()
                    if filename.endswith("drr_display_bridge.py") and name == "snapshot")
        self.assertLessEqual(calls, 4)

    def test_load_failure_keeps_standard_controls_available(self):
        missing = Path(self.temporary.name) / "missing_host.py"
        with patch("ui_qt.quick.drr_display_host.Path", return_value=missing):
            self.assertFalse(self.host.ensure_loaded())
        self.window.tabs.setCurrentIndex(1)
        self.window.show()
        self.app.processEvents()
        self.assertTrue(self.host.load_error)
        self.assertTrue(self.host.legacy_display.isVisible())
        self.assertFalse(self.host.quick_widget.isVisible())
        self.assertFalse(self.host.bridge._poll.isActive())


if __name__ == "__main__":
    unittest.main()
