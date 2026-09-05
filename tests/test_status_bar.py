"""Focused tests for the extracted status-bar shell view."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QStatusBar

from ui_qt.shell.status_bar import StatusBarView
from ui_qt.main_window import MainWindow


class StatusBarViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_view_is_a_status_bar_with_ready_message(self) -> None:
        view = StatusBarView()
        self.assertIsInstance(view, QStatusBar)
        self.assertEqual(view.currentMessage(), "Ready")

    def test_progress_widget_is_indeterminate_hidden_and_bounded(self) -> None:
        view = StatusBarView()
        progress = view.progress
        self.assertEqual(progress.minimum(), 0)
        self.assertEqual(progress.maximum(), 0)
        self.assertEqual(progress.maximumWidth(), 120)
        self.assertFalse(progress.isVisibleTo(view))
        self.assertEqual(progress.toolTip(), "Loading data in background…")

    def test_update_button_is_flat_hidden_and_accessible(self) -> None:
        view = StatusBarView()
        button = view.update_button
        self.assertTrue(button.isFlat())
        self.assertFalse(button.isVisibleTo(view))
        self.assertEqual(button.accessibleName(), "Update status")

    def test_cursor_readback_is_passive_accessible_and_separate_from_message_lane(self) -> None:
        view = StatusBarView()
        readback = view.cursor_readback

        self.assertIsInstance(readback, QLabel)
        self.assertEqual(readback.objectName(), "cursorReadback")
        self.assertEqual(readback.accessibleName(), "Cursor readback")
        self.assertEqual(readback.accessibleIdentifier(), "status.cursorReadback")
        self.assertEqual(readback.property("appRole"), "statusItemPassive")
        self.assertEqual(readback.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertTrue(readback.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))

        view.showMessage("Persistent workflow status")
        view.set_cursor_readback("Hover gate: 1.200 V")

        self.assertEqual(readback.text(), "Hover gate: 1.200 V")
        self.assertEqual(view.currentMessage(), "Persistent workflow status")

    def test_cursor_readback_can_be_cleared_without_touching_message_lane(self) -> None:
        view = StatusBarView()
        view.showMessage("Loading...")
        view.set_cursor_readback("Drag the highlighted MCD band left or right; its width stays fixed.")
        view.set_cursor_readback("")

        self.assertEqual(view.cursor_readback.text(), "")
        self.assertEqual(view.currentMessage(), "Loading...")

    def test_main_window_composes_the_same_widgets(self) -> None:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        try:
            view = window.status_bar_view
            self.assertIsInstance(view, StatusBarView)
            self.assertIs(window.statusBar(), view)
            self.assertIs(window._status_progress, view.progress)
            self.assertIs(window._update_status_button, view.update_button)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
