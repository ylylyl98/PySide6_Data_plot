"""Real-window interactions with explicit file-backed, disposable settings."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from tests.ui_test_helpers import dispose_owned_window
from ui_qt.main_window import MainWindow
from ui_qt.theme import install_theme


class WorkspaceRedesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = QSettings(str(Path(temporary.name) / "ui.ini"), QSettings.IniFormat)
        self.theme = install_theme(self.app, mode="light")
        with (patch("ui_qt.main_window.QSettings", lambda *_: self.settings),
              patch("ui_qt.presentation_widget.QSettings", lambda *_: self.settings),
              patch.object(MainWindow, "_restore_last_folder", lambda _: None),
              patch.object(MainWindow, "_schedule_automatic_update_check", lambda _: None)):
            self.window = MainWindow(drr_display_backend="quick")
        self.addCleanup(dispose_owned_window, self.window)
        self.window.resize(1180, 700)
        self.window.show()
        self.app.processEvents()

    def test_navigation_updates_context_and_sidebar_toggle_reclaims_canvas_width(self):
        w = self.window
        w.workflow_tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertEqual(w.workspace_shell.title_label.text(), "Differential reflectance")
        self.assertEqual(w.tabs.currentIndex(), 1)
        original = w.canvas.width()
        w.sidebar_toggle_btn.click()
        self.app.processEvents()
        self.assertFalse(w.left_panel.isVisible())
        self.assertGreater(w.canvas.width(), original + 250)
        w.show_sidebar_action.trigger()
        self.app.processEvents()
        self.assertTrue(w.left_panel.isVisible())
        self.assertTrue(w.sidebar_toggle_btn.isChecked())

    def test_appearance_action_updates_quick_and_persists_only_selected_preference(self):
        w = self.window
        w.workflow_tabs.setCurrentIndex(1)
        self.app.processEvents()
        self.assertTrue(w.drr_display_host.ensure_loaded())
        before = w.drr_display_host.quick_widget.rootObject().property("color")
        w.appearance_actions["dark"].trigger()
        self.app.processEvents()
        self.assertTrue(w.appearance_actions["dark"].isChecked())
        self.assertFalse(w.appearance_actions["light"].isChecked())
        self.assertEqual(self.settings.value("ui/appearance"), "dark")
        after = w.drr_display_host.quick_widget.rootObject().property("color")
        self.assertNotEqual(before, after)

    def test_busy_feedback_does_not_resize_the_plot_canvas(self):
        w = self.window
        original = w.canvas.size()
        w._set_stage("Loading...")
        self.app.processEvents()
        self.assertTrue(w._status_progress.isVisible())
        self.assertEqual(w.status_bar_view._busy_label.text(), "Loading…")
        self.assertEqual(w.canvas.size(), original)

        w._set_stage("No data")
        self.app.processEvents()
        self.assertFalse(w._status_progress.isVisible())
        self.assertEqual(w.canvas.size(), original)

    def test_export_feedback_survives_redraw_and_load_completion(self):
        w = self.window
        w._export_in_progress = True
        w._set_stage("Exporting...")
        w._set_stage("Plotted")
        w._on_load_finished()
        self.app.processEvents()
        self.assertTrue(w._status_progress.isVisible())
        self.assertEqual(w.status_bar_view._busy_label.text(), "Exporting…")
        w._export_in_progress = False
        w._set_stage("Exported")
        self.app.processEvents()
        self.assertFalse(w._status_progress.isVisible())

    def test_failed_first_load_updates_empty_canvas_without_a_redraw(self):
        w = self.window
        w._set_stage("Loading...")
        self.assertIn("Loading measurements", w.empty_canvas_overlay.text())
        with patch("ui_qt.main_window.QMessageBox.critical"):
            w._show_error("Synthetic read error")
        self.assertIn("Data could not be loaded", w.empty_canvas_overlay.text())
        w._on_load_finished()
        self.assertIn("Data could not be loaded", w.empty_canvas_overlay.text())

    def test_comparison_hides_only_navigation_and_honors_manual_override(self):
        w = self.window
        shell = w.workspace_shell
        w.pl_view_side_btn.click()
        self.app.processEvents()
        self.assertFalse(shell.is_rail_visible())
        self.assertTrue(w.left_panel.isVisible())
        w.pl_view_side_btn.click()
        self.assertTrue(shell.is_rail_visible())
        w.workflow_tabs.setCurrentIndex(1)
        w.drr_view_side_btn.click()
        self.assertFalse(shell.is_rail_visible())
        self.assertTrue(w.left_panel.isVisible())
        shell.rail_toggle_btn.click()
        self.assertTrue(shell.is_rail_visible())
        w.drr_view_side_btn.click()
        self.assertTrue(shell.is_rail_visible())


if __name__ == "__main__":
    unittest.main()
