from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton

from core.drr_sources import DrrSourceCache
from ui_qt.main_window import MainWindow


class DrrPickerStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window(self) -> MainWindow:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            window = MainWindow()
        self.addCleanup(window.deleteLater)
        self.addCleanup(self.app.processEvents)
        return window

    def test_refresh_starts_drr_scan_while_other_catalog_scan_is_running(self):
        window = self._window()
        with tempfile.TemporaryDirectory() as tmp:
            window.current_folder = str(Path(tmp))
            window.available_files = ["old.csv"]
            window._file_refresh_running = True
            with patch.object(MainWindow, "_start_drr_catalog_refresh") as start:
                window._refresh_file_lists(auto=False, mode="DRR")
            start.assert_called_once_with(auto=False, old_source_files={"old.csv"})

    def test_drr_completion_is_emitted_for_a_pending_refresh(self):
        window = self._window()
        with tempfile.TemporaryDirectory() as tmp:
            window.current_folder = str(Path(tmp))
            window._drr_refresh_generation = 1
            window._drr_refresh_running = True
            window._drr_refresh_pending = True
            seen = []
            window.drr_catalog_refresh_finished.connect(
                lambda folder, success: seen.append((folder, success))
            )
            with patch.object(MainWindow, "_start_drr_catalog_refresh"):
                window._on_drr_catalog_refresh_result(
                    (str(Path(tmp)), [], DrrSourceCache()),
                    1,
                    False,
                    set(),
                )
            self.assertEqual(seen, [(str(Path(tmp)), True)])

    def test_picker_does_not_show_refreshing_for_unrelated_file_scan(self):
        window = self._window()
        with tempfile.TemporaryDirectory() as tmp:
            window.current_folder = str(Path(tmp))
            window.drr_available_sources = []
            window._file_refresh_running = True
            window._drr_refresh_running = False
            observed = {}

            def fake_exec(dialog):
                button = next(
                    item for item in dialog.findChildren(QPushButton)
                    if item.text() in {"Refresh", "Refreshing..."}
                )
                observed["enabled"] = button.isEnabled()
                observed["text"] = button.text()
                return QDialog.Rejected

            with patch("ui_qt.controllers_drr.QDialog.exec", fake_exec):
                window.drr_controller._open_drr_source_dialog(
                    title="Choose DRR Measurement Group",
                    selected=[],
                    baseline_mode=False,
                )
            self.assertEqual(observed, {"enabled": True, "text": "Refresh"})

    def test_empty_catalog_shows_empty_hint_after_refresh_finishes(self):
        window = self._window()
        with tempfile.TemporaryDirectory() as tmp:
            window.current_folder = str(Path(tmp))
            window.drr_available_sources = []
            window._drr_refresh_running = True
            observed = {}

            def fake_exec(dialog):
                window.drr_catalog_refresh_finished.emit(window.current_folder, True)
                observed["loading"] = any(
                    item.text() == "Loading DRR catalog…"
                    for item in dialog.findChildren(QLabel)
                )
                return QDialog.Rejected

            with patch("ui_qt.controllers_drr.QDialog.exec", fake_exec):
                window.drr_controller._open_drr_source_dialog(
                    title="Choose DRR Measurement Group",
                    selected=[],
                    baseline_mode=False,
                )
            self.assertFalse(observed["loading"])

    def test_picker_restores_refresh_button_after_success(self):
        window = self._window()
        with tempfile.TemporaryDirectory() as tmp:
            window.current_folder = str(Path(tmp))
            window.drr_available_sources = []
            window._drr_refresh_running = True
            observed = {}

            def fake_exec(dialog):
                window.drr_catalog_refresh_finished.emit(window.current_folder, True)
                button = next(
                    item for item in dialog.findChildren(QPushButton)
                    if item.text() in {"Refresh", "Refreshing..."}
                )
                observed["enabled"] = button.isEnabled()
                observed["text"] = button.text()
                return QDialog.Rejected

            with patch("ui_qt.controllers_drr.QDialog.exec", fake_exec):
                window.drr_controller._open_drr_source_dialog(
                    title="Choose DRR Measurement Group",
                    selected=[],
                    baseline_mode=False,
                )
            self.assertEqual(observed, {"enabled": True, "text": "Refresh"})


if __name__ == "__main__":
    unittest.main()
