from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFileDialog


class MCDSharedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _window(self):
        from ui_qt.main_window import MainWindow

        return MainWindow()

    def test_peak_shift_has_compact_controls_for_the_single_mcd_selection(self):
        window = self._window()
        try:
            self.assertIs(window.mcd_peak_select_source_btn.parentWidget().window(), window)
            self.assertIs(window.mcd_peak_clear_source_btn.parentWidget().window(), window)
            self.assertEqual(window._selected(window.mcd_files), [])
            window.mcd_available_files = ["mcd/peak.csv"]
            window.mcd_files.addItem("mcd/peak.csv")
            blocked = window.mcd_files.blockSignals(True)
            window.mcd_files.item(0).setSelected(True)
            window.mcd_files.blockSignals(blocked)
            window.mcd_controller._update_mcd_selection_summary()
            self.assertIn("peak.csv", window.mcd_peak_source_selection_summary.toolTip())
            self.assertIn("peak.csv", window.mcd_peak_source_selection_summary.text())
        finally:
            window.close()

    def test_select_and_clear_from_peak_shift_share_mcd_selection_and_invalidate_peak_result(self):
        window = self._window()
        try:
            window.mcd_available_files = ["mcd/one.csv", "mcd/two.csv"]
            window.mcd_files.addItems(window.mcd_available_files)
            blocked = window.mcd_files.blockSignals(True)
            window.mcd_files.item(0).setSelected(True)
            window.mcd_files.blockSignals(blocked)
            window.loaded = SimpleNamespace(mode="MCD")
            window.mcd_peak_result = object()
            window.last_plotted_mode = "MCD Peak Shift"
            with (
                patch.object(type(window.mcd_controller), "_open_mcd_source_dialog", return_value="mcd/two.csv"),
                patch.object(type(window.mcd_controller), "_mcd_angles_ready", return_value=True),
                patch.object(window, "_start_load") as start_load,
            ):
                window.mcd_peak_select_source_btn.click()
            self.assertEqual(window._selected(window.mcd_files), ["mcd/two.csv"])
            self.assertIsNone(window.mcd_peak_result)
            start_load.assert_called_once_with("MCD")

            window.mcd_peak_clear_source_btn.click()
            self.assertEqual(window._selected(window.mcd_files), [])
            self.assertIsNone(window.mcd_peak_result)
            self.assertIn("No MCD CSV", window.mcd_peak_source_selection_summary.text())
        finally:
            window.close()

    def test_peak_shift_toolbar_load_uses_mcd_pipeline_and_preserves_peak_tab(self):
        window = self._window()
        try:
            peak_index = next(i for i in range(window.tabs.count()) if window.tabs.tabText(i) == "MCD Peak Shift")
            window.tabs.setCurrentIndex(peak_index)
            window.mcd_files.addItem("mcd/peak.csv")
            blocked = window.mcd_files.blockSignals(True)
            window.mcd_files.item(0).setSelected(True)
            window.mcd_files.blockSignals(blocked)
            with patch.object(window, "_start_load") as start_load:
                window._toolbar_load()
            self.assertIsNone(window._active_mode())
            start_load.assert_called_once_with("MCD")
            self.assertEqual(window.tabs.currentIndex(), peak_index)
        finally:
            window.close()

    def test_stale_mcd_load_result_is_ignored_after_shared_source_invalidation(self):
        from ui_qt.main_window import LoadedState

        window = self._window()
        try:
            window._mcd_source_generation = 2
            stale = LoadedState(mode="MCD", folder="", mcd_result=SimpleNamespace(source_file="old.csv"))
            window._on_loaded(stale, load_generation=1)
            self.assertIsNone(window.loaded)
        finally:
            window.close()

    def test_stale_mcd_worker_result_is_ignored_using_sender_generation(self):
        from ui_qt.common import WorkerSignals
        from ui_qt.main_window import LoadedState

        window = self._window()
        signals = WorkerSignals()
        try:
            window._mcd_source_generation = 2
            signals.setProperty("mcd_source_generation", 1)
            signals.result.connect(window._on_loaded)
            signals.result.emit(
                LoadedState(
                    mode="MCD",
                    folder="",
                    primary_file="mcd/old.csv",
                    mcd_result=SimpleNamespace(source_file="mcd/old.csv"),
                )
            )
            self.assertIsNone(window.loaded)
        finally:
            signals.result.disconnect(window._on_loaded)
            window.close()

    def test_relative_mcd_primary_file_is_accepted_from_direct_load_callback(self):
        from ui_qt.main_window import LoadedState

        window = self._window()
        try:
            loaded = LoadedState(
                mode="MCD",
                folder=tempfile.gettempdir(),
                primary_file="mcd/peak.csv",
            )
            with patch.object(window, "_plot_mode") as plot_mode:
                window._on_loaded(loaded)
            self.assertIs(window.loaded, loaded)
            plot_mode.assert_called_once_with("MCD", auto=True)
        finally:
            window.close()

    def test_source_selection_defers_load_until_async_angles_are_ready(self):
        window = self._window()
        try:
            window.current_folder = tempfile.gettempdir()
            window.mcd_available_files = ["mcd/peak.csv"]
            window.mcd_files.addItem("mcd/peak.csv")
            blocked = window.mcd_files.blockSignals(True)
            window.mcd_files.item(0).setSelected(True)
            window.mcd_files.blockSignals(blocked)
            with (
                patch.object(type(window.mcd_controller), "_open_mcd_source_dialog", return_value="mcd/peak.csv"),
                patch.object(type(window.mcd_controller), "_mcd_detect_available_angles"),
                patch.object(type(window.mcd_controller), "_mcd_angles_ready", return_value=False),
                patch.object(window, "_start_load") as start_load,
            ):
                window.mcd_controller._edit_mcd_source()
            start_load.assert_not_called()
            self.assertEqual(
                window.mcd_controller._mcd_load_after_angle_generation,
                window.mcd_controller._mcd_angle_generation,
            )
        finally:
            window.close()

    def test_open_file_selects_mcd_source_after_sync_and_pending_refresh(self):
        window = self._window()
        try:
            with tempfile.TemporaryDirectory() as folder_text:
                folder = Path(folder_text)
                mcd_folder = folder / "mcd"
                mcd_folder.mkdir()
                source = mcd_folder / "peak.csv"
                source.write_text("x,y\n1,2\n", encoding="utf-8")
                window.current_folder = str(folder)
                window.mcd_available_files = ["mcd/peak.csv"]
                window.mcd_files.addItem("mcd/peak.csv")
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(source), "")), patch.object(window, "_set_current_folder", return_value=True), patch.object(window, "_start_load"), patch.object(type(window.mcd_controller), "_mcd_detect_available_angles"):
                    window._open_file()
                self.assertEqual(window._selected(window.mcd_files), ["mcd/peak.csv"])

                window.mcd_files.clearSelection()
                window._file_refresh_running = True
                with patch.object(QFileDialog, "getOpenFileName", return_value=(str(source), "")), patch.object(window, "_set_current_folder", return_value=True):
                    window._open_file()
                self.assertEqual(window._pending_open_file, "mcd/peak.csv")
        finally:
            window.close()
