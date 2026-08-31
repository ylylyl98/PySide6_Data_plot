from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core import data_io
from ui_qt.main_window import MainWindow, QComboBox, QDoubleSpinBox, QSpinBox
from tests.ui_test_helpers import wait_for_file_catalog


class PlSourceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _window(self) -> MainWindow:
        with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
            return MainWindow()

    def test_pl_discovery_includes_initial_data_and_saved_dat(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            initial = root / "Initial Data" / "session"
            processed = root / "Processed Data" / "PL"
            initial.mkdir(parents=True)
            processed.mkdir(parents=True)
            (root / "legacy.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (initial / "measurement.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (processed / "measurement_PL_linear.dat").write_text("saved", encoding="utf-8")
            (processed / "measurement_PL_linear.metadata.json").write_text(
                json.dumps(
                    {
                        "workflow": "PL",
                        "created_utc": "2026-08-24T22:00:00+00:00",
                        "sources": [{"name": "Initial Data/session/measurement.csv"}],
                    }
                ),
                encoding="utf-8",
            )

            sources = data_io.list_pl_source_files(root)
            status = data_io.discover_pl_processing_status(root, sources)

        self.assertEqual(
            set(sources),
            {
                "legacy.csv",
                "Initial Data/session/measurement.csv",
                "Processed Data/PL/measurement_PL_linear.dat",
            },
        )
        self.assertEqual(
            status,
            {"Initial Data/session/measurement.csv": "2026-08-24T22:00:00+00:00"},
        )

    def test_pl_chooser_selection_loads_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            (root / "first.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "second.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            window = self._window()
            try:
                window._set_current_folder(str(root), remember=False)
                wait_for_file_catalog(window)
                with (
                    patch.object(window, "_open_pl_source_dialog", return_value="second.csv"),
                    patch.object(window, "_start_load") as start_load,
                ):
                    window._edit_pl_source()
                self.assertEqual(window._selected(window.pl_files), ["second.csv"])
                self.assertIn("● NEW", window.pl_selection_summary.text())
                start_load.assert_called_once_with("PL")
            finally:
                window.close()

    def test_auto_next_uses_newest_unprocessed_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as folder_text:
            root = Path(folder_text)
            for name, modified in (("done.csv", 1000), ("older.csv", 2000), ("newest.csv", 3000)):
                path = root / name
                path.write_text("a,b\n1,2\n", encoding="utf-8")
                os.utime(path, (modified, modified))
            window = self._window()
            try:
                window._set_current_folder(str(root), remember=False)
                wait_for_file_catalog(window)
                window.pl_processed_status = {"done.csv": "2026-08-24T22:00:00+00:00"}
                window.pl_auto_next_chk.setChecked(True)
                with patch.object(window, "_start_load") as start_load:
                    advanced = window._auto_load_next_unprocessed_pl("done.csv")
                self.assertTrue(advanced)
                self.assertEqual(window._selected(window.pl_files), ["newest.csv"])
                start_load.assert_called_once_with("PL")
            finally:
                window.close()

    def test_successful_first_save_triggers_auto_next_but_resave_does_not(self) -> None:
        window = self._window()
        result = {
            "mode": "PL",
            "out_folder": "Processed Data/PL",
            "folder": "experiment",
            "source_files": [],
            "save_status": "created",
        }
        try:
            window._pl_last_export_source = "current.csv"
            window._pl_export_source_was_processed = False
            with (
                patch.object(window, "_refresh_file_lists"),
                patch.object(window, "_auto_load_next_unprocessed_pl") as advance,
            ):
                window._on_export_done(result)
            advance.assert_called_once_with("current.csv")

            window._pl_last_export_source = "current.csv"
            window._pl_export_source_was_processed = True
            with (
                patch.object(window, "_refresh_file_lists"),
                patch.object(window, "_auto_load_next_unprocessed_pl") as advance,
            ):
                window._on_export_done(result)
            advance.assert_not_called()
        finally:
            window.close()

    def test_shared_value_controls_ignore_mouse_wheel(self) -> None:
        class WheelEvent:
            ignored = False

            @staticmethod
            def modifiers():
                return Qt.KeyboardModifier.ControlModifier

            def ignore(self) -> None:
                self.ignored = True

        combo = QComboBox()
        combo.addItems(["A", "B"])
        controls = [combo, QSpinBox(), QDoubleSpinBox()]
        for control in controls:
            event = WheelEvent()
            control.wheelEvent(event)
            self.assertTrue(event.ignored)


if __name__ == "__main__":
    unittest.main()
