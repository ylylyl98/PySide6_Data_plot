from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QDialog, QListWidget
from core.drr_sources import DrrSource

from ui_qt.main_window import MainWindow


class DrrFilterPreferenceTests(unittest.TestCase):
    KEY = "drr/source_unprocessed_only"

    def test_file_rows_show_newest_timestamp_without_reordering_processing_group(self) -> None:
        timestamp = datetime.now().timestamp()
        self.window.drr_available_sources = [
            DrrSource(name, name, "REF group", datetime.now().strftime("%Y-%m-%d"),
                      modified, False, spectral_grid=(740., 760., 780.))
            for name, modified in (("a_REF.csv", timestamp - 60), ("z_REF.csv", timestamp))
        ]

        def inspect(dialog):
            rows = dialog.findChild(QListWidget, "drr_source_file_list")
            self.assertEqual(rows.count(), 2)
            self.assertEqual(rows.item(0).data(Qt.UserRole), "z_REF.csv")
            self.assertIn(datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"), rows.item(0).toolTip())
            return QDialog.Rejected

        self._open(baseline_mode=False, on_exec=inspect)

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        settings_path = str(Path(self.tmp.name) / "settings.ini")
        self.settings = QSettings(settings_path, QSettings.Format.IniFormat)
        with (
            patch.object(MainWindow, "_restore_last_folder", lambda _self: None),
            patch("ui_qt.main_window.QSettings", return_value=self.settings),
        ):
            self.window = MainWindow()
        self.window.current_folder = self.tmp.name
        self.window.drr_available_sources = []

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.tmp.cleanup()

    def _open(self, *, baseline_mode: bool, on_exec) -> None:
        with patch.object(QDialog, "exec", on_exec):
            self.window.drr_controller._open_drr_source_dialog(
                title="Choose DRR files",
                selected=[],
                baseline_mode=baseline_mode,
            )

    def test_measurement_checkbox_preference_survives_rejected_dialog(self) -> None:
        observed = []

        def fake_exec(dialog: QDialog) -> int:
            checkbox = next(
                item for item in dialog.findChildren(QCheckBox)
                if item.text() == "Unprocessed only"
            )
            observed.append(checkbox.isChecked())
            if len(observed) == 1:
                checkbox.setChecked(False)
            return QDialog.Rejected

        self._open(baseline_mode=False, on_exec=fake_exec)
        self._open(baseline_mode=False, on_exec=fake_exec)

        self.assertEqual(observed, [True, False])
        self.assertFalse(self.settings.value(self.KEY, True, type=bool))

    def test_baseline_dialog_does_not_overwrite_measurement_preference(self) -> None:
        self.settings.setValue(self.KEY, False)
        self.settings.sync()
        observed = {}

        def fake_exec(dialog: QDialog) -> int:
            checkbox = next(
                item for item in dialog.findChildren(QCheckBox)
                if item.text() == "Unprocessed only"
            )
            observed["hidden"] = checkbox.isHidden()
            return QDialog.Rejected

        def measurement_exec(dialog: QDialog) -> int:
            checkbox = next(
                item for item in dialog.findChildren(QCheckBox)
                if item.text() == "Unprocessed only"
            )
            observed["measurement_checked"] = checkbox.isChecked()
            return QDialog.Rejected

        self._open(baseline_mode=True, on_exec=fake_exec)
        self._open(baseline_mode=False, on_exec=measurement_exec)

        self.assertTrue(observed["hidden"])
        self.assertFalse(observed["measurement_checked"])
        self.assertFalse(self.settings.value(self.KEY, True, type=bool))


if __name__ == "__main__":
    unittest.main()
