from __future__ import annotations

import gc
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import weakref
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QTimer
from PySide6.QtWidgets import QApplication

from core.mcd_extract import McdSeries, ProcessedMcdRecord
from ui_qt.mcd_organizer_window import McdOrganizerWindow


class McdOrganizerAsyncExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _window(self, root: Path) -> McdOrganizerWindow:
        window = McdOrganizerWindow(root, auto_scan=False)
        reference = weakref.ref(window)
        self.addCleanup(lambda: self._close_window(reference))
        return window

    def _close_window(self, reference: weakref.ReferenceType[McdOrganizerWindow]) -> None:
        window = reference()
        if window is None:
            return
        try:
            window.close()
            window.deleteLater()
        except RuntimeError:
            return
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()

    def _pump_until(self, predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            QTimer.singleShot(1, lambda: None)
        self.fail("Timed out waiting for Qt event-loop condition")

    @staticmethod
    def _series() -> tuple[McdSeries, ProcessedMcdRecord]:
        record = ProcessedMcdRecord(
            record_id="record-1", settings_path=Path("settings.json"),
            trace_path=Path("trace.csv"), source_file="raw.csv", package="pkg",
            created_utc="2026-09-11T00:00:00Z", center_ev=1.55, width_mev=5.0,
            primary_metric="signed", fit_window_t=0.2,
            acquisition_conditions={"E-field": (0.0, 0.0)},
            condition_sources={"E-field": "raw.csv"},
            increasing_slope_per_t=1.0, decreasing_slope_per_t=-1.0,
        )
        return (
            McdSeries("series-1", "E-field", "E-field series", (record,), {"Temperature": 4.0}),
            record,
        )

    def test_export_runs_off_gui_thread_with_immutable_snapshot_and_bounded_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as settings_root, tempfile.TemporaryDirectory() as folder:
            QSettings.setDefaultFormat(QSettings.Format.IniFormat)
            QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_root)
            root = Path(folder)
            window = self._window(root)
            series, original_record = self._series()
            window._checked_series = lambda: [series]
            window._selected_branches = lambda: ("B increasing",)
            window.output_label.setText(str(root))
            window._update_export_summary()

            entered = threading.Event()
            release = threading.Event()
            calls: list[tuple[int, tuple[ProcessedMcdRecord, ...], str, dict]] = []
            gui_thread = threading.get_ident()

            def fake_export(records, folder_name, *, progress=None, log=None, **options):
                calls.append((threading.get_ident(), records, folder_name, options))
                entered.set()
                release.wait(2.0)
                return {"xlsx": Path(folder_name) / "export.xlsx"}

            heartbeat = [0]
            timer = QTimer(window)
            timer.setInterval(2)
            timer.timeout.connect(lambda: heartbeat.__setitem__(0, heartbeat[0] + 1))
            timer.start()
            with patch("ui_qt.mcd_organizer_window.mcd_extract_export_worker", fake_export), patch(
                "ui_qt.mcd_organizer_window.QMessageBox.information"
            ) as message:
                window._export()
                self._pump_until(entered.is_set)
                heartbeat_before = heartbeat[0]
                deadline = time.monotonic() + 0.1
                while time.monotonic() < deadline:
                    self.app.processEvents()
                self.assertGreater(heartbeat[0], heartbeat_before)
                self.assertEqual(len(calls), 1)
                self.assertIn(window._export_worker, window._export_pool._workers)

                window._export()
                self.assertEqual(len(calls), 1)
                self.assertIn("already running", window.selection_summary.text())

                series.fixed_conditions["Temperature"] = 99.0
                original_record.acquisition_conditions["E-field"] = (99.0, 100.0)
                window.palette_combo.setCurrentIndex(window.palette_combo.findData("plasma"))
                window.export_csv_chk.setChecked(True)
                release.set()
                self._pump_until(lambda: window._export_worker is None)

                thread_id, records, folder_name, options = calls[0]
                self.assertNotEqual(thread_id, gui_thread)
                self.assertEqual(folder_name, str(root))
                self.assertEqual(records[0].acquisition_conditions["E-field"], (0.0, 0.0))
                self.assertEqual(options["palette"], "tab10")
                self.assertFalse(options["export_csv"])
                self.assertEqual(options["series_groups"][0].fixed_conditions["Temperature"], 4.0)
                self.assertEqual(window.selection_summary.text(), "Export complete")
                message.assert_called_once()
            timer.stop()

    def test_export_error_restores_button_and_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            window = self._window(root)
            series, _record = self._series()
            window._checked_series = lambda: [series]
            window._selected_branches = lambda: ("B increasing",)
            window.output_label.setText(str(root))
            window._update_export_summary()
            entered = threading.Event()
            release = threading.Event()

            def failing_export(*_args, **_kwargs):
                entered.set()
                release.wait(2.0)
                raise RuntimeError("synthetic export failure")

            with patch("ui_qt.mcd_organizer_window.mcd_extract_export_worker", failing_export):
                window._export()
                self._pump_until(entered.is_set)
                release.set()
                self._pump_until(lambda: window._export_worker is None)
            self.assertIn("MCD export failed: synthetic export failure", window.selection_summary.text())
            self.assertTrue(window.export_btn.isEnabled())

    def test_close_and_deletion_while_export_active_suppresses_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            window = self._window(root)
            series, _record = self._series()
            window._checked_series = lambda: [series]
            window._selected_branches = lambda: ("B increasing",)
            window.output_label.setText(str(root))
            window._update_export_summary()
            entered = threading.Event()
            release = threading.Event()

            def blocking_export(*_args, **_kwargs):
                entered.set()
                release.wait(2.0)
                return {"xlsx": root / "export.xlsx"}

            reference = weakref.ref(window)
            with patch("ui_qt.mcd_organizer_window.mcd_extract_export_worker", blocking_export), patch(
                "ui_qt.mcd_organizer_window.QMessageBox.information"
            ) as message:
                window._export()
                self._pump_until(entered.is_set)
                window.close()
                window.deleteLater()
                release.set()
                self._pump_until(lambda: not window._export_pool._workers)
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
                self.app.processEvents()
                self.assertEqual(message.call_count, 0)
            del window
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            self.app.processEvents()
            gc.collect()
            self.assertIsNone(reference())


if __name__ == "__main__":
    unittest.main()
