from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton
from PySide6.QtCore import QObject

from core.mcd_peak_shift import PeakShiftResult
from core.mcd_peak_export import record_peak_shift_history, source_descriptor
from ui_qt.common import Worker
from ui_qt.feature_pages import FeatureTabsMixin
from ui_qt.main_window import _MainWindowThreadPool


def _source(name: str) -> SimpleNamespace:
    return SimpleNamespace(source_file=name, pair_labels=("B",), pair_b=np.array([0.0]))


class _PeakHarness(FeatureTabsMixin, QObject):
    def __init__(self, folder: str, source_name: str = "a.csv") -> None:
        QObject.__init__(self)
        self.current_folder = folder
        self.loaded = SimpleNamespace(mcd_result=_source(source_name))
        self._mcd_peak_analysis_source = self.loaded.mcd_result
        self._mcd_peak_analysis_source_folder = folder
        self._mcd_peak_analysis_source_descriptor = {}
        self.mcd_peak_result = PeakShiftResult(
            np.array([0.0]), np.array(["B"]), ((),), (), "raw pos"
        )
        self.mcd_peak_method_results = {}
        self.mcd_peak_tracker_method_combo = QComboBox(); self.mcd_peak_tracker_method_combo.addItem("Raw spectrum")
        self.mcd_peak_selector_combo = QComboBox()
        self.mcd_peak_k_combo = QComboBox()
        self.mcd_peak_kp_combo = QComboBox()
        self.mcd_peak_source_combo = QComboBox(); self.mcd_peak_source_combo.addItem("Raw R")
        self.mcd_peak_background_combo = QComboBox(); self.mcd_peak_background_combo.addItem("Linear background")
        self.mcd_peak_export_btn = QPushButton()
        self.mcd_peak_status = QLabel()
        self.mcd_peak_export_history = QLabel()
        self._mcd_peak_export_worker = None
        self._mcd_peak_history_worker = None
        self._mcd_peak_history_generation = 0
        self._mcd_peak_history_cache = {}
        self._is_closing = False
        self.thread_pool = _MainWindowThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self._refresh_calls = 0

    def _refresh_mcd_peak_export_history(self) -> None:
        self._refresh_calls += 1


class McdPeakAsyncExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _pump_until(self, predicate, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        self.assertTrue(predicate())

    def test_export_worker_success_and_duplicate_submission_guard(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            release = threading.Event()
            started = threading.Event()

            def delayed_worker(snapshot, path, *, progress=None, log=None):
                started.set()
                release.wait(2.0)
                Path(path).write_text("csv", encoding="utf-8")
                return {"csv_path": str(path)}

            harness = _PeakHarness(folder)
            output = Path(folder) / "peak.csv"
            with patch("ui_qt.feature_pages.QFileDialog.getSaveFileName", return_value=(str(output), "CSV")), \
                 patch("ui_qt.feature_pages.mcd_peak_shift_export_worker", side_effect=delayed_worker), \
                 patch("ui_qt.feature_pages.source_descriptor", wraps=__import__("core.mcd_peak_export", fromlist=["source_descriptor"]).source_descriptor) as descriptor:
                harness._queue_mcd_peak_export()
                self.assertTrue(started.wait(1.0))
                harness._queue_mcd_peak_export()
                self.assertIn("already running", harness.mcd_peak_status.text())
                release.set()
                self._pump_until(lambda: harness._mcd_peak_export_worker is None)
                self.assertTrue(descriptor.call_args_list)
                self.assertTrue(all(not call.kwargs.get("include_hash", True) for call in descriptor.call_args_list))
            self.assertEqual(harness._refresh_calls, 1)
            self.assertTrue(harness.mcd_peak_export_btn.isEnabled())
            harness.thread_pool.waitForDone()

    def test_worker_error_releases_export_state_truthfully(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            def failing_worker(snapshot, path, *, progress=None, log=None):
                raise RuntimeError("writer failed")

            harness = _PeakHarness(folder)
            output = Path(folder) / "peak.csv"
            with patch("ui_qt.feature_pages.QFileDialog.getSaveFileName", return_value=(str(output), "CSV")), \
                 patch("ui_qt.feature_pages.mcd_peak_shift_export_worker", side_effect=failing_worker):
                harness._queue_mcd_peak_export()
                self._pump_until(lambda: harness._mcd_peak_export_worker is None)
            self.assertIn("export failed: writer failed", harness.mcd_peak_status.text())
            self.assertFalse(output.exists())
            harness.thread_pool.waitForDone()

    def test_completion_after_source_switch_does_not_mark_current_source_saved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            release = threading.Event()
            started = threading.Event()

            def delayed_worker(snapshot, path, *, progress=None, log=None):
                started.set(); release.wait(2.0)
                return {"csv_path": str(path)}

            harness = _PeakHarness(folder, "a.csv")
            output = Path(folder) / "peak.csv"
            with patch("ui_qt.feature_pages.QFileDialog.getSaveFileName", return_value=(str(output), "CSV")), \
                 patch("ui_qt.feature_pages.mcd_peak_shift_export_worker", side_effect=delayed_worker):
                harness._queue_mcd_peak_export()
                self.assertTrue(started.wait(1.0))
                harness._mcd_peak_analysis_source = _source("b.csv")
                harness.loaded.mcd_result = harness._mcd_peak_analysis_source
                release.set()
                self._pump_until(lambda: harness._mcd_peak_export_worker is None)
            self.assertEqual(harness._refresh_calls, 0)
            self.assertIn("previous MCD source", harness.mcd_peak_status.text())
            harness.thread_pool.waitForDone()

    def test_switch_away_then_back_invalidates_stale_source_history_cache(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            release = threading.Event()
            started = threading.Event()

            def delayed_worker(snapshot, path, *, progress=None, log=None):
                started.set(); release.wait(2.0)
                return {"csv_path": str(path)}

            harness = _PeakHarness(folder, "a.csv")
            a_descriptor = source_descriptor(folder, "a.csv")
            a_key = str(a_descriptor["path"]).casefold()
            harness._mcd_peak_history_cache[a_key] = ("new", None)
            record_peak_shift_history(
                Path(folder) / "saved_a.csv",
                experiment_folder=folder,
                source=a_descriptor,
            )
            output = Path(folder) / "peak.csv"
            with patch("ui_qt.feature_pages.QFileDialog.getSaveFileName", return_value=(str(output), "CSV")), \
                 patch("ui_qt.feature_pages.mcd_peak_shift_export_worker", side_effect=delayed_worker):
                harness._queue_mcd_peak_export()
                self.assertTrue(started.wait(1.0))
                harness._mcd_peak_analysis_source = _source("b.csv")
                harness.loaded.mcd_result = harness._mcd_peak_analysis_source
                release.set()
                self._pump_until(lambda: harness._mcd_peak_export_worker is None)
                harness._mcd_peak_analysis_source = _source("a.csv")
                FeatureTabsMixin._refresh_mcd_peak_export_history(harness)
                self._pump_until(lambda: harness._mcd_peak_history_worker is None)
            self.assertIn("Saved", harness.mcd_peak_export_history.text())
            harness.thread_pool.waitForDone()

    def test_completion_after_clear_cannot_mark_empty_selection_saved(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            release = threading.Event()
            started = threading.Event()

            def delayed_worker(snapshot, path, *, progress=None, log=None):
                started.set(); release.wait(2.0)
                return {"csv_path": str(path)}

            harness = _PeakHarness(folder, "a.csv")
            output = Path(folder) / "peak.csv"
            with patch("ui_qt.feature_pages.QFileDialog.getSaveFileName", return_value=(str(output), "CSV")), \
                 patch("ui_qt.feature_pages.mcd_peak_shift_export_worker", side_effect=delayed_worker):
                harness._queue_mcd_peak_export()
                self.assertTrue(started.wait(1.0))
                harness._mcd_peak_analysis_source = None
                harness.loaded.mcd_result = None
                release.set()
                self._pump_until(lambda: harness._mcd_peak_export_worker is None)
            self.assertEqual(harness._refresh_calls, 0)
            self.assertIn("previous MCD source", harness.mcd_peak_status.text())
            harness.thread_pool.waitForDone()

    def test_history_callback_from_previous_generation_is_ignored_and_no_hash_requested(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            harness = _PeakHarness(folder)
            harness._mcd_peak_history_worker = object()
            harness._mcd_peak_history_generation = 4
            with patch("ui_qt.feature_pages.source_descriptor", wraps=__import__("core.mcd_peak_export", fromlist=["source_descriptor"]).source_descriptor) as descriptor:
                harness._on_mcd_peak_history_result(harness._mcd_peak_history_worker, 3, ("processed", "stamp"))
                self.assertEqual(harness._mcd_peak_history_cache, {})
                descriptor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
