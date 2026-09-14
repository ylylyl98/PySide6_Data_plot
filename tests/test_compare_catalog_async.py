from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui_qt.controllers_compare import CompareController
from tests.ui_test_helpers import wait_for_file_catalog
from PySide6.QtWidgets import QApplication


class CompareCatalogAsyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_missing_mtime_uses_memory_cache_without_stat(self) -> None:
        owner = SimpleNamespace(
            current_folder="C:/experiment",
            _cmp_source_mtime_cache={"known.csv": 12.5},
            _cmp_source_mtime_cache_folder="C:/experiment",
        )
        controller = CompareController(owner)
        with patch("ui_qt.controllers_compare.Path.stat", side_effect=AssertionError("GUI stat")):
            self.assertEqual(controller._cmp_source_modified("missing.csv"), 0.0)
            self.assertEqual(controller._cmp_source_modified("known.csv"), 12.5)

    def test_cold_history_queues_async_refresh_and_marks_pending(self) -> None:
        calls = []
        owner = SimpleNamespace(
            current_folder="C:/experiment",
            _cmp_history_records=[],
            _cmp_history_cache_folder="",
            _refresh_file_lists=lambda **kwargs: calls.append(kwargs),
        )
        controller = CompareController(owner)
        controller._cmp_refresh_history_cache()
        self.assertEqual(calls, [{"auto": True, "mode": "Compare"}])
        self.assertTrue(controller._cmp_history_cache_pending)
        self.assertFalse(controller._cmp_history_cache_ready)
        self.assertEqual(controller._cmp_history_cache_folder, "C:/experiment")

    def test_forced_warm_history_queues_refresh_and_retains_rows(self) -> None:
        calls = []
        records = [{"created_utc": "old"}]
        owner = SimpleNamespace(
            current_folder="C:/experiment",
            _cmp_history_records=records,
            _cmp_history_cache_folder="C:/experiment",
            _refresh_file_lists=lambda **kwargs: calls.append(kwargs),
        )
        controller = CompareController(owner)
        controller._cmp_refresh_history_cache()
        controller._cmp_refresh_history_cache(force=True)
        self.assertEqual(calls, [{"auto": False, "mode": "Compare"}])
        self.assertEqual(controller._cmp_history_records, records)
        self.assertTrue(controller._cmp_history_cache_pending)

    def test_closed_dialog_publication_marks_empty_cold_snapshot_ready(self) -> None:
        from unittest.mock import patch
        from ui_qt.main_window import MainWindow
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sample_PL.csv").write_text("wavelength,intensity\n500,1\n", encoding="utf-8")
            with patch.object(MainWindow, "_restore_last_folder", lambda _self: None):
                window = MainWindow()
            try:
                window._set_current_folder(folder, remember=False)
                wait_for_file_catalog(window)
                controller = window.compare_controller
                # Exercise the real controller-owned picker lifecycle.  The
                # modal result is rejected, then the worker publication must
                # complete independently of the closed dialog.
                from ui_qt.source_picker_dialog import SourcePickerDialog
                with patch.object(SourcePickerDialog, "exec", lambda _dialog: SourcePickerDialog.Rejected):
                    controller._cmp_open_group_dialog()
                controller._cmp_refresh_history_cache(force=True)
                observed = {"pending_after_close": controller._cmp_history_cache_pending}
                wait_for_file_catalog(window, timeout_ms=30000)
                controller._cmp_refresh_history_cache()
                self.assertTrue(observed["pending_after_close"])
                self.assertIn("Compare", getattr(window, "_catalog_ready_modes", set()))
                self.assertTrue(controller._cmp_history_cache_ready)
                self.assertFalse(controller._cmp_history_cache_pending)
            finally:
                window.close()

    def test_closed_dialog_force_warm_publication_is_ready_on_reopen(self) -> None:
        calls = []
        records = [{"created_utc": "old"}]
        owner = SimpleNamespace(
            current_folder="C:/experiment", _file_refresh_generation=3,
            _catalog_ready_modes=set(),
            _cmp_history_records=records, _cmp_history_cache_folder="C:/experiment",
            _refresh_file_lists=lambda **kwargs: (calls.append(kwargs), setattr(owner, "_file_refresh_generation", 4)),
        )
        controller = CompareController(owner)
        controller._cmp_refresh_history_cache(force=True)
        self.assertTrue(controller._cmp_history_cache_pending)
        controller._cmp_refresh_history_cache()
        self.assertFalse(controller._cmp_history_cache_ready)
        self.assertTrue(controller._cmp_history_cache_pending)
        owner._catalog_ready_modes.add("Compare")
        owner._cmp_history_published_generation = 4
        owner._cmp_history_published_folder = "C:/experiment"
        owner._file_refresh_pending = False
        owner._catalog_pending_requests = {}
        owner._catalog_ready_modes.add("Compare")
        controller._cmp_refresh_history_cache()
        self.assertTrue(controller._cmp_history_cache_ready)
        self.assertFalse(controller._cmp_history_cache_pending)
        self.assertEqual(calls, [{"auto": False, "mode": "Compare"}])

    def test_force_waits_for_final_coalesced_publication(self) -> None:
        records = [{"created_utc": "old"}]
        owner = SimpleNamespace(
            current_folder="C:/experiment", _file_refresh_generation=5,
            _cmp_history_records=[{"created_utc": "new"}],
            _cmp_history_cache_folder="C:/experiment", _catalog_ready_modes={"Compare"},
            _cmp_history_published_generation=5,
            _cmp_history_published_folder="C:/experiment",
            _file_refresh_pending=True, _catalog_pending_requests={"Compare": True},
            _refresh_file_lists=lambda **kwargs: None,
        )
        controller = CompareController(owner)
        controller._cmp_history_records = records
        controller._cmp_history_cache_folder = "C:/experiment"
        controller._cmp_history_cache_ready = False
        controller._cmp_refresh_history_cache(force=True)
        controller._cmp_refresh_history_cache()
        self.assertFalse(controller._cmp_history_cache_ready)
        self.assertTrue(controller._cmp_history_cache_pending)
        owner._file_refresh_pending = False
        owner._catalog_pending_requests = {}
        owner._catalog_ready_modes.add("Compare")
        controller._cmp_refresh_history_cache()
        self.assertTrue(controller._cmp_history_cache_ready)
        self.assertFalse(controller._cmp_history_cache_pending)
        self.assertEqual(controller._cmp_history_records, owner._cmp_history_records)


if __name__ == "__main__":
    unittest.main()
