"""R2 regressions: Power acceptance must use an owned, fresh worker check."""
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidget, QListWidgetItem, QWidget
from PySide6.QtCore import QThreadPool

from ui_qt.power_group_dialog import PowerGroupDialog, _PowerValidationWorker


class PowerAcceptFreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _source(self, name):
        return SimpleNamespace(file_name=name, records=())

    def test_worker_returns_actual_signatures_and_deletion_is_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "sample.csv")
            path.write_text("Power_uW,1\n1,2\n", encoding="utf-8")
            worker = _PowerValidationWorker(folder, (("g", self._source("sample.csv")),))
            result, error = [], []
            worker.signals.result.connect(result.append)
            worker.signals.error.connect(error.append)
            with patch("core.data_io.load_power_series_cube", return_value=object()):
                worker.run()
            self.assertEqual(result[0]["signatures"][0][0], "sample.csv")
            self.assertEqual(result[0]["signatures"][0][1], path.stat().st_size)
            path.unlink()
            worker = _PowerValidationWorker(folder, (("g", self._source("sample.csv")),))
            worker.signals.error.connect(error.append)
            worker.run()
            self.assertTrue(error)

    def test_worker_rejects_mutation_during_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "sample.csv")
            path.write_text("Power_uW,1\n1,2\n", encoding="utf-8")
            worker = _PowerValidationWorker(folder, (("g", self._source("sample.csv")),))
            errors = []
            worker.signals.error.connect(errors.append)
            def decode(*_args, **_kwargs):
                path.write_text("Power_uW,1\n9,9\n", encoding="utf-8")
                stamp = path.stat().st_mtime_ns + 1_000_000_000
                os.utime(path, ns=(stamp, stamp))
                return object()
            with patch("core.data_io.load_power_series_cube", side_effect=decode):
                worker.run()
            self.assertTrue(errors)
            self.assertIn("changed during validation", errors[0])

    def test_acceptance_does_not_short_circuit_warm_prefetch_cache(self):
        dialog = PowerGroupDialog.__new__(PowerGroupDialog)
        item = QListWidgetItem("group"); item.setData(Qt.UserRole, "group")
        dialog.source_list = QListWidget(); dialog.source_list.addItem(item); dialog.source_list.setCurrentItem(item)
        dialog._groups = [SimpleNamespace(key="group", duplicates={}, issues={}, sources={"s"}, mapping={})]
        dialog._drafts = {"group": {"single": "s"}}
        dialog._sources = {"s": self._source("sample.csv")}
        dialog._folder = "folder"
        dialog.controller = SimpleNamespace(current_folder="folder", _owner=SimpleNamespace(thread_pool=object()))
        dialog.action_combo = SimpleNamespace(currentText=lambda: "Single intensity")
        dialog.pair_combo = SimpleNamespace(currentText=lambda: "Pair by Stage")
        dialog._validation_cache = {("folder", "Single intensity", (), "Pair by Stage"): {"s": object()}}
        dialog._validation_errors = {}
        dialog._validation_jobs = {}
        dialog._manual_edits = set()
        dialog._catalog_generation = 2
        dialog._accept_token_counter = 0
        dialog._dialog_closed = False
        dialog.details_toggle = SimpleNamespace(isChecked=lambda: True)
        dialog.ok_button = SimpleNamespace(setEnabled=Mock(), setText=Mock())
        dialog.set_details = Mock()
        dialog._validate = Mock(return_value="")
        dialog._validation_key = Mock(return_value=("catalog-key",))
        dialog._start_async_validation = Mock(return_value=True)
        dialog._accept_checked()
        dialog._start_async_validation.assert_called_once()
        self.assertTrue(dialog._accept_after_validation)

    def test_old_worker_cannot_accept_after_refresh_or_close(self):
        dialog = PowerGroupDialog.__new__(PowerGroupDialog)
        item = QListWidgetItem("group"); item.setData(Qt.UserRole, "group")
        dialog.source_list = QListWidget(); dialog.source_list.addItem(item); dialog.source_list.setCurrentItem(item)
        dialog.controller = SimpleNamespace(current_folder="folder")
        dialog._folder = "folder"; dialog._dialog_closed = False
        dialog._accept_after_validation = True; dialog._accept_validation_key = "request"
        dialog._accept_validation_group_key = "group"; dialog._accept_validation_token = 1
        dialog._catalog_generation = 3; dialog._accept_validation_generation = 2
        dialog._validation_cache = {}; dialog._validation_errors = {}
        dialog._group_changed = Mock(); dialog.accept = Mock(); dialog.set_details = Mock()
        dialog._on_async_validation_result("request", {"results": {}, "signatures": (("sample.csv", 1, 2),)})
        dialog.accept.assert_not_called()
        dialog._accept_validation_generation = 3; dialog._dialog_closed = True
        dialog._on_async_validation_result("request", {"results": {}, "signatures": (("sample.csv", 1, 2),)})
        dialog.accept.assert_not_called()

    def test_full_callback_chain_keeps_fresh_decoded_results(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sample_1uW.csv").write_text(
                "Power_uW,1.4,1.5,stage_pos\n1,2,3,0\n2,3,4,1\n", encoding="utf-8")
            owner = QWidget(); owner.thread_pool = QThreadPool(owner)
            controller = SimpleNamespace(current_folder=folder, _owner=owner,
                                         _power_measurement_group_key="",
                                         _power_picker_status_filter="All",
                                         _power_include_legacy=False)
            dialog = PowerGroupDialog(controller)
            try:
                dialog._accept_checked()
                deadline = time.monotonic() + 3
                while dialog.result() == 0 and time.monotonic() < deadline:
                    self.app.processEvents(); time.sleep(.01)
                self.assertEqual(dialog.result(), 1)
                self.assertTrue(dialog._validation_results)
            finally:
                dialog.close(); owner.close()

    def test_late_accept_result_cannot_cancel_newer_real_dialog_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sample_1uW.csv").write_text(
                "Power_uW,1.4,1.5,stage_pos\n1,2,3,0\n2,3,4,1\n", encoding="utf-8")
            owner = QWidget()
            class Pool:
                def __init__(self): self.started = []
                def start(self, worker): self.started.append(worker)
            owner.thread_pool = Pool()
            controller = SimpleNamespace(current_folder=folder, _owner=owner,
                                         _power_measurement_group_key="",
                                         _power_picker_status_filter="All",
                                         _power_include_legacy=False)
            dialog = PowerGroupDialog(controller)
            try:
                dialog._accept_checked()
                first = dialog._accept_validation_key
                dialog._accept_checked()
                second = dialog._accept_validation_key
                self.assertNotEqual(first, second)
                dialog._on_async_validation_result(
                    first, {"results": {}, "signatures": (("sample_1uW.csv", 1, 2),)})
                self.assertEqual(dialog._accept_validation_key, second)
                self.assertTrue(dialog._accept_after_validation)
            finally:
                dialog.close(); owner.close()

    def test_prefetch_callback_cannot_cancel_current_accept_or_refresh_generation(self):
        """A catalog-key callback is display-only once accept is pending."""
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sample_1uW.csv").write_text(
                "Power_uW,1.4,1.5,stage_pos\n1,2,3,0\n2,3,4,1\n", encoding="utf-8")
            owner = QWidget()
            class Pool:
                def __init__(self): self.started = []
                def start(self, worker): self.started.append(worker)
            owner.thread_pool = Pool()
            controller = SimpleNamespace(current_folder=folder, _owner=owner,
                                         _power_measurement_group_key="",
                                         _power_picker_status_filter="All",
                                         _power_include_legacy=False)
            dialog = PowerGroupDialog(controller)
            try:
                item = dialog.source_list.currentItem()
                group_key = str(item.data(Qt.UserRole))
                source_key = dialog._drafts[group_key]["single"]
                prefetch_key = dialog._validation_key("Single intensity", [source_key])
                self.assertTrue(dialog._start_async_validation(prefetch_key, [source_key]))
                prefetch_generation = dialog._catalog_generation
                dialog._accept_checked()
                accept_key = dialog._accept_validation_key
                self.assertNotEqual(prefetch_key, accept_key)
                dialog._on_async_validation_result(
                    prefetch_key, {"results": {}, "signatures": ()}, prefetch_generation)
                self.assertTrue(dialog._accept_after_validation)
                self.assertEqual(dialog._accept_validation_key, accept_key)
                dialog._on_async_validation_error(prefetch_key, "late", prefetch_generation)
                self.assertTrue(dialog._accept_after_validation)
                dialog.refresh()
                generation = dialog._catalog_generation
                dialog._on_async_validation_result(prefetch_key, {"results": {}}, generation - 1)
                self.assertFalse(dialog._validation_results)
            finally:
                dialog.close(); owner.close()
if __name__ == "__main__":
    unittest.main()
