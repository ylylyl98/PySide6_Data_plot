from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QDialog

from ui_qt.controllers_power import PowerController
from ui_qt.main_window import MainWindow
from ui_qt.power_group_dialog import PowerGroupDialog


def _power_table(folder: str, name: str) -> None:
    Path(folder, name).write_text(
        "Power_uW,1.4,1.5,stage_pos\n1,2,3,0\n2,3,4,1\n",
        encoding="utf-8",
    )


class _DialogController:
    def __init__(self, folder: str) -> None:
        from PySide6.QtWidgets import QWidget

        self._owner = QWidget()
        self.current_folder = folder
        self._power_measurement_group_key = ""
        self._power_picker_status_filter = "All"
        self._power_include_legacy = False


class _Pool:
    def __init__(self) -> None:
        self.started = []

    def start(self, worker) -> None:
        self.started.append(worker)


class _Check:
    def __init__(self, checked: bool) -> None:
        self.checked = checked

    def isChecked(self) -> bool:
        return self.checked


class PowerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_browsing_to_group_b_cancels_group_a_async_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _power_table(folder, "sample_a_1uW.csv")
            _power_table(folder, "sample_b_2uW.csv")
            controller = _DialogController(folder)
            controller._owner.thread_pool = _Pool()
            dialog = PowerGroupDialog(controller)
            try:
                keys = [str(dialog.source_list.item(i).data(Qt.UserRole)) for i in range(dialog.source_list.count())]
                self.assertGreaterEqual(len(keys), 2)
                dialog.source_list.setCurrentRow(keys.index("sample_a_1uw"))
                dialog._accept_checked()
                self.assertEqual(len(controller._owner.thread_pool.started), 1)
                accepted_key = dialog._accept_validation_key
                self.assertIsNotNone(accepted_key)

                dialog.source_list.setCurrentRow(keys.index("sample_b_2uw"))
                with patch.object(dialog, "_accept_checked") as accept:
                    dialog._on_async_validation_result(accepted_key, {})

                accept.assert_not_called()
                self.assertFalse(dialog._accept_after_validation)
                self.assertIsNone(dialog._accept_validation_key)
                self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
            finally:
                dialog.close()
                controller._owner.close()

    def test_power_ensure_reloads_when_selected_group_result_cache_was_cleared(self) -> None:
        owner = MainWindow.__new__(MainWindow)
        owner.loaded = SimpleNamespace(
            mode="Power Dependent",
            power_group_key="csv::sample.csv",
            selected_files=["sample.csv"],
        )
        owner.power_compare_chk = _Check(False)
        owner.power_group_combo = QComboBox()
        owner.power_group_combo.addItem("sample", "csv::sample.csv")
        owner._power_result_cache = {}
        owner.power_controller = PowerController(owner)
        owner._start_load = Mock()
        owner._status = Mock()

        MainWindow._ensure_loaded_matches_ui_params(owner, "Power Dependent")

        owner._start_load.assert_called_once_with("Power Dependent")


if __name__ == "__main__":
    unittest.main()
