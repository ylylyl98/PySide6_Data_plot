from __future__ import annotations

import unittest
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication, QListWidget, QComboBox
from PySide6.QtCore import Qt

from ui_qt.common import LoadedState
from ui_qt.controllers_shg import ShgController


class _ShgOwner:
    def __init__(self, loaded: LoadedState) -> None:
        self.loaded = loaded
        self.invalidated = []
        self.summary_updates = 0
        self.redraws = []

    def _invalidate_active_load(self, mode: str) -> None:
        self.invalidated.append(mode)

    def _shg_update_summary(self) -> None:
        self.summary_updates += 1

    def _schedule_plot_redraw(self, mode: str, delay_ms: int = 90) -> None:
        self.redraws.append((mode, delay_ms))

    def _status(self, _message: str) -> None:
        pass


class _Index:
    def __init__(self, value: int) -> None:
        self.value = value

    def currentIndex(self) -> int:
        return self.value


class _Enabled:
    def setEnabled(self, _value: bool) -> None:
        pass


class ShgControllerRegressionTests(unittest.TestCase):
    def test_source_filter_uses_cached_status_and_preserves_hidden_selection(self) -> None:
        QApplication.instance() or QApplication([])

        class Owner:
            available_files = ["new.csv", "saved.csv", "legacy.csv"]
            shg_processed_status = {"saved.csv": "2026-09-11T12:00:00+00:00"}
            shg_processing_ambiguous = {"legacy.csv"}
            _shg_source_mtime_cache = {"new.csv": 3.0, "saved.csv": 2.0, "legacy.csv": 1.0}

            def __init__(self):
                self.shg_files = QListWidget()
                for source in self.available_files:
                    item = self.shg_files.addItem(source)
                self.shg_files.item(1).setSelected(True)
                self.shg_source_filter_combo = QComboBox()

            @staticmethod
            def _selected(widget):
                return [item.text() for item in widget.selectedItems()]

        owner = Owner()
        controller = ShgController(owner)
        controller._shg_selected_file()
        owner.shg_source_filter_combo.addItem("History unknown", "unknown")
        owner.shg_source_filter_combo.setCurrentIndex(0)
        controller._on_shg_source_filter_changed(0)
        self.assertEqual([owner.shg_files.item(i).data(Qt.UserRole) for i in range(owner.shg_files.count())], ["legacy.csv"])
        self.assertEqual(controller._shg_selected_file(), "saved.csv")
        owner.shg_source_filter_combo.addItem("All", "all")
        owner.shg_source_filter_combo.setCurrentIndex(1)
        controller._on_shg_source_filter_changed(1)
        self.assertEqual(controller._shg_selected_file(), "saved.csv")

    def test_reprocess_result_rejects_same_roles_from_new_folder_and_load(self) -> None:
        first = LoadedState(
            mode="SHG Processing",
            folder="folder-a",
            selected_files=["same.csv"],
            shg_data=SimpleNamespace(source_file="same.csv"),
            shg_result=None,
        )
        second = LoadedState(
            mode="SHG Processing",
            folder="folder-b",
            selected_files=["same.csv"],
            shg_data=SimpleNamespace(source_file="same.csv"),
            shg_result=None,
        )
        owner = _ShgOwner(second)
        controller = ShgController(owner)
        self.assertTrue(callable(getattr(controller, "_shg_source_identity", None)))
        controller._shg_reprocess_generation = 4
        controller._shg_reprocess_key = (4, object(), object(), False)
        controller._shg_reprocess_source_key = controller._shg_source_identity(first)

        controller._on_shg_reprocessed(4, ("new", None, None, None, None))

        self.assertIsNone(second.shg_result)
        self.assertEqual(owner.summary_updates, 0)

    def test_incomplete_workflow_switch_invalidates_active_shg_load(self) -> None:
        loaded = LoadedState(mode="SHG Processing", folder="folder-a")
        owner = _ShgOwner(loaded)
        owner.shg_workflow_tabs = _Index(1)
        owner.shg_fit_branch_spin = _Enabled()
        controller = ShgController(owner)

        controller._shg_inputs_complete = lambda: False
        controller._on_shg_workflow_changed()

        self.assertEqual(owner.invalidated, ["SHG Processing"])

    def test_export_query_blocks_while_current_load_reprocess_is_active_or_pending(self) -> None:
        data = SimpleNamespace(source_file="same.csv")
        loaded = LoadedState(
            mode="SHG Processing",
            folder="folder-a",
            selected_files=["same.csv"],
            shg_data=data,
            shg_result=object(),
        )
        owner = _ShgOwner(loaded)
        controller = ShgController(owner)
        self.assertTrue(callable(getattr(controller, "_shg_reprocess_pending_for_export", None)))
        identity = controller._shg_source_identity(loaded)
        controller._shg_reprocess_source_key = identity
        controller._shg_reprocess_key = (2, object(), object(), False)
        controller._shg_reprocess_workers = [object()]

        self.assertTrue(controller._shg_reprocess_pending_for_export())

        controller._shg_loaded_processing_key = controller._shg_reprocess_key
        controller._shg_reprocess_pending_payload = (
            data, None, object(), object(), None, None, False, identity,
        )
        self.assertTrue(controller._shg_reprocess_pending_for_export())


if __name__ == "__main__":
    unittest.main()
