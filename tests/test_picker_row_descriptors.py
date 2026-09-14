from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QListWidget

from ui_qt.controllers_shg import ShgController
from ui_qt.source_picker_dialog import SourcePickerDialog


class PickerRowDescriptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_shg_rows_reuse_when_visible_descriptors_are_unchanged(self) -> None:
        owner = SimpleNamespace(
            available_files=["sample.csv"],
            shg_files=QListWidget(),
            shg_processed_status={},
            shg_processing_ambiguous=set(),
            _shg_source_mtime_cache={"sample.csv": 10.0},
            _shg_source_filter_preference="all",
            _selected=lambda _widget: [],
        )
        controller = ShgController(owner)
        resets = QSignalSpy(owner.shg_files.model().modelReset)
        controller._shg_apply_source_filter()
        controller._shg_apply_source_filter()
        self.assertEqual(resets.count(), 1)
        self.assertEqual(owner.shg_files.item(0).data(Qt.UserRole), "sample.csv")

    def test_unkeyed_fallback_and_changed_appearance_repopulate_rows(self) -> None:
        widget = QListWidget()
        def populate(target):
            target.addItem("● NEW — sample.csv")
        self.assertTrue(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("new", "green")))
        self.assertFalse(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("new", "green")))
        self.assertFalse(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("new", "blue")))
        self.assertFalse(SourcePickerDialog.replace_rows_if_changed(widget, populate))
        self.assertIsNone(getattr(widget, "_source_picker_content_key", None))


if __name__ == "__main__":
    unittest.main()
