from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QListWidget, QListWidgetItem

from ui_qt.source_picker_dialog import SourcePickerDialog


class PickerAcceptPendingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _dialog(self) -> SourcePickerDialog:
        dialog = SourcePickerDialog(title="Choose", filter_interval=40)
        item = QListWidgetItem("old.csv")
        item.setData(Qt.UserRole, "old.csv")
        dialog.source_list.addItem(item)
        dialog.source_list.setCurrentRow(0)
        return dialog

    def test_pending_filter_rejects_double_click_and_default_button(self) -> None:
        dialog = self._dialog()
        self.addCleanup(dialog.deleteLater)
        dialog.filter_edit.setText("new")
        self.assertFalse(dialog.ok_button.isEnabled())

        dialog.source_list.itemDoubleClicked.emit(dialog.source_list.currentItem())
        dialog.button_box.accepted.emit()
        self.assertEqual(dialog.result(), QDialog.Rejected)

    def test_pending_filter_rejects_keyboard_enter(self) -> None:
        dialog = self._dialog()
        self.addCleanup(dialog.deleteLater)
        dialog.show()
        self.app.processEvents()
        dialog.filter_edit.setText("new")
        QTest.keyClick(dialog, Qt.Key_Return)
        self.assertEqual(dialog.result(), QDialog.Rejected)

    def test_pending_filter_keeps_ok_disabled_after_selection_change(self) -> None:
        dialog = self._dialog()
        self.addCleanup(dialog.deleteLater)
        dialog.filter_edit.setText("new")
        dialog.source_list.setCurrentRow(-1)
        dialog.source_list.setCurrentRow(0)
        self.assertFalse(dialog.ok_button.isEnabled())

    def test_accept_is_allowed_after_filter_settles(self) -> None:
        dialog = self._dialog()
        self.addCleanup(dialog.deleteLater)
        dialog.filter_edit.setText("new")
        QTest.qWait(60)
        self.app.processEvents()
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Accepted)

    def test_unkeyed_equal_rows_invalidate_remembered_key(self) -> None:
        widget = QListWidget()
        populate = lambda target: target.addItem("a.csv")
        self.assertTrue(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("a",)))
        self.assertFalse(SourcePickerDialog.replace_rows_if_changed(widget, populate))
        self.assertFalse(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("b",)))
        self.assertFalse(SourcePickerDialog.replace_rows_if_changed(widget, populate, content_key=("b",)))

    def test_population_failure_does_not_store_key(self) -> None:
        widget = QListWidget()
        widget.addItem("a.csv")
        SourcePickerDialog.replace_rows_if_changed(widget, lambda target: target.addItem("a.csv"), content_key=("old",))
        with self.assertRaisesRegex(RuntimeError, "populate"):
            SourcePickerDialog.replace_rows_if_changed(
                widget, lambda _target: (_ for _ in ()).throw(RuntimeError("populate")), content_key=("bad",)
            )
        self.assertEqual(getattr(widget, "_source_picker_content_key", None), ("old",))


if __name__ == "__main__":
    unittest.main()
