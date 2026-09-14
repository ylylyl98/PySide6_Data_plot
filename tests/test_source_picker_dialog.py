from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QAbstractItemView, QApplication, QDialog, QWidget

from ui_qt.source_picker_dialog import SourcePickerDialog


class SourcePickerDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_shell_builds_common_controls_and_configures_wrapped_list(self) -> None:
        dialog = SourcePickerDialog(
            title="Choose source",
            hint="Choose one source.",
        )
        try:
            self.assertEqual(dialog.windowTitle(), "Choose source")
            self.assertEqual(dialog.hint_label.text(), "Choose one source.")
            self.assertEqual(dialog.filter_edit.placeholderText(), "Search filename...")
            self.assertEqual(
                dialog.source_list.selectionMode(), QAbstractItemView.SingleSelection
            )
            self.assertEqual(dialog.source_list.textElideMode(), Qt.ElideNone)
            self.assertTrue(dialog.source_list.wordWrap())
            self.assertIsNotNone(dialog.source_list.itemDelegate())
            self.assertFalse(dialog.ok_button.isEnabled())
            self.assertEqual(dialog.details_label.minimumHeight(), 42)
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_repopulate_preserves_selection_and_falls_back_to_only_item(self) -> None:
        dialog = SourcePickerDialog(title="Choose source", selected="b.csv")
        try:
            dialog.repopulate(
                lambda widget: [
                    widget.addItem(item)
                    for item in (
                        self._item("a.csv"),
                        self._item("b.csv"),
                    )
                ]
            )
            self.assertEqual(dialog.selected_source(), "b.csv")

            dialog.repopulate(lambda widget: widget.addItem(self._item("c.csv")))
            self.assertEqual(dialog.selected_source(), "c.csv")
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_text_filter_is_debounced_and_double_click_accepts(self) -> None:
        dialog = SourcePickerDialog(title="Choose source", filter_interval=30)
        spy = QSignalSpy(dialog.filter_requested)
        try:
            dialog.filter_edit.setText("first")
            self.assertEqual(spy.count(), 0)
            QTest.qWait(45)
            self.app.processEvents()
            self.assertEqual(spy.count(), 1)

            dialog.source_list.addItem(self._item("first.csv"))
            dialog.source_list.setCurrentRow(0)
            dialog.source_list.itemDoubleClicked.emit(dialog.source_list.currentItem())
            self.assertEqual(dialog.result(), QDialog.Accepted)
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_unchanged_repopulate_preserves_rows_scroll_and_selection(self):
        dialog = SourcePickerDialog(title="Choose source")
        self.addCleanup(dialog.close)
        def populate(widget):
            for i in range(60):
                widget.addItem(self._item(f"long_measurement_filename_{i}.csv"))
        dialog.repopulate(populate)
        dialog.show()
        self.app.processEvents()
        dialog.source_list.setCurrentRow(25)
        bar = dialog.source_list.verticalScrollBar()
        bar.setValue(bar.maximum() // 2)
        position = bar.value()
        resets = QSignalSpy(dialog.source_list.model().modelReset)
        dialog.repopulate(populate)
        self.assertEqual(resets.count(), 0)
        self.assertEqual(dialog.selected_source(), "long_measurement_filename_25.csv")
        self.assertEqual(bar.value(), position)

    def test_repopulate_updates_changed_status_and_flags_for_same_source(self):
        dialog = SourcePickerDialog(title="Choose source", selected="a.csv")
        self.addCleanup(dialog.close)
        dialog.repopulate(lambda widget: widget.addItem(self._item("a.csv")))
        def populate(widget):
            item = self._item("a.csv")
            item.setText("PROCESSED — a.csv")
            item.setToolTip("Saved today")
            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            widget.addItem(item)
        dialog.repopulate(populate)
        self.assertEqual(dialog.source_list.item(0).text(), "PROCESSED — a.csv")
        self.assertEqual(dialog.source_list.item(0).toolTip(), "Saved today")
        self.assertFalse(dialog.source_list.item(0).flags() & Qt.ItemIsEnabled)

    def test_zero_interval_filter_emits_synchronously(self) -> None:
        dialog = SourcePickerDialog(title="Choose source", filter_interval=0)
        spy = QSignalSpy(dialog.filter_requested)
        try:
            dialog.filter_edit.setText("first")
            self.assertEqual(spy.count(), 1)
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_restore_selection_clears_nonmatching_multiple_items(self) -> None:
        dialog = SourcePickerDialog(title="Choose source")
        try:
            dialog.source_list.addItem(self._item("first.csv"))
            dialog.source_list.addItem(self._item("second.csv"))
            dialog.source_list.setCurrentRow(0)
            dialog.restore_selection("missing.csv")
            self.assertIsNone(dialog.source_list.currentItem())
            self.assertFalse(dialog.ok_button.isEnabled())
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_accept_reject_buttons_and_refresh_are_exposed(self) -> None:
        dialog = SourcePickerDialog(title="Choose source")
        refreshed = []
        dialog.refresh_button.clicked.connect(lambda: refreshed.append(True))
        try:
            dialog.source_list.addItem(self._item("source.csv"))
            dialog.source_list.setCurrentRow(0)
            self.assertTrue(dialog.ok_button.isEnabled())
            dialog.refresh_button.click()
            self.assertEqual(refreshed, [True])
            dialog.button_box.rejected.emit()
            self.assertEqual(dialog.result(), QDialog.Rejected)
        finally:
            dialog.close()
            dialog.deleteLater()

    @staticmethod
    def _item(source: str):
        from PySide6.QtWidgets import QListWidgetItem

        item = QListWidgetItem(source)
        item.setData(Qt.UserRole, source)
        return item


if __name__ == "__main__":
    unittest.main()
