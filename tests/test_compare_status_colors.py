import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QWidget

from tests.test_compare_source_workflow import _CompareOwner
from ui_qt.controllers_compare import CompareController
from ui_qt.source_picker_dialog import SourcePickerDialog
from ui_qt.theme import alias as theme_alias


class CompareStatusColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_compare_group_rows_use_status_theme_colors_and_keep_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner = QWidget()
            template = _CompareOwner()
            for name, value in template.__dict__.items():
                setattr(owner, name, value)
            owner.current_folder = str(Path(tmp))
            controller = CompareController(owner)
            groups = [
                SimpleNamespace(key=status, label=f"{status} group", sources=())
                for status in ("new", "processed", "mixed", "unknown")
            ]
            seen = {}

            def inspect(dialog):
                seen["rows"] = [
                    (
                        dialog.source_list.item(index).text(),
                        dialog.source_list.item(index).foreground().color(),
                        dialog.source_list.item(index).font().bold(),
                    )
                    for index in range(dialog.source_list.count())
                ]
                dialog.source_list.setCurrentRow(1)
                seen["selected"] = dialog.source_list.currentItem().data(Qt.UserRole)
                dialog.reject()
                return SourcePickerDialog.Rejected

            with (
                patch.object(CompareController, "_cmp_source_groups", return_value=groups),
                patch.object(CompareController, "_cmp_group_status", side_effect=lambda group: group.key),
                patch.object(CompareController, "_cmp_refresh_history_cache"),
                patch.object(SourcePickerDialog, "exec", inspect),
            ):
                controller._cmp_open_group_dialog()

            rows = {row[0].split(" — ", 1)[0]: row for row in seen["rows"]}
            self.assertEqual(rows["● NEW"][1], QColor(theme_alias("source_new_foreground")))
            self.assertEqual(rows["✓ PROCESSED"][1], QColor(theme_alias("source_processed_foreground")))
            self.assertEqual(rows["◐ MIXED"][1], QColor(theme_alias("source_new_foreground")))
            self.assertEqual(rows["? HISTORY UNKNOWN"][1], QColor(theme_alias("source_new_foreground")))
            self.assertTrue(rows["● NEW"][2])
            self.assertFalse(rows["✓ PROCESSED"][2])
            self.assertTrue(rows["◐ MIXED"][2])
            self.assertTrue(rows["? HISTORY UNKNOWN"][2])
            self.assertEqual(seen["selected"], "processed")


if __name__ == "__main__":
    unittest.main()
