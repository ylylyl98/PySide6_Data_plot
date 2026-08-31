from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QAbstractItemView, QListView, QListWidget

from ui_qt.main_window import WrappedFilenameDelegate


class WrappedFilenameDelegateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_complete_long_filename_wraps_without_horizontal_scrolling(self) -> None:
        file_list = QListWidget()
        file_list.resize(500, 220)
        file_list.setWordWrap(True)
        file_list.setTextElideMode(Qt.ElideNone)
        file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        file_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        file_list.setUniformItemSizes(False)
        # DRR dialogs use Fixed to avoid a full relayout on every resize; the
        # delegate must still size wrapped rows correctly in that mode.
        file_list.setResizeMode(QListView.Fixed)
        file_list.setItemDelegate(WrappedFilenameDelegate(file_list))
        full_name = (
            "YZ364_0Tpa_3.6KREF_620nmc_0p1sx10_"
            + "very_long_measurement_token_" * 8
            + "TG−1.05BG=0.csv"
        )
        file_list.addItem(full_name)
        file_list.show()
        self.app.processEvents()

        item = file_list.item(0)
        row_rect = file_list.visualItemRect(item)
        single_line_height = file_list.fontMetrics().height()

        self.assertEqual(item.text(), full_name)
        self.assertGreater(row_rect.height(), single_line_height * 2)
        self.assertEqual(row_rect.width(), file_list.viewport().width())
        self.assertEqual(file_list.horizontalScrollBar().maximum(), 0)

        file_list.resize(240, 220)
        self.app.processEvents()
        resized_rect = file_list.visualItemRect(item)
        self.assertGreater(resized_rect.height(), row_rect.height())
        self.assertEqual(resized_rect.width(), file_list.viewport().width())
        file_list.close()


if __name__ == "__main__":
    unittest.main()
