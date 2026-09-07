from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QListView,
    QListWidget,
    QStyleOptionViewItem,
)

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

    def test_spacing_is_accounted_for_when_sizing_painted_text(self) -> None:
        class CapturingDelegate(WrappedFilenameDelegate):
            def paint(self, painter, option, index):
                self.paint_option = QStyleOptionViewItem(option)
                return super().paint(painter, option, index)

        file_list = QListWidget()
        file_list.resize(330, 180)
        file_list.setWordWrap(True)
        file_list.setTextElideMode(Qt.ElideNone)
        file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        file_list.setUniformItemSizes(False)
        file_list.setResizeMode(QListView.Fixed)
        file_list.setSpacing(3)
        delegate = CapturingDelegate(file_list)
        file_list.setItemDelegate(delegate)
        try:
            file_list.addItem("probe.csv")
            file_list.show()
            self.app.processEvents()
            option = delegate.paint_option
            metrics = QFontMetrics(option.font)
            measured_width = option.rect.width() + 2 * file_list.spacing() - 2 * delegate.HORIZONTAL_PADDING
            painted_width = option.rect.width() - 2 * delegate.HORIZONTAL_PADDING
            stem = ""
            for count in range(1, 200):
                for suffix_count in range(12):
                    candidate = "W" * count + "i" * suffix_count
                    filename_candidate = candidate + ".csv"
                    measured_height = metrics.boundingRect(
                        QRect(0, 0, measured_width, 10000), Qt.AlignLeft | Qt.TextWrapAnywhere,
                        filename_candidate,
                    ).height()
                    painted_height = metrics.boundingRect(
                        QRect(0, 0, painted_width, 10000), Qt.AlignLeft | Qt.TextWrapAnywhere,
                        filename_candidate,
                    ).height()
                    if painted_height > measured_height:
                        stem = candidate
                        break
                if stem:
                    break
            self.assertTrue(stem)
            file_list.clear()
            file_list.addItem(stem + ".csv")
            self.app.processEvents()
            option = delegate.paint_option
            expected = metrics.boundingRect(
                option.rect.adjusted(
                    delegate.HORIZONTAL_PADDING, delegate.VERTICAL_PADDING,
                    -delegate.HORIZONTAL_PADDING, -delegate.VERTICAL_PADDING,
                ), Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWrapAnywhere,
                stem + ".csv",
            ).height() + 2 * delegate.VERTICAL_PADDING
            self.assertGreaterEqual(file_list.visualItemRect(file_list.item(0)).height(), expected)
        finally:
            file_list.close()


if __name__ == "__main__":
    unittest.main()
