import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QListWidget, QListWidgetItem, QStyleOptionViewItem
from ui_qt.common import WrappedFilenameDelegate
from ui_qt.controllers_drr import DrrSessionList, _sync_drr_rows


class PickerLayoutCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_large_catalog_reuses_measurements(self):
        view = QListWidget()
        delegate = WrappedFilenameDelegate(view)
        view.setItemDelegate(delegate)
        view.addItems([f'UNPROCESSED\nlong_filename_{i}_TG_BG_REF.csv' for i in range(700)])
        option = QStyleOptionViewItem()
        with patch.object(delegate, '_layout_text', wraps=delegate._layout_text) as measure:
            for _ in range(2):
                for row in range(view.count()):
                    delegate.sizeHint(option, view.model().index(row, 0))
            self.assertEqual(measure.call_count, 700)
        view.close()

    def test_hidden_population_defers_explicit_layout(self):
        view = DrrSessionList()
        with patch.object(view, 'doItemsLayout') as layout:
            _sync_drr_rows(view, [QListWidgetItem('long_filename.csv')])
            layout.assert_not_called()
        view.close()

    def test_json_cache_survives_new_delegate_and_width_change_remeasures(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'heights.json'
            view = QListWidget()
            view.addItem('status\n' + 'very_long_filename_' * 12 + '.csv')
            option = QStyleOptionViewItem()
            index = view.model().index(0, 0)
            first = WrappedFilenameDelegate(view, cache_path=path)
            expected = first.sizeHint(option, index)
            first.save_size_cache()
            second = WrappedFilenameDelegate(view, cache_path=path)
            with patch.object(second, '_layout_text', wraps=second._layout_text) as measure:
                self.assertEqual(second.sizeHint(option, index), expected)
                measure.assert_not_called()
                view.viewport().resize(150, 300)
                self.assertGreater(second.sizeHint(option, index).height(), expected.height())
                measure.assert_called_once()
            path.write_text('{broken')
            third = WrappedFilenameDelegate(view, cache_path=path)
            self.assertGreater(third.sizeHint(option, index).height(), 0)
            view.close()
