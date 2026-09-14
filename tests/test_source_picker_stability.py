from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QDialog, QStyleOptionViewItem
from ui_qt.common import WrappedFilenameDelegate

from ui_qt.main_window import MainWindow
from ui_qt.power_group_dialog import PowerGroupDialog
from ui_qt.source_picker_dialog import SourcePickerDialog
from tests.test_power_group_dialog import _Controller, _table
from core import data_io
from PySide6.QtCore import Qt


class SourcePickerStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def window(self):
        with patch.object(MainWindow, "_restore_last_folder", lambda self: None):
            window = MainWindow()
        self.addCleanup(window.close)
        return window

    def test_power_group_unchanged_refresh_keeps_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, "sample_KK.csv", "KK")
            controller = _Controller(folder)
            dialog = PowerGroupDialog(controller)
            self.addCleanup(controller._owner.close)
            self.addCleanup(dialog.close)
            self.assertGreater(dialog.source_list.count(), 0)
            dialog.source_list.setCurrentRow(0)
            selected = dialog.selected_source()
            resets = QSignalSpy(dialog.source_list.model().modelReset)
            dialog.refresh()
            self.assertEqual(resets.count(), 0)
            self.assertEqual(dialog.selected_source(), selected)

    def test_shg_unchanged_filter_keeps_rows(self):
        window = self.window()
        window.available_files = ["sample.csv", "reference.csv"]
        window.shg_controller._shg_apply_source_filter()
        self.assertGreater(window.shg_files.count(), 0)
        resets = QSignalSpy(window.shg_files.model().modelReset)
        window.shg_controller._shg_apply_source_filter()
        self.assertEqual(resets.count(), 0)

    def test_power_sweep_unchanged_catalog_keeps_selection(self):
        window = self.window()
        with tempfile.TemporaryDirectory() as folder:
            _table(folder, "sample_KK.csv", "KK")
            window.current_folder = folder
            sources = data_io.get_power_series_sources(folder, ["sample_KK.csv"])
            self.assertTrue(sources)
            def execute(dialog):
                dialog.source_list.setCurrentRow(0)
                selected = dialog.selected_source()
                resets = QSignalSpy(dialog.source_list.model().modelReset)
                window.file_catalog_refresh_finished.emit(folder, True)
                self.assertEqual(resets.count(), 0)
                self.assertEqual(dialog.selected_source(), selected)
                dialog.reject()
                return QDialog.Rejected
            with patch.object(type(window.power_controller), "_power_current_sources", return_value=sources), patch.object(SourcePickerDialog, "exec", execute):
                window.power_controller._power_choose_dataset(_force_legacy=True)

    def test_mcd_unchanged_source_refresh_keeps_rows(self):
        window = self.window()
        window.mcd_available_files = ["sample.csv", "reference.csv"]
        with patch("ui_qt.controllers_mcd.discover_mcd_processing_status", return_value={}):
            window.mcd_controller._mcd_refresh_sources()
            resets = QSignalSpy(window.mcd_files.model().modelReset)
            window.mcd_controller._mcd_refresh_sources()
        self.assertEqual(resets.count(), 0)

    def test_pl_and_mcd_repeated_filter_keeps_rows(self):
        window = self.window()
        window.pl_available_files = ["sample_PL.csv", "reference_PL.csv"]
        window.mcd_available_files = ["sample.csv", "reference.csv"]
        window._pl_source_filter_preference = "all"
        window._mcd_source_filter_preference = "all"
        def execute(dialog):
            self.assertGreater(dialog.source_list.count(), 0)
            dialog.source_list.setCurrentRow(0)
            selected = dialog.selected_source()
            resets = QSignalSpy(dialog.source_list.model().modelReset)
            dialog.filter_requested.emit()
            self.assertEqual(resets.count(), 0)
            self.assertEqual(dialog.selected_source(), selected)
            return QDialog.Rejected
        with patch.object(SourcePickerDialog, "exec", execute):
            window.pl_controller._open_pl_source_dialog("")
            window.mcd_controller._open_mcd_source_dialog("")

    def test_pl_unchanged_catalog_completion_keeps_source_rows(self):
        window = self.window()
        window.current_folder = os.getcwd()
        files = ["sample_PL.csv", "reference_PL.csv"]
        window.pl_available_files = files
        window.pl_files.addItems(files)
        resets = QSignalSpy(window.pl_files.model().modelReset)
        window._on_file_lists_result(
            (window.current_folder, files, [], files, {}, [], {},
             {"tag": "catalog_scope", "mode": "PL"}),
            window._file_refresh_generation, True, set(), set(files), set(), set(files), [],
        )
        self.assertEqual(resets.count(), 0)

    def test_compare_unchanged_filter_keeps_rows(self):
        window = self.window()
        window.pl_available_files = [
            "YZ212_1077uW_Rot224deg_Stage200.csv",
            "YZ212_1078uW_Rot269deg_Stage200.csv",
        ]
        window.available_files = list(window.pl_available_files)
        window.cmp_source_filter_combo.setCurrentIndex(1)
        def execute(dialog):
            self.assertGreater(dialog.source_list.count(), 0)
            dialog.source_list.setCurrentRow(0)
            selected = dialog.selected_source()
            resets = QSignalSpy(dialog.source_list.model().modelReset)
            dialog.filter_requested.emit()
            self.assertEqual(resets.count(), 0)
            self.assertEqual(dialog.selected_source(), selected)
            dialog.reject()
            return QDialog.Rejected
        with patch.object(SourcePickerDialog, "exec", execute):
            window.compare_controller._cmp_open_group_dialog()

    def test_shared_picker_first_paint_and_resize_fit_long_filenames(self):
        dialog = SourcePickerDialog(title="Choose source")
        self.addCleanup(dialog.close)
        dialog.repopulate(lambda widget: widget.addItems([
            "sample_PL_760nmc_" + "measurement_token_" * (i + 2) + ".csv\nNEW"
            for i in range(30)
        ]))
        extents = []
        original = WrappedFilenameDelegate.paint
        def paint(delegate, painter, option, index):
            opt = QStyleOptionViewItem(option)
            delegate.initStyleOption(opt, index)
            layout = delegate._layout_text(
                delegate._normalize_text(opt.text), opt.font,
                option.rect.width() - 2 * delegate.HORIZONTAL_PADDING,
            )
            extents.append((option.rect.height(), layout.boundingRect().height()
                            + 2 * delegate.VERTICAL_PADDING))
            original(delegate, painter, option, index)
        with patch.object(WrappedFilenameDelegate, "paint", paint):
            dialog.show()
            for width in (980, 820, 1200):
                dialog.resize(width, 560)
                self.app.processEvents()
                dialog.grab()
        self.assertTrue(extents)
        self.assertTrue(all(actual >= needed for actual, needed in extents), extents)


if __name__ == "__main__":
    unittest.main()
