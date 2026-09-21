from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QListWidget, QStyleOptionViewItem

from core.drr_sources import DrrSource
from ui_qt.common import WrappedFilenameDelegate
from ui_qt.main_window import MainWindow


class DrrPickerStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.settings_folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.settings_folder.cleanup)
        settings = QSettings(self.settings_folder.name + "/settings.ini", QSettings.IniFormat)
        with patch.object(MainWindow, "_restore_last_folder", lambda self: None), patch(
            "ui_qt.main_window.QSettings", return_value=settings
        ), patch.object(MainWindow, "_schedule_automatic_update_check", lambda self: None):
            self.window = MainWindow()
        self.addCleanup(self.window.close)
        self.window.current_folder = os.getcwd()
        self.window.drr_available_sources = [
            DrrSource(
                name, name, "session", "2026-09-13", 1.0, False,
                gate_varies=True, frame_count=2, wavelength_center_nm=760.0,
            )
            for name in [
                "YZ365_p5n2_5T_1.67KREF_760nmc_0p08sx10_RotIn100deg_TG-1.087BG=0.csv",
                "YZ365_p5n2_5T_1.67KREF_760nmc_0p08sx10_RotIn100deg_TG-1.087BG=0_repeat.csv",
            ]
        ]

    def open_picker(self, execute, *, baseline_mode=False):
        with patch.object(QDialog, "exec", execute), patch.object(self.window, '_queue_drr_catalog_refresh'):
            self.window.drr_controller._open_drr_source_dialog(
                title="Choose", selected=[], baseline_mode=baseline_mode,
            )

    def test_filter_reuses_group_tooltips(self):
        from PySide6.QtWidgets import QCheckBox
        from ui_qt import controllers_drr
        def execute(dialog):
            toggle = next(w for w in dialog.findChildren(QCheckBox) if w.text() == 'Show all history')
            with patch.object(controllers_drr.data_io, 'is_xlsx_map_file',
                              wraps=controllers_drr.data_io.is_xlsx_map_file) as classify:
                toggle.setChecked(False)
                toggle.setChecked(True)
                classify.assert_not_called()
            return QDialog.Rejected
        self.open_picker(execute)

    def test_identical_catalog_publication_does_not_rebuild_groups(self):
        from ui_qt import controllers_drr
        def execute(dialog):
            with patch.object(controllers_drr, 'group_drr_sources',
                              wraps=controllers_drr.group_drr_sources) as groups:
                self.window.drr_catalog_preview_ready.emit(self.window.current_folder)
                self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
                groups.assert_not_called()
            return QDialog.Rejected
        self.open_picker(execute)

    def test_reopening_unchanged_picker_reuses_groups_and_row_metadata(self):
        from ui_qt import controllers_drr
        self.open_picker(lambda dialog: QDialog.Rejected)
        with patch.object(controllers_drr, 'group_drr_sources', wraps=controllers_drr.group_drr_sources) as groups, patch.object(
                controllers_drr, 'format_drr_source_summary', wraps=controllers_drr.format_drr_source_summary) as summaries:
            self.open_picker(lambda dialog: QDialog.Rejected)
        groups.assert_not_called()
        summaries.assert_not_called()

    def test_closed_picker_releases_widgets_but_keeps_detached_rows(self):
        from PySide6.QtCore import QCoreApplication, QEvent
        destroyed = []
        def execute(dialog):
            dialog.destroyed.connect(lambda: destroyed.append(True))
            return QDialog.Rejected
        self.open_picker(execute)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.assertEqual(destroyed, [True])
        self.test_reopening_unchanged_picker_reuses_groups_and_row_metadata()

    def test_batch_add_formats_peer_conditions_once(self):
        from PySide6.QtWidgets import QPushButton, QMessageBox
        from ui_qt import controllers_drr
        self.window.drr_available_sources = [replace(
            self.window.drr_available_sources[0], source=f'YZ365_{i}_1.67KREF_760nmc.csv', filename=f'YZ365_{i}_1.67KREF_760nmc.csv')
            for i in range(40)]
        def execute(dialog):
            add = next(w for w in dialog.findChildren(QPushButton) if w.text() == 'Add Entire Group')
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes), patch.object(
                    controllers_drr, '_drr_condition_values', wraps=controllers_drr._drr_condition_values) as values:
                add.click()
                self.assertLessEqual(values.call_count, 80)
            chosen = dialog.findChild(QListWidget, 'drr_source_chosen_list')
            self.assertEqual(chosen.count(), 40)
            add.click()
            self.assertEqual(chosen.count(), 40)
            return QDialog.Rejected
        with patch('ui_qt.controllers_drr.QMessageBox.question', return_value=16384):
            self.open_picker(execute)

    def test_first_painted_rows_fit_wrapped_filenames(self):
        self.window.drr_available_sources = [
            replace(self.window.drr_available_sources[0], group_key=f"session-{i}")
            for i in range(30)
        ]
        extents = []
        original = WrappedFilenameDelegate.paint

        def paint(delegate, painter, option, index):
            if delegate.parent().objectName() == "drr_source_group_list":
                opt = QStyleOptionViewItem(option)
                delegate.initStyleOption(opt, index)
                layout = delegate._layout_text(
                    delegate._normalize_text(opt.text), opt.font,
                    option.rect.width() - 2 * delegate.HORIZONTAL_PADDING,
                )
                extents.append((option.rect.height(), layout.boundingRect().height()
                                + 2 * delegate.VERTICAL_PADDING))
            original(delegate, painter, option, index)

        def execute(dialog):
            dialog.show()
            self.app.processEvents()
            dialog.grab()
            for width in (920, 1300, 1000):
                dialog.resize(width, 560)
                self.app.processEvents()
                dialog.grab()
            dialog.close()
            return QDialog.Rejected

        with patch.object(WrappedFilenameDelegate, "paint", paint):
            self.open_picker(execute)
        self.assertTrue(extents)
        self.assertTrue(all(actual >= needed for actual, needed in extents), extents)

    def test_default_history_includes_older_groups_after_search_and_refresh(self):
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QLineEdit

        self.window.drr_available_sources = [
            replace(self.window.drr_available_sources[0], group_key=f"session-{i:02}",
                    source=f"sample_REF_{i:02}.csv", filename=f"sample_REF_{i:02}.csv",
                    modified_time=1000-i)
            for i in range(40)
        ]

        def execute(dialog):
            groups = dialog.findChild(QListWidget, "drr_source_group_list")
            files = dialog.findChild(QListWidget, "drr_source_file_list")
            self.assertEqual(groups.count(), 40)
            groups.setCurrentRow(39)
            self.assertEqual(files.item(0).data(Qt.UserRole), "sample_REF_39.csv")
            search = dialog.findChild(QLineEdit)
            search.setText("sample_REF_39")
            QTest.qWait(250)
            self.assertEqual(groups.count(), 1)
            search.clear()
            QTest.qWait(250)
            self.assertEqual(groups.count(), 40)
            self.window.drr_available_sources.append(replace(
                self.window.drr_available_sources[-1], group_key="older-session",
                source="older_REF.csv", filename="older_REF.csv", modified_time=1,
            ))
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertEqual(groups.count(), 41)
            groups.setCurrentRow(40)
            self.assertEqual(files.item(0).data(Qt.UserRole), "older_REF.csv")
            return QDialog.Rejected

        self.open_picker(execute)

    def test_unchanged_completion_keeps_file_selection_and_rows(self):
        def execute(dialog):
            files = dialog.findChild(QListWidget, "drr_source_file_list")
            groups = dialog.findChild(QListWidget, "drr_source_group_list")
            self.assertEqual(files.count(), 2)
            files.setCurrentRow(1)
            selected_path = files.currentItem().data(Qt.UserRole)
            resets = []
            groups.model().modelReset.connect(lambda: resets.append("groups"))
            files.model().modelReset.connect(lambda: resets.append("files"))
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertEqual(resets, [])
            self.assertIsNotNone(files.currentItem())
            self.assertEqual(files.currentItem().data(Qt.UserRole), selected_path)
            return QDialog.Rejected

        self.open_picker(execute)

    def test_preloaded_catalog_first_paint_is_stable_before_refresh(self):
        self.window.drr_available_sources = [
            replace(
                self.window.drr_available_sources[0], group_key=f"session-{i}",
                filename=f"sample_REF_760nmc_{'measurement_' * (i + 1)}.csv",
                source=f"sample_REF_760nmc_{'measurement_' * (i + 1)}.csv",
            )
            for i in range(25)
        ]
        self.window.show()
        self.app.processEvents()
        painted = []
        original_paint = WrappedFilenameDelegate.paint
        original_exec = QDialog.exec
        def paint(delegate, painter, option, index):
            if delegate.parent().objectName() == "drr_source_group_list":
                opt = QStyleOptionViewItem(option)
                delegate.initStyleOption(opt, index)
                needed = delegate._layout_text(
                    delegate._normalize_text(opt.text), opt.font,
                    option.rect.width() - 2 * delegate.HORIZONTAL_PADDING,
                ).boundingRect().height() + 2 * delegate.VERTICAL_PADDING
                painted.append((index.row(), option.rect.height(), needed))
            original_paint(delegate, painter, option, index)
        def execute(dialog):
            # Let the modal loop paint naturally. No grab/visualItemRect call
            # before first paint: those can force layout and mask stale rows.
            QTimer.singleShot(120, lambda: self.window.drr_catalog_refresh_finished.emit(
                self.window.current_folder, True))
            QTimer.singleShot(220, dialog.update)
            QTimer.singleShot(350, dialog.reject)
            return original_exec(dialog)
        with patch.object(WrappedFilenameDelegate, "paint", paint):
            self.open_picker(execute)
        self.assertTrue(painted)
        self.assertTrue(all(actual >= needed for _, actual, needed in painted), painted)
        for row in {row for row, _, _ in painted}:
            self.assertEqual(len({height for index, height, _ in painted if index == row}), 1)

    def test_changed_catalog_still_updates_filenames(self):
        def execute(dialog):
            files = dialog.findChild(QListWidget, "drr_source_file_list")
            self.assertEqual(files.count(), 2)
            self.window.drr_available_sources = self.window.drr_available_sources[:1]
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertEqual(files.count(), 1)
            self.assertEqual(files.item(0).data(Qt.UserRole),
                             self.window.drr_available_sources[0].source)
            return QDialog.Rejected

        self.open_picker(execute)

    def test_catalog_changes_preserve_scrolled_view_and_current_file(self):
        self.window.drr_available_sources = [
            replace(self.window.drr_available_sources[0], group_key=f"session-{i:02}",
                    source=f"sample_REF_{i:02}.csv", filename=f"sample_REF_{i:02}.csv",
                    modified_time=1000-i)
            for i in range(25)
        ]
        def execute(dialog):
            dialog.show()
            self.app.processEvents()
            groups = dialog.findChild(QListWidget, "drr_source_group_list")
            current = groups.currentItem().data(Qt.UserRole)
            bar = groups.verticalScrollBar()
            bar.setValue(bar.maximum() // 2)
            anchor = next(groups.item(i) for i in range(groups.count())
                          if groups.visualItemRect(groups.item(i)).bottom() >= 0)
            anchor_key = anchor.data(Qt.UserRole)
            offset = groups.visualItemRect(anchor).top()
            resets = QSignalSpy(groups.model().modelReset)
            bar.setSliderDown(True)
            # A newly arriving file changes catalog order while the user has
            # scrolled away from the automatically selected first session.
            self.window.drr_available_sources.insert(0, replace(
                self.window.drr_available_sources[0], group_key="new-session",
                source="new_REF.csv", filename="new_REF.csv", modified_time=2000,
            ))
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.app.processEvents()
            self.assertEqual(groups.currentItem().data(Qt.UserRole), current)
            new_anchor = next(groups.item(i) for i in range(groups.count())
                              if groups.item(i).data(Qt.UserRole) == anchor_key)
            self.assertEqual(groups.visualItemRect(new_anchor).top(), offset)
            self.assertGreater(bar.value(), 0)
            self.assertEqual(resets.count(), 0)
            self.assertTrue(bar.isSliderDown())
            bar.setSliderDown(False)
            dialog.reject()
            return QDialog.Rejected
        self.open_picker(execute)

    def test_processing_status_is_first_line_even_for_long_filename(self):
        self.window.settings.setValue("drr/source_unprocessed_only", False)
        def execute(dialog):
            groups = dialog.findChild(QListWidget, "drr_source_group_list")
            self.assertTrue(groups.item(0).text().startswith("UNPROCESSED"))
            self.window.drr_available_sources[0] = replace(self.window.drr_available_sources[0], processed=True)
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertTrue(groups.item(0).text().startswith("PARTIAL"))
            self.window.drr_available_sources = [replace(s, processed=True)
                                                  for s in self.window.drr_available_sources]
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertTrue(groups.item(0).text().startswith("PROCESSED"))
            return QDialog.Rejected
        self.open_picker(execute)

    def test_new_group_keeps_selected_session_beyond_recent_cutoff(self):
        self.window.drr_available_sources = [
            replace(self.window.drr_available_sources[0], group_key=f"session-{i:02}",
                    source=f"sample_REF_{i:02}.csv", filename=f"sample_REF_{i:02}.csv",
                    modified_time=1000-i)
            for i in range(25)
        ]
        def execute(dialog):
            groups = dialog.findChild(QListWidget, "drr_source_group_list")
            files = dialog.findChild(QListWidget, "drr_source_file_list")
            groups.setCurrentRow(24)
            current = groups.currentItem()
            also_selected = groups.item(23)
            also_selected.setSelected(True)
            files.setCurrentRow(0)
            current_file = files.currentItem()
            self.window.drr_available_sources.insert(0, replace(
                self.window.drr_available_sources[0], group_key="new-session",
                source="new_REF.csv", filename="new_REF.csv", modified_time=2000,
            ))
            self.window.drr_available_sources.insert(0, replace(
                self.window.drr_available_sources[0], group_key="new-session-2",
                source="new_2_REF.csv", filename="new_2_REF.csv", modified_time=2001,
            ))
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertIs(groups.currentItem(), current)
            self.assertIs(files.currentItem(), current_file)
            self.assertIn(also_selected, groups.selectedItems())
            return QDialog.Rejected
        self.open_picker(execute)

    def test_chosen_files_keep_processing_status_after_selection_update(self):
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            self.window.current_folder = folder
            name = "sample_REF.csv"
            Path(folder, name).write_text("Vbg,740,760\n0,1,2\n")
            self.window.drr_available_sources = [replace(
                self.window.drr_available_sources[0], source=name, filename=name,
                processed=True,
            )]
            self.window.settings.setValue("drr/source_unprocessed_only", False)
            def execute(dialog):
                chosen = dialog.findChild(QListWidget, "drr_source_chosen_list")
                self.assertTrue(chosen.item(0).text().startswith("PROCESSED"))
                return QDialog.Rejected
            with patch.object(QDialog, "exec", execute):
                self.window.drr_controller._open_drr_source_dialog(
                    title="Choose", selected=[name], baseline_mode=False,
                )

    def test_baseline_unchanged_completion_preserves_selection(self):
        self.window.drr_available_sources = [
            replace(source, spectral_grid=(740.0, 760.0, 780.0), grid_complete=True)
            for source in self.window.drr_available_sources
        ]
        self.window.drr_selected_files = [self.window.drr_available_sources[0].source]
        def execute(dialog):
            files = dialog.findChild(QListWidget, "drr_source_file_list")
            self.assertGreater(files.count(), 0)
            files.setCurrentRow(0)
            selected = files.currentItem()
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertIs(files.currentItem(), selected)
            return QDialog.Rejected
        self.open_picker(execute, baseline_mode=True)

    def test_unchanged_filtered_catalog_stops_showing_loading(self):
        self.window._drr_refresh_running = True
        self.window.drr_available_sources = [
            replace(source, is_background=True)
            for source in self.window.drr_available_sources
        ]

        def execute(dialog):
            hint = dialog.findChild(QLabel, "drrSourceEmptyHint")
            self.assertEqual(hint.text(), "Loading DRR catalog…")
            self.window.drr_catalog_refresh_finished.emit(self.window.current_folder, True)
            self.assertNotIn("Loading", hint.text())
            return QDialog.Rejected

        self.open_picker(execute)


if __name__ == "__main__":
    unittest.main()
