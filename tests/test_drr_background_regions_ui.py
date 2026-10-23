from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import matplotlib
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QDialog

from core.drr_background_regions import DrrBackgroundRegion
from ui_qt.drr_background_regions_dialog import DrrBackgroundRegionsDialog
from ui_qt.theme import install_theme


class DrrBackgroundRegionsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        cls.test_font = -1
        # Windows' offscreen platform can expose no system fonts. Register a
        # portable test font so layout checks exercise text instead of tofu.
        if not QFontDatabase.families():
            font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
            cls.test_font = QFontDatabase.addApplicationFont(str(font_path))
            cls.app.setFont(QFont("DejaVu Sans", 9))

    @classmethod
    def tearDownClass(cls):
        if cls.test_font >= 0:
            cls.app.setFont(cls.original_font)
            QFontDatabase.removeApplicationFont(cls.test_font)

    def setUp(self):
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.dialogs = []
        self.energy = np.linspace(1., 2., 11)

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.close()
        self.pool.waitForDone(5000)
        self.app.processEvents()
        for dialog in self.dialogs:
            dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()

    def dialog(self, **kwargs):
        args = dict(energy=self.energy, default_files=("default.csv",),
                    select_files=lambda selected: ["selected.csv"],
                    preview=lambda regions: (self.energy, np.ones_like(self.energy)), pool=self.pool)
        args.update(kwargs)
        dialog = DrrBackgroundRegionsDialog(**args)
        self.dialogs.append(dialog)
        return dialog

    def wait_preview(self, dialog):
        deadline = time.monotonic() + 5
        while dialog._worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertIsNone(dialog._worker, "preview did not finish")

    def test_private_draft_file_selection_and_cancel_preserve_original_recipe(self):
        region = DrrBackgroundRegion(1.2, 1.5, ("original.csv",), "first")
        selected = []
        def choose(files):
            selected.append(files)
            files.append("second.csv")
            return files
        dialog = self.dialog(regions=(region,), select_files=choose)
        applied = QSignalSpy(dialog.applied)
        dialog.table.setCurrentIndex(dialog.model.index(0, 2))
        dialog.choose_files()
        dialog.model.setData(dialog.model.index(0, 3), "all")
        self.assertEqual(dialog.regions()[0].baseline_files, ("original.csv", "second.csv"))
        self.assertEqual(region.baseline_files, ("original.csv",))
        self.assertEqual(region.baseline_which, "first")
        dialog.reject()
        self.assertEqual(dialog.result(), QDialog.Rejected)
        self.assertEqual(applied.count(), 0)

    def test_add_edit_and_choose_files_through_editor_controls(self):
        dialog = self.dialog()
        dialog.show()
        self.app.processEvents()
        dialog.add_button.click()
        editor = self.app.focusWidget()
        QTest.keyClicks(editor, "1.25")
        QTest.keyClick(editor, Qt.Key_Tab)
        dialog.files_button.click()
        self.assertEqual(dialog.regions(), (DrrBackgroundRegion(1.25, 2., ("selected.csv",)),))
        dialog.preview_background()
        self.wait_preview(dialog)
        self.assertTrue(dialog.apply_button.isEnabled())

    def test_preview_runs_off_gui_thread_and_apply_returns_validated_recipe(self):
        worker_threads = []
        region = DrrBackgroundRegion(1.2, 1.5, ("local.csv",))
        def preview(regions):
            worker_threads.append(threading.get_ident())
            background = np.full(self.energy.shape, 2.)
            background[(self.energy >= 1.2) & (self.energy < 1.5)] = 7.
            return self.energy, background
        dialog = self.dialog(regions=(region,), preview=preview)
        applied = QSignalSpy(dialog.applied)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.accept()
        self.assertEqual(applied.count(), 0)
        dialog.preview_background()
        self.assertFalse(dialog.preview_button.isEnabled())
        self.wait_preview(dialog)
        self.assertNotEqual(worker_threads, [threading.get_ident()])
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertEqual(len(dialog.ax.lines), 3)
        np.testing.assert_array_equal(dialog.ax.lines[0].get_ydata(), dialog._background)
        dialog.apply_button.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(applied.count(), 1)
        self.assertEqual(dialog.regions(), (region,))

    def test_changed_recipe_cannot_reuse_previous_preview(self):
        dialog = self.dialog(regions=(DrrBackgroundRegion(1.2, 1.5, ("a.csv",)),))
        dialog.preview_background()
        self.wait_preview(dialog)
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.model.setData(dialog.model.index(0, 3), "all")
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertIsNone(dialog._background)
        self.assertIn("not been previewed", dialog.status.text())
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Rejected)

    def test_overlap_rejected_before_loading_files(self):
        calls = []
        dialog = self.dialog(regions=(DrrBackgroundRegion(1., 1.5, ("a.csv",)),
                                      DrrBackgroundRegion(1.5, 2., ("b.csv",))),
                             preview=lambda regions: calls.append(regions))
        dialog.model.setData(dialog.model.index(1, 0), 1.4)
        dialog.preview_background()
        self.assertEqual(calls, [])
        self.assertIn("overlap", dialog.status.text())
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertTrue(dialog.table.property("fluentInvalid"))

    def test_unknown_axis_loaded_in_worker_and_limits_checked(self):
        dialog = self.dialog(energy=None, regions=(DrrBackgroundRegion(1.2, 1.5, ("a.csv",)),))
        dialog.preview_background()
        self.wait_preview(dialog)
        np.testing.assert_array_equal(dialog.energy, self.energy)
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.model.setData(dialog.model.index(0, 0), .5)
        dialog.preview_background()
        self.assertIn("within", dialog.status.text())
        self.assertFalse(dialog.apply_button.isEnabled())

    def test_failed_preview_never_allows_apply(self):
        def preview(regions):
            raise ValueError("Background spectral grid differs from measurement.")
        dialog = self.dialog(preview=preview)
        dialog.preview_background()
        self.wait_preview(dialog)
        self.assertIn("spectral grid differs", dialog.status.text())
        self.assertFalse(dialog.apply_button.isEnabled())
        self.assertTrue(dialog.preview_button.isEnabled())

    def test_invalid_background_samples_are_reported_without_changing_loader_policy(self):
        background = np.ones_like(self.energy)
        background[3], background[5] = 0., np.nan
        dialog = self.dialog(preview=lambda regions: (self.energy, background))
        dialog.preview_background()
        self.wait_preview(dialog)
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertIn("2 zero/non-finite", dialog.status.text())
        np.testing.assert_array_equal(dialog._background, background)

    def test_cancel_while_loading_ignores_late_result(self):
        release = threading.Event()
        started = threading.Event()
        def preview(regions):
            started.set()
            release.wait(3)
            return self.energy, np.ones_like(self.energy)
        dialog = self.dialog(preview=preview)
        dialog.preview_background()
        self.assertTrue(started.wait(2))
        cancellation = dialog._worker.cancellation
        dialog.reject()
        release.set()
        self.wait_preview(dialog)
        self.assertTrue(cancellation.cancelled)
        self.assertIsNone(dialog._validated_regions)
        self.assertEqual(dialog.result(), QDialog.Rejected)

    def test_zero_intervals_previews_default_and_can_remove_existing_override(self):
        dialog = self.dialog(regions=(DrrBackgroundRegion(1.2, 1.5, ("a.csv",)),))
        dialog.table.setCurrentIndex(dialog.model.index(0, 0))
        dialog.remove_region()
        self.assertEqual(dialog.regions(), ())
        self.assertFalse(dialog.remove_button.isEnabled())
        dialog.preview_background()
        self.wait_preview(dialog)
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.apply_button.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)

    def test_light_dark_minimum_layout_and_escape_keyboard(self):
        manager = install_theme(self.app, mode="light")
        self.addCleanup(manager.set_mode, "light")
        dialog = self.dialog(regions=(DrrBackgroundRegion(1.2, 1.5, ("a.csv",)),))
        dialog.resize(dialog.minimumSize())
        dialog.show()
        self.app.processEvents()
        for mode in ("light", "dark"):
            manager.set_mode(mode)
            self.app.processEvents()
            dialog.canvas.draw()
            self.assertEqual(dialog.canvas._resolved_theme.name, mode)
            self.assertEqual(dialog.styleSheet(), "")
            for widget in (dialog.table, dialog.files_button, dialog.preview_button,
                           dialog.apply_button, dialog.button_box):
                position = widget.mapTo(dialog, widget.rect().topLeft())
                self.assertGreaterEqual(position.x(), 0)
                self.assertGreaterEqual(position.y(), 0)
                self.assertLessEqual(position.x() + widget.width(), dialog.width())
                self.assertLessEqual(position.y() + widget.height(), dialog.height())
        dialog.table.setFocus()
        QTest.keyClick(dialog.table, Qt.Key_Tab)
        self.assertIs(self.app.focusWidget(), dialog.add_button)
        QTest.keyClick(dialog, Qt.Key_Escape)
        self.assertEqual(dialog.result(), QDialog.Rejected)
        self.assertFalse(dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
