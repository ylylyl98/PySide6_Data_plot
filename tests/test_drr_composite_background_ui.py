from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
import numpy as np
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_composite_background import (
    CompositeBackground, CompositeRegion, CompositeSource, compose_background,
    load_composite_source, read_composite_background, save_composite_background,
)
from ui_qt.drr_composite_background_dialog import DrrCompositeBackgroundDialog
from ui_qt.theme import install_theme


class CompositeBackgroundDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        cls.test_font = -1
        if not QFontDatabase.families():
            cls.test_font = QFontDatabase.addApplicationFont(str(
                Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"))
            cls.app.setFont(QFont("DejaVu Sans", 9))

    @classmethod
    def tearDownClass(cls):
        if cls.test_font >= 0:
            cls.app.setFont(cls.original_font)
            QFontDatabase.removeApplicationFont(cls.test_font)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.dialogs = []
        self.energy = np.linspace(1., 2., 21)
        self.write_csv("p.csv", (10., 20.))
        self.write_csv("n.csv", (30., 40.))

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.close()
        self.pool.waitForDone(5000)
        self.app.processEvents()
        for dialog in self.dialogs:
            dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()
        self.temporary.cleanup()

    def write_csv(self, name, values, energy=None):
        energy = self.energy if energy is None else energy
        header = "Vbg,Vtg," + ",".join(f"{value:.17g}" for value in 1240. / energy[::-1])
        rows = [f"{i},0," + ",".join([str(value)] * len(energy)) for i, value in enumerate(values)]
        (self.root / name).write_text("\n".join([header, *rows]) + "\n")

    def source(self, key, intensity, *, files=None):
        return CompositeSource(key, key + ".csv", files or (str(self.root / (key + ".csv")),), "last",
                               self.energy, tuple(map(str, self.energy)), np.full(21, intensity))

    def document(self):
        return CompositeBackground((self.source("p", 20.), self.source("n", 40.)), "p")

    def dialog(self, **kwargs):
        args = dict(folder=str(self.root), select_files=lambda selected: ["p.csv", "n.csv"], pool=self.pool)
        args.update(kwargs)
        dialog = DrrCompositeBackgroundDialog(**args)
        self.dialogs.append(dialog)
        return dialog

    def wait_job(self, dialog):
        self.app.processEvents()
        deadline = time.monotonic() + 7
        while dialog._worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertIsNone(dialog._worker, dialog.status.text())

    def select_source(self, dialog, row):
        dialog.source_table.setCurrentIndex(dialog.source_model.index(row, 0))

    def test_empty_editor_does_not_require_measurement_and_add_reads_off_gui_thread(self):
        dialog = self.dialog()
        self.assertIsNone(dialog.document)
        self.assertFalse(dialog.save_button.isEnabled())
        threads = []
        def read(*args, **kwargs):
            threads.append(threading.get_ident())
            return load_composite_source(*args, **kwargs)
        with patch("ui_qt.drr_composite_background_dialog.load_composite_source", read):
            dialog.add_sources()
            self.wait_job(dialog)
        self.assertEqual(len(dialog.document.sources), 2)
        self.assertTrue(all(thread != threading.get_ident() for thread in threads))
        np.testing.assert_allclose(compose_background(dialog.document)[1], 20.)
        self.assertTrue(dialog.save_button.isEnabled())

    def test_group_average_is_explicit_and_frame_reload_changes_only_selected_snapshot(self):
        dialog = self.dialog()
        dialog.group_check.setChecked(True)
        dialog.add_sources()
        self.wait_job(dialog)
        self.assertEqual(len(dialog.document.sources), 1)
        np.testing.assert_allclose(compose_background(dialog.document)[1], 30.)
        original = dialog.document
        dialog.frame_combo.setCurrentIndex(dialog.frame_combo.findData("first"))
        self.wait_job(dialog)
        np.testing.assert_allclose(compose_background(dialog.document)[1], 20.)
        self.assertEqual(dialog.document.sources[0].which, "first")
        dialog.undo()
        self.assertIs(dialog.document, original)

    def test_real_span_drag_selects_new_interval_and_use_assigns_selected_source(self):
        original = self.document()
        dialog = self.dialog(initial_document=original)
        dialog.show()
        self.app.processEvents()
        self.select_source(dialog, 1)
        dialog.canvas.draw()
        y = sum(dialog.ax.get_ylim()) / 2
        start = dialog.ax.transData.transform((1.2, y))
        finish = dialog.ax.transData.transform((1.6, y))
        for event, point in (("button_press_event", start), ("motion_notify_event", finish),
                             ("button_release_event", finish)):
            MouseEvent(event, dialog.canvas, *point, button=1)._process()
        self.assertAlmostEqual(float(dialog.lower_edit.text()), 1.2, places=6)
        self.assertAlmostEqual(float(dialog.upper_edit.text()), 1.6, places=6)
        self.assertIs(dialog.document, original)
        dialog.use_button.click()
        energy, background, boundaries = compose_background(dialog.document)
        region = dialog.document.regions[0]
        mask = (energy >= region.xmin_ev) & (energy < region.xmax_ev)
        np.testing.assert_allclose(background[mask], 40.)
        np.testing.assert_allclose(background[~mask], 20.)
        self.assertEqual(len(boundaries), 2)

    def test_edit_existing_interval_removes_old_extent_and_undo_restores_snapshot(self):
        base = self.document()
        original = CompositeBackground(base.sources, "p", (CompositeRegion(1.2, 1.6, "n"),))
        dialog = self.dialog(initial_document=original)
        dialog.region_table.setCurrentIndex(dialog.region_model.index(0, 0))
        dialog._span_selected(1.4, 1.8)
        dialog.use_source()
        self.assertEqual(dialog.document.regions, (CompositeRegion(1.4, 1.8, "n"),))
        _, background, _ = compose_background(dialog.document)
        self.assertEqual(background[5], 20.)
        dialog.undo()
        self.assertIs(dialog.document, original)
        self.assertEqual(dialog.region_model.rowCount(), 1)

    def test_selected_span_handle_drag_shrinks_assignment(self):
        base = self.document()
        original = CompositeBackground(base.sources, "p", (CompositeRegion(1.2, 1.7, "n"),))
        dialog = self.dialog(initial_document=original)
        dialog.show()
        self.app.processEvents()
        dialog.region_table.setCurrentIndex(dialog.region_model.index(0, 0))
        dialog.canvas.draw()
        y = sum(dialog.ax.get_ylim()) / 2
        start = dialog.ax.transData.transform((1.7, y))
        finish = dialog.ax.transData.transform((1.5, y))
        for event, point in (("button_press_event", start), ("motion_notify_event", finish),
                             ("button_release_event", finish)):
            MouseEvent(event, dialog.canvas, *point, button=1)._process()
        self.assertAlmostEqual(float(dialog.upper_edit.text()), 1.5, places=6)
        dialog.use_source()
        self.assertEqual(len(dialog.document.regions), 1)
        self.assertAlmostEqual(dialog.document.regions[0].xmax_ev, 1.5, places=6)
        self.assertEqual(compose_background(dialog.document)[1][13], 20.)

    def test_default_remove_interval_and_undo_keep_complete_document_history(self):
        base = self.document()
        original = CompositeBackground(base.sources, "p", (CompositeRegion(1.2, 1.6, "n"),))
        dialog = self.dialog(initial_document=original)
        dialog.region_table.setCurrentIndex(dialog.region_model.index(0, 0))
        dialog.remove_region()
        self.assertEqual(dialog.document.regions, ())
        dialog.undo()
        self.assertIs(dialog.document, original)
        dialog.default_combo.setCurrentIndex(dialog.default_combo.findData("n"))
        self.assertEqual(dialog.document.default_source, "n")
        np.testing.assert_allclose(compose_background(dialog.document)[1], 40.)
        dialog.undo()
        self.assertIs(dialog.document, original)

    def test_frame_reload_failure_keeps_snapshot_and_restores_frame_selection(self):
        original = self.document()
        dialog = self.dialog(initial_document=original)
        with patch("ui_qt.drr_composite_background_dialog.load_composite_source",
                   side_effect=FileNotFoundError("Original p.csv is unavailable")):
            dialog.frame_combo.setCurrentIndex(dialog.frame_combo.findData("first"))
            self.wait_job(dialog)
        self.assertIs(dialog.document, original)
        self.assertEqual(dialog.frame_combo.currentData(), "last")
        self.assertIn("unavailable", dialog.status.text())

    def test_mismatched_grid_add_is_atomic_and_keeps_existing_document(self):
        initial = load_composite_source(str(self.root), ("p.csv",), key="p")
        original = CompositeBackground((initial,), "p")
        self.write_csv("wrong.csv", (5., 6.), energy=np.linspace(1.01, 2.01, 21))
        dialog = self.dialog(initial_document=original, select_files=lambda files: ["n.csv", "wrong.csv"])
        dialog.add_sources()
        self.wait_job(dialog)
        self.assertIs(dialog.document, original)
        self.assertIn("spectral grid", dialog.status.text())
        self.assertEqual(len(dialog._undo), 0)

    def test_existing_range_recipe_initializes_equivalent_source_document(self):
        default = load_composite_source(str(self.root), ("p.csv",))
        lo, hi = default.energy[4], default.energy[12]
        region = DrrBackgroundRegion(lo, hi, ("n.csv",), "first")
        dialog = self.dialog(initial_files=("p.csv",), initial_regions=(region,))
        self.wait_job(dialog)
        self.assertEqual(len(dialog.document.sources), 2)
        energy, intensity, _ = compose_background(dialog.document)
        mask = (energy >= lo) & (energy < hi)
        np.testing.assert_allclose(intensity[mask], 30.)
        np.testing.assert_allclose(intensity[~mask], 20.)

    def test_save_then_reopen_without_original_files_preserves_editable_snapshots(self):
        original = self.document()
        dialog = self.dialog(initial_document=original)
        path = self.root / "combined.csv"
        with patch("ui_qt.drr_composite_background_dialog.QFileDialog.getSaveFileName",
                   return_value=(str(path), "CSV")):
            dialog.save_as()
            self.wait_job(dialog)
        self.assertFalse(dialog._closed)
        self.assertIsNone(dialog.applied_background)
        self.assertTrue(dialog.apply_button.isEnabled())
        self.assertEqual(Path(dialog.saved_path), path)
        (self.root / "p.csv").unlink()
        (self.root / "n.csv").unlink()
        reopened = self.dialog(initial_files=(str(path),))
        self.wait_job(reopened)
        self.assertEqual(len(reopened.document.sources), 2)
        self.select_source(reopened, 1)
        reopened._span_selected(1.2, 1.7)
        reopened.use_source()
        self.assertEqual(reopened.document.regions, (CompositeRegion(1.2, 1.7, "n"),))
        np.testing.assert_array_equal(read_composite_background(path).sources[1].intensity,
                                      original.sources[1].intensity)

    def test_open_composite_is_undoable_and_cancel_does_not_save(self):
        original = self.document()
        alternate = CompositeBackground(original.sources, "n")
        path = save_composite_background(self.root / "other.csv", alternate)
        dialog = self.dialog(initial_document=original)
        with patch("ui_qt.drr_composite_background_dialog.QFileDialog.getOpenFileName",
                   return_value=(str(path), "CSV")):
            dialog.open_composite()
            self.wait_job(dialog)
        self.assertEqual(dialog.document.default_source, "n")
        dialog.undo()
        self.assertIs(dialog.document, original)
        dialog.reject()
        self.assertEqual(dialog.saved_path, str(path))
        self.assertIsNone(dialog.applied_background)
        self.assertEqual(dialog.result(), QDialog.Rejected)

    def test_appended_csv_destination_requires_its_own_overwrite_confirmation(self):
        path = self.root / "existing.csv"
        path.write_text("preserve existing file")
        dialog = self.dialog(initial_document=self.document())
        with patch("ui_qt.drr_composite_background_dialog.QFileDialog.getSaveFileName",
                   return_value=(str(path.with_suffix("")), "CSV")), \
             patch("ui_qt.drr_composite_background_dialog.QMessageBox.question", return_value=QMessageBox.No) as confirm:
            dialog.save_as()
        confirm.assert_called_once()
        self.assertIsNone(dialog._worker)
        self.assertIsNone(dialog.saved_path)
        self.assertEqual(path.read_text(), "preserve existing file")

    def test_cancel_pending_source_read_ignores_late_result(self):
        release, started = threading.Event(), threading.Event()
        source = self.source("p", 20.)
        def read(*args, **kwargs):
            started.set()
            release.wait(4)
            return source
        dialog = self.dialog(select_files=lambda files: ["p.csv"])
        with patch("ui_qt.drr_composite_background_dialog.load_composite_source", read):
            dialog.add_sources()
            self.assertTrue(started.wait(2))
            cancellation = dialog._worker.cancellation
            dialog.reject()
            release.set()
            self.wait_job(dialog)
        self.assertTrue(cancellation.cancelled)
        self.assertIsNone(dialog.document)

    def test_save_finishes_before_window_can_be_cancelled(self):
        release, started = threading.Event(), threading.Event()
        destination = self.root / "saved.csv"
        def save(*args, **kwargs):
            started.set()
            release.wait(4)
            return destination
        dialog = self.dialog(initial_document=self.document())
        with patch("ui_qt.drr_composite_background_dialog.save_composite_background", save), \
             patch("ui_qt.drr_composite_background_dialog.QFileDialog.getSaveFileName",
                   return_value=(str(destination), "CSV")):
            dialog.save_as()
            self.assertTrue(started.wait(2))
            self.assertFalse(dialog.cancel_button.isEnabled())
            dialog.reject()
            self.assertFalse(dialog._closed)
            release.set()
            self.wait_job(dialog)
        self.assertFalse(dialog._closed)
        self.assertIsNone(dialog.applied_background)
        self.assertEqual(dialog.saved_path, str(destination))

    def test_light_dark_minimum_layout_and_keyboard_actions(self):
        manager = install_theme(self.app, mode="light")
        self.addCleanup(manager.set_mode, "light")
        dialog = self.dialog(initial_document=self.document())
        dialog.resize(dialog.minimumSize())
        dialog.show()
        self.app.processEvents()
        for mode in ("light", "dark"):
            manager.set_mode(mode)
            self.app.processEvents()
            dialog.canvas.draw()
            self.assertEqual(dialog.canvas._resolved_theme.name, mode)
            self.assertEqual(dialog.styleSheet(), "")
            self.assertTrue(all(line.get_alpha() == 1. for line in dialog.ax.lines[:2]))
            self.assertIn("p.csv", dialog.default_combo.toolTip())
            for widget in (dialog.source_table, dialog.default_combo, dialog.use_button,
                           dialog.lower_edit, dialog.upper_edit, dialog.save_button, dialog.cancel_button):
                point = widget.mapTo(dialog, widget.rect().topLeft())
                self.assertGreaterEqual(point.x(), 0)
                self.assertGreaterEqual(point.y(), 0)
                self.assertLessEqual(point.x() + widget.width(), dialog.width())
                self.assertLessEqual(point.y() + widget.height(), dialog.height())
        dialog.source_table.setFocus()
        QTest.keyClick(dialog.source_table, Qt.Key_Tab)
        self.assertIs(self.app.focusWidget(), dialog.frame_combo)
        QTest.keyClick(dialog, Qt.Key_Escape)
        self.assertFalse(dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
