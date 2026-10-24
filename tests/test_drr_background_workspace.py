"""Current-data selection, private edits, saved snapshots and explicit Use."""
import json
import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool
from PySide6.QtWidgets import QApplication, QDialog

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_composite_background import (CompositeBackground, assign_region, compose_background,
    load_composite_source, read_composite_background, save_composite_background)
from core.drr_pn_pairing import DrrPnPair
from core.drr_pn_store import DrrPnStore
from ui_qt.drr_background_workspace import DrrBackgroundWorkspace, validate_editor_selection
from ui_qt.drr_composite_background_dialog import _initial_document
from ui_qt.theme import install_theme


class BackgroundWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.dialogs = []
        self.energy = np.linspace(1.5, 2., 81)
        for name, values in (("p.csv", (20., 40.)), ("n.csv", (50., 70.)),
                             ("sample.csv", (100., 110.)), ("repeat.csv", (105., 115.))):
            self.write(name, values)
        self.store = DrrPnStore(self.root)
        self.pair = DrrPnPair(("p.csv",), ("n.csv",))

    def write(self, name, values):
        header = "Vbg,Vtg," + ",".join(f"{v:.17g}" for v in 1240 / self.energy[::-1])
        rows = [f"{i},0," + ",".join([str(v)] * len(self.energy)) for i, v in enumerate(values)]
        (self.root / name).write_text("\n".join([header, *rows]) + "\n")

    def document(self):
        return CompositeBackground(tuple(load_composite_source(self.root, (name,), key=key)
            for name, key in (("p.csv", "p"), ("n.csv", "n"))), "p")

    def save_version(self, associate=False, p_which="all", n_which="all"):
        document = self.document()
        report = dict(algorithm_version="reviewed_fixture", scientifically_validated=True)
        document = replace(document, sources=(replace(document.sources[0], generation_json=json.dumps(report)),
                                             document.sources[1]))
        path = save_composite_background(self.root / "fixture_reference.csv", document)
        version = self.store.import_saved(path, self.pair, algorithm_version="reviewed_fixture", settings={},
                                         p_which=p_which, n_which=n_which)
        self.store.mark_used(version.version_id, measurements=("sample.csv",) if associate else ())
        return version

    def workspace(self, **kwargs):
        args = dict(folder=self.root, measurements=("sample.csv", "repeat.csv"),
            select_files=lambda current, title: current, pool=self.pool)
        args.update(kwargs)
        dialog = DrrBackgroundWorkspace(**args)
        self.dialogs.append(dialog)
        self.wait(dialog)
        return dialog

    def wait(self, dialog):
        self.app.processEvents()
        deadline = time.monotonic() + 15
        while (dialog.pn._worker is not None or dialog.editor._worker is not None) and time.monotonic() < deadline:
            self.app.processEvents()
            # Let Python QRunnable work acquire the GIL between Qt event pumps.
            time.sleep(0.005)
            self.app.processEvents()
        self.assertIsNone(dialog.pn._worker, dialog.pn.status.text())
        self.assertIsNone(dialog.editor._worker, dialog.editor.status.text())

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.close()
        self.pool.waitForDone(10000)
        self.app.processEvents()
        for dialog in self.dialogs:
            dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.temp.cleanup()

    def test_new_group_does_not_autoselect_pair_history_but_can_explicitly_reuse(self):
        version = self.save_version()
        dialog = self.workspace()
        self.assertEqual(dialog.pn.generation_mode.currentData(), "reconstruction")
        dialog.pn._set_pair(self.pair)
        dialog.pn._refresh_versions()
        self.wait(dialog)
        self.assertIsNone(dialog.pn._current)
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.pn.history.setCurrentIndex(dialog.pn.history.findData(version.version_id))
        self.wait(dialog)
        self.assertEqual(dialog.pn._current.version_id, version.version_id)
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIsNone(self.store.find_applied(("sample.csv", "repeat.csv")))

    def test_unchanged_exact_association_restores_but_new_selection_or_changed_bytes_does_not(self):
        version = self.save_version(associate=True, p_which="first", n_which="last")
        dialog = self.workspace(measurements=("sample.csv",))
        self.assertEqual(dialog.pn._current.version_id, version.version_id)
        self.assertEqual(dialog.pn.generation_mode.currentData(), "saved")
        self.assertEqual(dialog.pn.frames["P"].currentData(), "first")
        self.assertEqual(dialog.pn.frames["N"].currentData(), "last")
        self.assertFalse(dialog.pn._closed)
        other = self.workspace(measurements=("repeat.csv",))
        self.assertIsNone(other.pn._current)
        self.write("sample.csv", (111., 121.))
        changed = self.workspace(measurements=("sample.csv",))
        self.assertIsNone(changed.pn._current)

    def test_two_embedded_editors_keep_recipe_private_until_explicit_use(self):
        dialog = self.workspace(initial_files=("p.csv",), initial_page="edit", kept_background=True)
        self.assertEqual(dialog.tabs.count(), 2)
        self.assertFalse(dialog.pn.isWindow())
        self.assertFalse(dialog.editor.isWindow())
        self.assertTrue(dialog.pn.use_btn.isHidden())
        self.assertTrue(dialog.editor.apply_button.isHidden())
        self.assertIn("not a verified association", dialog.context_label.text())
        self.assertIsNone(dialog.selection)
        self.assertTrue(dialog.apply_btn.isEnabled())
        dialog._apply()
        self.wait(dialog)
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.selection.files, ("p.csv",))

    def test_cancel_during_editor_read_discards_result(self):
        dialog = self.workspace(initial_page="edit")
        release, started = threading.Event(), threading.Event()
        document = self.document()
        def read():
            started.set()
            release.wait(5)
            return document, "p"
        dialog.editor._start_job("add", read, "Reading")
        self.assertTrue(started.wait(2))
        token = dialog.editor._worker.cancellation
        self.assertFalse(dialog.tabs.tabBar().isEnabled())
        dialog.reject()
        release.set()
        self.wait(dialog)
        self.assertTrue(token.cancelled)
        self.assertIsNone(dialog.editor.document)
        self.assertIsNone(dialog.selection)

    def test_file_group_and_interval_recipe_matches_preview_exactly(self):
        region = DrrBackgroundRegion(1.65, 1.85, ("n.csv",), "first")
        document = _initial_document(self.root, ("p.csv", "n.csv"), "all", (region,))
        selection = validate_editor_selection(str(self.root), ("sample.csv",), document)
        self.assertEqual(selection.files, ("p.csv", "n.csv"))
        self.assertEqual(selection.which, "all")
        self.assertEqual(selection.regions, (region,))
        energy, values, bounds = compose_background(document)
        np.testing.assert_array_equal(values[(energy >= 1.65) & (energy < 1.85)], 50.)
        self.assertEqual(bounds, (1.65, 1.85))

    def test_changed_original_cannot_silently_replace_preview(self):
        document = self.document()
        self.write("p.csv", (21., 41.))
        with self.assertRaisesRegex(ValueError, "changed after preview"):
            validate_editor_selection(str(self.root), ("sample.csv",), document)

    def test_save_then_cancel_keeps_file_without_applying_and_reopen_uses_frozen_snapshot(self):
        dialog = self.workspace(initial_files=("p.csv",), initial_page="edit")
        path = self.root / "saved.csv"
        with patch("ui_qt.drr_composite_background_dialog.QFileDialog.getSaveFileName", return_value=(str(path), "CSV")):
            dialog.editor.save_as()
            self.wait(dialog)
        self.assertFalse(dialog._closed)
        self.assertIsNone(dialog.selection)
        self.assertTrue(path.exists())
        dialog.reject()
        (self.root / "p.csv").unlink()
        reopened = self.workspace(initial_files=(str(path),), initial_page="edit")
        self.assertTrue(reopened.apply_btn.isEnabled())
        reopened._apply()
        self.wait(reopened)
        self.assertEqual(reopened.selection.files, ("saved.csv",))

    def test_pn_copy_has_distinct_manual_identity_and_preserves_parent_report(self):
        version = self.save_version(associate=True)
        dialog = self.workspace(measurements=("sample.csv",))
        original = dialog.pn._payload["document"]
        dialog._edit_selected()
        source = next(s for s in dialog.editor.document.sources if s.key == original.default_source)
        report = json.loads(source.generation_json)
        self.assertEqual(report["algorithm_version"], "manual_pn_derived_v1")
        self.assertEqual(report["derived_from_pn_version"]["version_id"], version.version_id)
        self.assertEqual(report["parent_generation_report"]["algorithm_version"], "reviewed_fixture")
        self.assertNotIn("region_applicability", report)
        self.assertFalse(report["scientifically_validated"])
        self.assertFalse(dialog.apply_btn.isEnabled())
        edited = assign_region(dialog.editor.document, 1.65, 1.85, "n")
        dialog.editor._commit(edited)
        before = version.csv_path.read_bytes()
        with patch("ui_qt.drr_composite_background_dialog.QFileDialog.getSaveFileName", return_value=(str(version.csv_path), "CSV")):
            dialog.editor.save_as()
        self.assertIsNone(dialog.editor._worker)
        self.assertEqual(version.csv_path.read_bytes(), before)
        self.assertIn("history is preserved", dialog.editor.status.text())
        path = self.root / "derived.csv"
        with patch("ui_qt.drr_composite_background_dialog.QFileDialog.getSaveFileName", return_value=(str(path), "CSV")):
            dialog.editor.save_as()
            self.wait(dialog)
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertEqual(read_composite_background(path).regions, edited.regions)
        parent = read_composite_background(version.csv_path)
        self.assertEqual(parent.regions, ())
        self.assertNotIn("derived_from_pn_version", json.loads(parent.sources[0].generation_json))
        dialog.tabs.setCurrentWidget(dialog.pn_page)
        with patch("ui_qt.drr_pn_background_dialog.QFileDialog.getOpenFileName", return_value=(str(path), "CSV")):
            dialog.pn._import()
            self.wait(dialog)
        self.assertEqual(dialog.pn._current.algorithm_version, "manual_pn_derived_v1")
        self.assertTrue(dialog.pn._current.is_draft)

    def test_opening_pn_snapshot_then_editing_also_records_manual_identity(self):
        document = self.document()
        original = replace(document.sources[0], generation_json=json.dumps(dict(
            algorithm_version="pn_raw_conditioned_bounded_fixture", scientifically_validated=False,
            region_applicability={"scope": "parent only"})))
        parent = assign_region(replace(document, sources=(original, document.sources[1])), 1.65, 1.85, "n")
        path = save_composite_background(self.root / "raw_parent.csv", parent)
        dialog = self.workspace(initial_files=(str(path),), initial_page="edit")
        self.assertEqual(json.loads(dialog.editor.document.sources[0].generation_json)["algorithm_version"],
                         "pn_raw_conditioned_bounded_fixture")
        dialog.editor.source_table.setCurrentIndex(dialog.editor.source_model.index(1, 0))
        dialog.editor.frame_combo.setCurrentIndex(dialog.editor.frame_combo.findData("first"))
        self.wait(dialog)
        report = json.loads(dialog.editor.document.sources[0].generation_json)
        self.assertEqual(report["algorithm_version"], "manual_pn_derived_v1")
        self.assertIn("region_applicability", report["parent_generation_report"])
        self.assertNotIn("region_applicability", report)
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.editor._commit(replace(dialog.editor.document, default_source="n"))
        raw_default = json.loads(dialog.editor.document.sources[1].generation_json)
        self.assertEqual(raw_default["algorithm_version"], "manual_pn_derived_v1")
        self.assertFalse(raw_default["scientifically_validated"])

    def test_main_actions_remain_inside_small_window_with_scrollable_editors(self):
        dialog = self.workspace(initial_files=("p.csv",), initial_page="edit")
        dialog.resize(850, 600)
        dialog.show()
        self.app.processEvents()
        point = dialog.apply_btn.mapTo(dialog, dialog.apply_btn.rect().bottomRight())
        self.assertLessEqual(point.y(), dialog.height())
        self.assertLessEqual(point.x(), dialog.width())
        self.assertTrue(dialog.editor_page.widgetResizable())
        dialog.editor.reject()
        self.assertTrue(dialog._closed)
        self.assertIsNone(dialog.selection)


if __name__ == "__main__":
    unittest.main()
