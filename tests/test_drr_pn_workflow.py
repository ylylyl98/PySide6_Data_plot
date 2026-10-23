"""Explicit PN application is asynchronous and cannot replace a newer recipe."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QSpinBox

from ui_qt.drr_pn_workflow import apply_pn_version, cancel_pn_apply, edit_pn_background


class PnWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.jobs = []
        self.owner = SimpleNamespace(
            current_folder=str(self.root), drr_selected_files=["sample.csv"],
            drr_baseline_files_manual=["old.csv"], drr_baseline_files_found=[],
            _drr_background_regions=(), _drr_assignments=(), _drr_assignments_automatic=False,
            _is_closing=False, _show_error=MagicMock(), _status=MagicMock(),
            _start_load=MagicMock(), _queue_drr_catalog_refresh=MagicMock(),
            drr_available_sources=[], thread_pool=SimpleNamespace(start=self.jobs.append),
            drr_baseline_combo=QComboBox(), drr_baseline_combine_combo=QComboBox(),
            drr_pin_baseline_chk=QCheckBox(), drr_sg_window_spin=QSpinBox(),
            drr_sg_poly_spin=QSpinBox(),
            drr_controller=SimpleNamespace(
                _cancel_auto_external=MagicMock(), _update_drr_selection_labels=MagicMock(),
                _open_drr_source_dialog=MagicMock(return_value=["p.csv"]),
            ),
        )
        self.owner.drr_baseline_combo.addItems(["Self (last frame)", "External"])
        self.owner.drr_baseline_combo.setCurrentText("External")
        self.owner.drr_baseline_combine_combo.addItems([
            "First frame from each file, then average",
            "Last frame from each file, then average",
        ])
        self.owner.drr_sg_window_spin.setValue(21)
        self.owner.drr_sg_poly_spin.setValue(3)
        self.version = SimpleNamespace(
            version_id="v1", csv_path=self.root / "PN_saved.csv", fresh=True,
        )
        self.factory = MagicMock()
        self.dialog = self.factory.return_value
        self.dialog.exec.return_value = QDialog.Accepted
        self.dialog.selected_version = self.version
        self.dialog.allow_stale = False
        self.store = MagicMock()
        self.store.mark_used.return_value = self.version
        self.module_patch = patch.dict(sys.modules, {
            "ui_qt.drr_pn_background_dialog": SimpleNamespace(DrrPnBackgroundDialog=self.factory),
        })
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.store_patch = patch("ui_qt.drr_pn_workflow.DrrPnStore", return_value=self.store)
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        self.addCleanup(lambda: cancel_pn_apply(self.owner))

    def finish(self, index=0):
        self.jobs[index].run()
        self.app.processEvents()

    def test_selection_only_and_outer_cancel_have_no_application_side_effects(self):
        self.dialog.allow_stale = True
        self.assertIs(edit_pn_background(self.owner, selection_only=True), self.version)
        self.assertTrue(self.owner._drr_pn_selection_allow_stale)
        self.store.mark_used.assert_not_called()
        self.assertEqual(self.jobs, [])
        self.assertEqual(self.owner.drr_baseline_files_manual, ["old.csv"])
        self.owner._queue_drr_catalog_refresh.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_explicit_chooser_cancels_auto_matching_before_dialog_construction(self):
        def construct(*args, **kwargs):
            self.owner.drr_controller._cancel_auto_external.assert_called_once_with()
            self.assertEqual(self.owner.drr_baseline_files_manual, ["old.csv"])
            return self.dialog
        self.factory.side_effect = construct
        self.assertIs(edit_pn_background(self.owner, selection_only=True), self.version)
        self.store.mark_used.assert_not_called()
        self.assertEqual(self.jobs, [])

    def test_accepted_outer_picker_uses_same_worker_and_consent(self):
        version = edit_pn_background(self.owner, selection_only=True)
        apply_pn_version(self.owner, version, allow_stale=True)
        self.store.mark_used.assert_not_called()
        self.finish()
        self.store.mark_used.assert_called_once_with(
            "v1", measurements=("sample.csv",), allow_stale=True,
        )
        self.assertEqual(self.owner.drr_baseline_files_manual, ["PN_saved.csv"])

    def test_dialog_cancel_preserves_existing_recipe(self):
        self.dialog.exec.return_value = QDialog.Rejected
        self.assertIsNone(edit_pn_background(self.owner))
        self.store.mark_used.assert_not_called()
        self.assertEqual(self.jobs, [])
        self.assertEqual(self.owner.drr_baseline_files_manual, ["old.csv"])

    def test_worker_marks_then_applies_exactly_one_csv(self):
        self.assertIs(edit_pn_background(self.owner), self.version)
        self.store.mark_used.assert_not_called()  # No hashing on the GUI call.
        self.assertEqual(self.owner.drr_baseline_files_manual, ["old.csv"])
        self.finish()
        self.store.mark_used.assert_called_once_with(
            "v1", measurements=("sample.csv",), allow_stale=False,
        )
        self.assertEqual(self.owner.drr_baseline_files_manual, ["PN_saved.csv"])
        self.assertEqual(self.owner.drr_baseline_combine_combo.currentText(),
                         "Last frame from each file, then average")
        self.assertTrue(self.owner.drr_pin_baseline_chk.isChecked())
        self.owner._start_load.assert_called_once_with("DRR")
        self.assertIsNone(self.owner._drr_pn_apply_request)
        callback = self.factory.call_args.kwargs["select_files"]
        callback(["p.csv"], "Choose P")
        self.owner.drr_controller._open_drr_source_dialog.assert_called_once_with(
            title="Choose P", selected=["p.csv"], baseline_mode=True, independent_background=True,
        )
        self.assertTrue({"y_axis", "window_length", "polyorder", "auto_window"}.isdisjoint(
            self.factory.call_args.kwargs))

    def test_empty_measurement_selection_can_pin_without_loading(self):
        self.owner.drr_selected_files = []
        edit_pn_background(self.owner)
        self.finish()
        self.store.mark_used.assert_called_once_with("v1", measurements=(), allow_stale=False)
        self.assertEqual(self.owner.drr_baseline_files_manual, ["PN_saved.csv"])
        self.owner._start_load.assert_not_called()

    def test_changed_context_while_dialog_open_does_not_enqueue(self):
        def changed():
            self.owner.drr_selected_files = ["other.csv"]
            return QDialog.Accepted
        self.dialog.exec.side_effect = changed
        self.assertIsNone(edit_pn_background(self.owner))
        self.assertEqual(self.jobs, [])
        self.store.mark_used.assert_not_called()

    def test_changed_folder_discards_queued_application(self):
        edit_pn_background(self.owner)
        self.jobs[0].run()
        self.owner.current_folder = str(self.root / "other_experiment")
        self.app.processEvents()
        self.assertEqual(self.owner.drr_baseline_files_manual, ["old.csv"])
        self.owner._start_load.assert_not_called()

    def test_late_worker_cannot_apply_after_background_changed(self):
        edit_pn_background(self.owner)
        self.owner.drr_baseline_files_manual = ["new_manual.csv"]
        self.finish()
        self.assertEqual(self.owner.drr_baseline_files_manual, ["new_manual.csv"])
        self.owner._start_load.assert_not_called()
        self.owner._queue_drr_catalog_refresh.assert_not_called()

    def test_completed_result_cannot_apply_after_measurement_changed(self):
        edit_pn_background(self.owner)
        self.jobs[0].run()  # Completion is queued; user changes before delivery.
        self.owner.drr_selected_files = ["other.csv"]
        self.app.processEvents()
        self.assertEqual(self.owner.drr_baseline_files_manual, ["old.csv"])
        self.owner._start_load.assert_not_called()

    def test_changed_axis_while_dialog_open_keeps_background_selection_valid(self):
        self.owner._selected_y_axis_spec = MagicMock(return_value="TG")
        def changed():
            self.owner._selected_y_axis_spec.return_value = "BG"
            return QDialog.Accepted
        self.dialog.exec.side_effect = changed
        self.assertIs(edit_pn_background(self.owner), self.version)
        self.owner._selected_y_axis_spec.assert_not_called()
        self.finish()
        self.assertEqual(self.owner.drr_baseline_files_manual, ["PN_saved.csv"])

    def test_changed_sg_settings_do_not_discard_queued_application(self):
        edit_pn_background(self.owner)
        self.jobs[0].run()
        self.owner.drr_sg_window_spin.setValue(11)
        self.owner.drr_sg_poly_spin.setValue(2)
        self.app.processEvents()
        self.assertEqual(self.owner.drr_baseline_files_manual, ["PN_saved.csv"])
        self.owner._start_load.assert_called_once_with("DRR")

    def test_auto_window_change_does_not_discard_queued_application(self):
        self.owner.drr_sg_auto_chk = QCheckBox()
        self.owner.drr_sg_auto_chk.setChecked(True)
        edit_pn_background(self.owner)
        self.assertNotIn("auto_window", self.factory.call_args.kwargs)
        self.jobs[0].run()
        self.owner.drr_sg_auto_chk.setChecked(False)
        self.app.processEvents()
        self.assertEqual(self.owner.drr_baseline_files_manual, ["PN_saved.csv"])
        self.owner._start_load.assert_called_once_with("DRR")

    def test_explicit_cancellation_prevents_marking(self):
        edit_pn_background(self.owner)
        cancel_pn_apply(self.owner)
        self.finish()
        self.store.mark_used.assert_not_called()
        self.assertEqual(self.owner.drr_baseline_files_manual, ["old.csv"])

    def test_new_request_supersedes_old_worker_and_old_finished(self):
        edit_pn_background(self.owner)
        newer = SimpleNamespace(version_id="v2", csv_path=self.root / "PN_new.csv", fresh=True)
        self.dialog.selected_version = newer
        self.store.mark_used.return_value = newer
        edit_pn_background(self.owner)
        request = self.owner._drr_pn_apply_request
        self.finish(0)
        self.assertIs(self.owner._drr_pn_apply_request, request)
        self.store.mark_used.assert_not_called()
        self.finish(1)
        self.assertEqual(self.owner.drr_baseline_files_manual, ["PN_new.csv"])

    def test_verification_failure_preserves_recipe(self):
        self.store.mark_used.side_effect = ValueError("Source changed")
        edit_pn_background(self.owner)
        self.finish()
        self.assertEqual(self.owner.drr_baseline_files_manual, ["old.csv"])
        self.owner._show_error.assert_called_once()
        self.owner._start_load.assert_not_called()

    def test_stale_consent_is_forwarded_but_invalid_result_is_not_applied_on_close(self):
        self.dialog.allow_stale = True
        edit_pn_background(self.owner)
        self.jobs[0].run()
        self.owner._is_closing = True
        self.app.processEvents()
        self.store.mark_used.assert_called_once_with(
            "v1", measurements=("sample.csv",), allow_stale=True,
        )
        self.owner._start_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
