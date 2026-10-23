"""Queued BN results cannot commit after a different recipe becomes current."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QCoreApplication, QEvent, QObject
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QPushButton, QSpinBox

from ui_qt import drr_bn_workflow as workflow
from ui_qt.drr_bn_calibration_dialog import DrrBnCalibrationDialog
from tests.bn_joint_fixtures import accepted_joint_report


class _Owner(QObject):
    def __init__(self, **values):
        super().__init__()
        self.__dict__.update(values)


class BnWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.jobs = []
        self.dialogs = []
        self.owner = _Owner(
            current_folder=str(self.root), drr_selected_files=["sample_a.csv", "sample_b.csv"],
            drr_baseline_files_manual=["reference_BNbackground.csv"], drr_baseline_files_found=[],
            _drr_background_regions=(), _drr_assignments=(), _drr_assignments_automatic=False,
            _is_closing=False, _show_error=MagicMock(), _status=MagicMock(), _start_load=MagicMock(),
            _selected_y_axis_spec=MagicMock(return_value="BG"),
            thread_pool=SimpleNamespace(start=self.jobs.append),
            drr_baseline_combo=QComboBox(), drr_baseline_combine_combo=QComboBox(),
            drr_pin_baseline_chk=QCheckBox(), drr_sg_window_spin=QSpinBox(), drr_sg_poly_spin=QSpinBox(),
            drr_bn_auto_chk=QCheckBox(), drr_bn_calibration_btn=QPushButton(), drr_bn_status=QLabel(),
            drr_controller=SimpleNamespace(_cancel_auto_external=MagicMock(),
                                           _update_drr_selection_labels=MagicMock()),
        )
        self.owner.drr_baseline_combo.addItems(["Self (last frame)", "External"])
        self.owner.drr_baseline_combo.setCurrentText("External")
        self.owner.drr_baseline_combine_combo.addItems([
            "First frame from each file, then average", "Last frame from each file, then average",
            "Average all frames in each file, then average files",
        ])
        self.owner.drr_baseline_combine_combo.setCurrentIndex(2)
        self.owner.drr_sg_window_spin.setValue(21)
        self.owner.drr_sg_poly_spin.setValue(3)
        self.owner.loaded = SimpleNamespace(mode="DRR", selected_files=tuple(self.owner.drr_selected_files),
                                            folder=str(self.root))
        self.payload = {
            "eligible": True, "summary": "Held-out checks passed.", "version_id": "version_1",
            "csv_path": str(self.root / ".drr_bn_calibration" / "version_1" / "background.csv"),
            "source_identity": {
                "root": str(self.root), "measurements": [{"path": x} for x in self.owner.drr_selected_files],
                "baseline_files": [{"path": "reference_BNbackground.csv"}], "baseline_which": "all",
                "y_axis": "BG",
            },
            "settings": {"derivative_window": 21, "derivative_polyorder": 3},
            "report": accepted_joint_report(21, 3),
            "original_chosen": False,
        }
        self.version = SimpleNamespace(version_id="version_1", csv_path=Path(self.payload["csv_path"]),
                                       fresh=True, eligible=True,
                                       settings=deepcopy(self.payload['settings']),
                                       source_identity=deepcopy(self.payload["source_identity"]))
        self.store = MagicMock()
        self.store.get.return_value = self.version
        self.store.find_current.return_value = self.version
        self.store.version_for_csv.return_value = self.version
        self.store.load_result.side_effect = lambda version: deepcopy(self.payload)
        self.store_patch = patch.object(workflow, "DrrBnStore", return_value=self.store)
        self.store_patch.start()

    def tearDown(self):
        workflow.cancel_bn_requests(self.owner)
        self.app.processEvents()
        for dialog in self.dialogs:
            dialog.set_error("Test cleanup")
            dialog.close()
            dialog.deleteLater()
        self.owner.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.store_patch.stop()
        self.temp.cleanup()

    def dialog(self):
        dialog = DrrBnCalibrationDialog(context_label="Current group", settings=self.payload['settings'], initial_payload=self.payload)
        self.dialogs.append(dialog)
        return dialog

    def deliver(self, index=0):
        self.jobs[index].run()
        self.app.processEvents()

    def test_ineligible_or_mismatched_payload_cannot_queue_application(self):
        dialog = self.dialog()
        invalid = deepcopy(self.payload)
        invalid["eligible"] = False
        workflow.apply_bn_payload(self.owner, invalid, dialog)
        self.assertEqual(self.jobs, [])
        self.assertFalse(dialog._application_busy)
        invalid = deepcopy(self.payload)
        invalid["source_identity"]["measurements"] = [{"path": "other.csv"}]
        workflow.apply_bn_payload(self.owner, invalid, dialog)
        self.assertEqual(self.jobs, [])
        self.store.activate_verified.assert_not_called()
        self.owner._selected_y_axis_spec.return_value = "TG"
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.assertEqual(self.jobs, [])

    def test_worker_only_verifies_then_current_gui_turn_commits_and_binds_one_group(self):
        self.owner.drr_pin_baseline_chk.setChecked(True)
        dialog = self.dialog()
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.assertTrue(dialog._application_busy)
        self.store.get.assert_not_called()
        self.jobs[0].run()
        self.store.get.assert_called_once_with("version_1")
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()
        self.app.processEvents()
        self.store.activate_verified.assert_called_once_with(
            self.version, measurements=("sample_a.csv", "sample_b.csv"))
        self.assertFalse(self.owner.drr_pin_baseline_chk.isChecked())
        self.assertEqual(len(self.owner._drr_assignments), 2)
        self.assertEqual({x.measurement_file for x in self.owner._drr_assignments},
                         {"sample_a.csv", "sample_b.csv"})
        self.assertTrue(all(x.baseline_files == (".drr_bn_calibration/version_1/background.csv",)
                            for x in self.owner._drr_assignments))
        self.assertFalse(self.owner.drr_pin_baseline_chk.isEnabled())
        self.assertFalse(self.owner.drr_baseline_combine_combo.isEnabled())
        self.owner._start_load.assert_called_once_with("DRR")
        self.assertFalse(dialog._application_busy)
        self.assertIn("validated BN candidate", dialog.mode_label.text())

    def test_queued_result_is_discarded_after_group_change(self):
        dialog = self.dialog()
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.jobs[0].run()
        self.owner.drr_selected_files = ["other.csv"]
        self.app.processEvents()
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()
        self.assertFalse(dialog._application_busy)

    def test_queued_result_is_discarded_after_reference_change(self):
        workflow.apply_bn_payload(self.owner, self.payload)
        self.jobs[0].run()
        self.owner.drr_baseline_files_manual = ["new_BNbackground.csv"]
        self.app.processEvents()
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_queued_result_is_discarded_after_frame_method_change(self):
        workflow.apply_bn_payload(self.owner, self.payload)
        self.jobs[0].run()
        self.owner.drr_baseline_combine_combo.setCurrentIndex(0)
        self.app.processEvents()
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_queued_result_is_discarded_when_owner_closes(self):
        dialog = self.dialog()
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.jobs[0].run()
        self.owner._is_closing = True
        self.app.processEvents()
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()
        self.assertFalse(dialog._application_busy)

    def test_sg_change_discards_queued_joint_verification(self):
        workflow.apply_bn_payload(self.owner, self.payload)
        self.jobs[0].run()
        self.owner.drr_sg_window_spin.setValue(11)
        self.owner.drr_sg_poly_spin.setValue(2)
        self.app.processEvents()
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_joint_report_is_required_even_if_payload_says_eligible(self):
        for report in ({}, {'joint_validation': {'passed': False}}):
            payload = deepcopy(self.payload)
            payload['report'] = report
            workflow.apply_bn_payload(self.owner, payload)
        self.assertEqual(self.jobs, [])

    def test_d2_priority_application_accepts_negative_drr_gain_when_constraint_passes(self):
        payload = deepcopy(self.payload)
        payload['report']['final_confirmation'].update(
            mean_original_rms=.001, mean_candidate_rms=.00101, mean_improvement_fraction=-.01)
        for role in ('selection', 'final'):
            payload['report']['joint_validation'][role]['drr_benefit_passed'] = False
        workflow.apply_bn_payload(self.owner, payload)
        self.deliver()
        self.store.activate_verified.assert_called_once()

    def test_actual_drr_noninferiority_failure_prevents_application(self):
        payload = deepcopy(self.payload)
        payload['report']['joint_validation']['final']['drr_noninferiority_passed'] = False
        workflow.apply_bn_payload(self.owner, payload)
        self.assertEqual(self.jobs, [])
        self.store.activate_verified.assert_not_called()

    def test_previous_joint_policy_is_preview_only_even_if_saved_version_says_eligible(self):
        self.payload['candidates'] = [{'name': 'local_shift', 'eligible': True}]
        self.payload['report'].update(schema_version=2, optimization_policy='joint_benefit_v1')
        self.payload['report']['joint_validation'].update(schema_version=1, policy='joint_benefit_v1')
        result = workflow._load_saved(str(self.root), tuple(self.owner.drr_selected_files),
                                      ('reference_BNbackground.csv',), 'all', 'BG', (21, 3),
                                      progress=MagicMock(), log=MagicMock())
        self.assertFalse(result['eligible'])
        self.assertFalse(result['candidates'][0]['eligible'])
        self.assertIn('older optimization policy', result['summary'])
        self.assertIn('D2E as the target and DRR as a constraint', result['summary'])
        workflow.apply_bn_payload(self.owner, self.payload)
        self.assertEqual(self.jobs, [])

    def test_current_policy_failed_comparison_is_not_mislabeled_legacy(self):
        self.payload['eligible'] = False
        self.payload['report']['joint_validation']['passed'] = False
        result = workflow._load_saved(str(self.root), tuple(self.owner.drr_selected_files),
                                      ('reference_BNbackground.csv',), 'all', 'BG', (21, 3),
                                      progress=MagicMock(), log=MagicMock())
        self.assertFalse(result['eligible'])
        self.assertNotIn('older optimization policy', result['summary'])

    def test_previous_d2_priority_report_can_preview_but_never_queues_activation(self):
        self.payload['candidates'] = [{'name': 'local_shift', 'eligible': True}]
        self.payload['report'].update(schema_version=3, optimization_policy='d2_priority_v1')
        self.payload['report']['joint_validation'].update(schema_version=2, policy='d2_priority_v1')
        result = workflow._load_saved(str(self.root), tuple(self.owner.drr_selected_files),
                                      ('reference_BNbackground.csv',), 'all', 'BG', (21, 3),
                                      progress=MagicMock(), log=MagicMock())
        self.assertFalse(result['eligible'])
        self.assertFalse(result['candidates'][0]['eligible'])
        self.assertIn('current weak-feature checks', result['summary'])
        workflow.apply_bn_payload(self.owner, self.payload)
        self.assertEqual(self.jobs, [])
        self.store.activate_verified.assert_not_called()

    def test_status_displays_readable_strength_and_reason_without_mutating_payload(self):
        self.payload['summary'] = 'fallback: local_shift_s25. selection_feature_fidelity_not_confirmed'
        workflow._remember(self.owner, self.payload)
        workflow.update_bn_controls(self.owner)
        self.assertIn('Local fringe shift · 25% correction', self.owner.drr_bn_status.text())
        self.assertIn('Selection feature fidelity not confirmed', self.owner.drr_bn_status.text())
        self.assertNotIn('local_shift_s25', self.owner.drr_bn_status.text())
        self.assertIn('local_shift_s25', self.payload['summary'])

    def test_version_settings_are_checked_again_in_verification_worker(self):
        dialog = self.dialog()
        self.version.settings['derivative_window'] = 11
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.deliver()
        self.assertIn('different SG', dialog.status.text())
        self.store.activate_verified.assert_not_called()

    def test_cached_joint_confirmation_is_scoped_to_sg_and_axis(self):
        workflow._remember(self.owner, self.payload)
        self.assertIs(workflow._cached_payload(self.owner), self.payload)
        self.owner.drr_sg_window_spin.setValue(11)
        self.assertIsNone(workflow._cached_payload(self.owner))
        self.assertFalse(workflow._matches(self.owner, self.payload))
        self.assertTrue(workflow._matches(self.owner, self.payload, check_derivative=False))
        workflow.apply_bn_payload(self.owner, self.payload)
        self.assertEqual(self.jobs, [])

    def test_derivative_hook_withdraws_dialog_confirmation_and_cancels_apply(self):
        dialog = self.dialog()
        self.owner._drr_bn_dialog = dialog
        dialog._bn_workflow_context = workflow._context(self.owner)
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.owner.drr_sg_window_spin.setValue(11)
        workflow.derivative_settings_changed(self.owner)
        self.assertIsNone(self.owner._drr_bn_apply_request)
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertIn('not been validated', dialog.joint_validation.text())
        self.assertEqual(dialog.settings()['derivative_window'], 11)
        self.assertEqual(dialog._bn_workflow_context, workflow._context(self.owner))
        self.deliver()
        self.store.activate_verified.assert_not_called()

    def test_auto_comparison_uses_actual_sg_controls(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.owner.drr_sg_window_spin.setValue(11)
        self.owner.drr_sg_poly_spin.setValue(2)
        with patch.object(workflow, '_compare_and_save', return_value={'eligible': False}) as compare:
            workflow.maybe_auto_compare(self.owner)
            self.deliver()
        self.assertEqual(compare.call_args.args[4], {'derivative_window': 11, 'derivative_polyorder': 2})

    def test_different_sg_saved_report_is_preview_only(self):
        result = workflow._load_saved(str(self.root), tuple(self.owner.drr_selected_files),
                                     ('reference_BNbackground.csv',), 'all', 'BG', (11, 2),
                                     progress=MagicMock(), log=MagicMock())
        self.assertFalse(result['eligible'])
        self.assertIn('different SG', result['summary'])
        self.assertEqual(self.payload['eligible'], True)

    def test_original_can_be_restored_after_sg_changes(self):
        workflow.apply_bn_payload(self.owner, self.payload)
        self.deliver()
        self.owner.drr_sg_window_spin.setValue(11)
        workflow.restore_original(self.owner, self.payload)
        self.assertEqual(self.owner.drr_baseline_files_manual, ['reference_BNbackground.csv'])
        self.store.deactivate.assert_called_once()

    def test_gate_axis_change_discards_sample_based_verification(self):
        workflow.apply_bn_payload(self.owner, self.payload)
        self.jobs[0].run()
        self.owner._selected_y_axis_spec.return_value = "TG"
        self.app.processEvents()
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_explicit_cancel_clears_busy_and_prevents_verification_commit(self):
        dialog = self.dialog()
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        workflow.cancel_bn_requests(self.owner)
        self.assertFalse(dialog._application_busy)
        self.deliver()
        self.store.get.assert_not_called()
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_old_finished_does_not_clear_new_request_busy_state(self):
        dialog = self.dialog()
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        second = self.owner._drr_bn_apply_request
        self.deliver(0)
        self.assertIs(self.owner._drr_bn_apply_request, second)
        self.assertTrue(dialog._application_busy)
        self.deliver(1)
        self.assertFalse(dialog._application_busy)
        self.store.activate_verified.assert_called_once()

    def test_verification_failure_restores_busy_without_recipe_change(self):
        dialog = self.dialog()
        self.store.get.side_effect = ValueError("Reference bytes changed.")
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.deliver()
        self.assertFalse(dialog._application_busy)
        self.assertIn("Reference bytes changed", dialog.status.text())
        self.assertEqual(self.owner.drr_baseline_files_manual, ["reference_BNbackground.csv"])
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_commit_failure_does_not_change_recipe(self):
        dialog = self.dialog()
        self.store.activate_verified.side_effect = OSError("Manifest could not be written.")
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.deliver()
        self.assertFalse(dialog._application_busy)
        self.assertIn("Manifest could not be written", dialog.status.text())
        self.assertEqual(self.owner.drr_baseline_files_manual, ["reference_BNbackground.csv"])
        self.owner._start_load.assert_not_called()

    def test_load_start_failure_after_commit_releases_application_busy(self):
        dialog = self.dialog()
        self.owner._start_load.side_effect = RuntimeError("DRR reload could not start.")
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.deliver()
        self.store.activate_verified.assert_called_once()
        self.assertFalse(dialog._application_busy)
        self.assertIn("DRR reload could not start", dialog.status.text())
        self.assertIn("DRR reload could not start", self.owner._status.call_args.args[0])
        self.assertIsNone(self.owner._drr_bn_apply_request)

    def test_restore_original_records_tombstone_and_prevents_automatic_reapply(self):
        dialog = self.dialog()
        workflow.apply_bn_payload(self.owner, self.payload, dialog)
        self.deliver()
        workflow.restore_original(self.owner, self.payload, dialog)
        self.store.deactivate.assert_called_once_with(
            ("sample_a.csv", "sample_b.csv"), ("reference_BNbackground.csv",), "all", restore_original=True)
        self.assertTrue(self.payload["original_chosen"])
        self.assertEqual(self.owner.drr_baseline_files_manual, ["reference_BNbackground.csv"])
        self.assertEqual(workflow.frame_method(self.owner), "all")
        self.assertFalse(self.owner.drr_pin_baseline_chk.isChecked())
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.store.activate_verified.reset_mock()
        with patch.object(workflow, "_compare_and_save", return_value=self.payload):
            workflow.maybe_auto_compare(self.owner)
            self.deliver(1)
        self.assertEqual(len(self.jobs), 2)
        self.store.activate_verified.assert_not_called()
        self.assertIsNone(self.owner._drr_bn_apply_request)

    def test_restore_original_failure_is_reported_without_recipe_mutation(self):
        dialog = self.dialog()
        self.store.deactivate.side_effect = OSError("Manifest is read-only.")
        workflow.restore_original(self.owner, self.payload, dialog)
        self.assertFalse(self.payload["original_chosen"])
        self.assertIn("Manifest is read-only", dialog.status.text())
        self.assertEqual(self.owner.drr_baseline_files_manual, ["reference_BNbackground.csv"])
        self.owner._start_load.assert_not_called()

    def test_auto_compare_requires_checked_drr_original_group_and_deduplicates_load(self):
        workflow.maybe_auto_compare(self.owner)
        self.assertEqual(self.jobs, [])
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.owner.loaded.mode = "PL"
        workflow.maybe_auto_compare(self.owner)
        self.assertEqual(self.jobs, [])
        self.owner.loaded.mode = "DRR"
        self.owner.loaded.selected_files = ("another.csv",)
        workflow.maybe_auto_compare(self.owner)
        self.assertEqual(self.jobs, [])
        self.owner.loaded.selected_files = tuple(self.owner.drr_selected_files)
        self.payload["eligible"] = False
        with patch.object(workflow, "_compare_and_save", return_value=self.payload) as compare:
            workflow.maybe_auto_compare(self.owner)
            self.deliver()
            workflow.maybe_auto_compare(self.owner)
            self.assertEqual(len(self.jobs), 1)
            compare.assert_called_once()
        self.store.activate_verified.assert_not_called()

    def test_auto_compare_does_not_run_for_verified_managed_background_or_open_dialog(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.owner._drr_bn_dialog = self.dialog()
        workflow.maybe_auto_compare(self.owner)
        self.assertEqual(self.jobs, [])
        self.owner._drr_bn_dialog = None
        self.owner.drr_baseline_files_manual = [".drr_bn_calibration/version_1/background.csv"]
        workflow._remember(self.owner, self.payload)
        workflow.maybe_auto_compare(self.owner)
        self.assertEqual(self.jobs, [])
        self.owner.loaded = SimpleNamespace(mode='DRR', folder=str(self.root),
                                            selected_files=tuple(self.owner.drr_selected_files))
        workflow.maybe_auto_compare(self.owner)
        self.assertEqual(self.jobs, [])

    def test_managed_version_recomparison_binds_only_exact_input_recipe(self):
        original_csv = '.drr_bn_calibration/version_1/background.csv'
        self.owner.drr_baseline_files_manual = [original_csv]
        self.owner.drr_baseline_combine_combo.setCurrentIndex(1)
        newer = deepcopy(self.payload)
        newer['version_id'] = 'version_2'
        newer['csv_path'] = str(self.root / '.drr_bn_calibration/version_2/background.csv')
        self.version.version_id = newer['version_id']
        self.version.csv_path = Path(newer['csv_path'])
        with patch('core.drr_bn_service.compare_bn_group', return_value=newer):
            result = workflow._compare_and_save(
                str(self.root), tuple(self.owner.drr_selected_files), (original_csv,), 'last',
                dict(self.payload['settings']), 'BG', progress=MagicMock(), log=MagicMock())
        self.assertEqual(result['comparison_input_backgrounds'], (original_csv,))
        self.assertTrue(workflow._matches(self.owner, result))
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_3/background.csv']
        self.assertFalse(workflow._matches(self.owner, result))
        self.owner.drr_baseline_files_manual = [original_csv]
        workflow.apply_bn_payload(self.owner, result)
        self.deliver()
        self.store.activate_verified.assert_called_once()
        self.assertEqual(self.owner.drr_baseline_files_manual,
                         ['.drr_bn_calibration/version_2/background.csv'])

    def test_managed_recomparison_can_restore_its_raw_original(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        self.owner.drr_baseline_combine_combo.setCurrentIndex(1)
        payload = deepcopy(self.payload)
        payload['csv_path'] = str(self.root / '.drr_bn_calibration/version_2/background.csv')
        payload['comparison_input_backgrounds'] = tuple(self.owner.drr_baseline_files_manual)
        payload['comparison_input_which'] = 'last'
        workflow.restore_original(self.owner, payload)
        self.assertEqual(self.owner.drr_baseline_files_manual, ['reference_BNbackground.csv'])
        self.store.deactivate.assert_called_once()

    def test_managed_sg_recomparison_preserves_saved_fitting_settings(self):
        self.version.settings.update(trusted_regions=[[1.88, 1.99]], uncertain_regions=[])
        with patch('core.drr_bn_service.compare_bn_group', return_value=deepcopy(self.payload)) as compare:
            workflow._compare_and_save(str(self.root), tuple(self.owner.drr_selected_files),
                                      ('.drr_bn_calibration/version_1/background.csv',), 'last',
                                      {'derivative_window': 11, 'derivative_polyorder': 2}, 'BG',
                                      progress=MagicMock(), log=MagicMock())
        settings = compare.call_args.args[4]
        self.assertEqual(settings['derivative_window'], 11)
        self.assertEqual(settings['derivative_polyorder'], 2)
        self.assertEqual(settings['trusted_regions'], [[1.88, 1.99]])
        self.assertEqual(settings['uncertain_regions'], [])

    def test_sg_hook_automatically_recompares_managed_background(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        workflow._remember(self.owner, self.payload)
        self.owner.drr_sg_window_spin.setValue(11)
        workflow.derivative_settings_changed(self.owner)
        self.assertEqual(len(self.jobs), 1)
        self.assertIsNotNone(self.owner._drr_bn_compare_request)

    def test_managed_derivative_guard_checks_the_current_csv_not_a_new_preview(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        self.owner.drr_baseline_combine_combo.setCurrentIndex(1)
        workflow._remember(self.owner, self.payload)
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))
        preview = deepcopy(self.payload)
        preview['comparison_input_backgrounds'] = tuple(self.owner.drr_baseline_files_manual)
        preview['comparison_input_which'] = 'last'
        preview['csv_path'] = str(self.root / '.drr_bn_calibration/version_2/background.csv')
        workflow._remember(self.owner, preview)
        self.assertTrue(workflow._matches(self.owner, preview))
        self.assertFalse(workflow.managed_joint_validation_current(self.owner))
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_2/background.csv']
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))
        self.owner.drr_sg_window_spin.setValue(11)
        self.assertFalse(workflow.managed_joint_validation_current(self.owner))

    def test_fallback_keeps_derivative_blocked_until_raw_cube_reload_finishes(self):
        managed = '.drr_bn_calibration/version_1/background.csv'
        self.owner.drr_baseline_files_manual = [managed]
        self.owner.loaded.baseline_files = [managed]
        workflow._remember(self.owner, self.payload)
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))
        payload = deepcopy(self.payload)
        payload['eligible'] = False
        self.assertTrue(workflow._fallback_unverified_managed(self.owner, payload))
        self.assertEqual(self.owner.drr_baseline_files_manual, ['reference_BNbackground.csv'])
        self.assertFalse(workflow.managed_joint_validation_current(self.owner))
        self.owner.loaded.baseline_files = ['reference_BNbackground.csv']
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))

    def test_new_candidate_certificate_does_not_unlock_old_loaded_cube(self):
        managed = '.drr_bn_calibration/version_1/background.csv'
        self.owner.drr_baseline_files_manual = [managed]
        self.owner.loaded.baseline_files = ['.drr_bn_calibration/older/background.csv']
        workflow._remember(self.owner, self.payload)
        self.assertFalse(workflow.managed_joint_validation_current(self.owner))
        self.owner.loaded.baseline_files = [managed]
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))

    def test_certificate_waits_for_pending_recipe_cube_but_can_start_in_loaded_callback(self):
        managed = '.drr_bn_calibration/version_1/background.csv'
        self.owner.drr_baseline_files_manual = [managed]
        self.owner.loaded.baseline_files = ['.drr_bn_calibration/older/background.csv']
        self.owner._load_in_progress = True
        workflow.ensure_managed_bn_validation(self.owner)
        self.assertEqual(self.jobs, [])
        self.assertFalse(workflow.managed_joint_validation_current(self.owner))
        self.owner.loaded.baseline_files = [managed]
        workflow.ensure_managed_bn_validation(self.owner)
        self.assertEqual(len(self.jobs), 1)
        self.deliver()
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))
        self.store.clear_activation.assert_not_called()

    def test_failed_managed_auto_comparison_restores_raw_without_original_choice(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        payload = deepcopy(self.payload)
        payload['eligible'] = False
        payload['report']['joint_validation']['passed'] = False
        with patch.object(workflow, '_compare_and_save', return_value=payload):
            workflow.maybe_auto_compare(self.owner)
            self.deliver()
        self.store.clear_activation.assert_called_once_with(('sample_a.csv', 'sample_b.csv'))
        self.store.deactivate.assert_not_called()
        self.assertFalse(payload['original_chosen'])
        self.assertEqual(self.owner.drr_baseline_files_manual, ['reference_BNbackground.csv'])
        self.owner._start_load.assert_called_once_with('DRR')
        self.assertIn('Original BN restored', self.owner._status.call_args.args[0])

    def test_same_managed_saved_certificate_restores_derivative_without_reload(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        self.owner._schedule_plot_redraw = MagicMock()
        self.assertFalse(workflow.managed_joint_validation_current(self.owner))
        with patch.object(workflow, '_compare_and_save', return_value=self.payload):
            workflow.maybe_auto_compare(self.owner)
            self.deliver()
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))
        self.owner._schedule_plot_redraw.assert_called_once_with('DRR')
        self.owner._start_load.assert_not_called()
        self.store.activate_verified.assert_not_called()
        self.assertEqual(len(self.jobs), 1)

    def test_cold_managed_certificate_is_recovered_in_worker_with_auto_disabled(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        self.owner._schedule_plot_redraw = MagicMock()
        self.assertFalse(self.owner.drr_bn_auto_chk.isChecked())
        workflow.ensure_managed_bn_validation(self.owner)
        workflow.ensure_managed_bn_validation(self.owner)
        self.assertEqual(len(self.jobs), 1)
        self.store.version_for_csv.assert_not_called()
        self.assertFalse(workflow.managed_joint_validation_current(self.owner))
        self.deliver()
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))
        self.owner._schedule_plot_redraw.assert_called_once_with('DRR')
        self.owner._start_load.assert_not_called()
        self.store.activate_verified.assert_not_called()
        self.store.clear_activation.assert_not_called()
        workflow.ensure_managed_bn_validation(self.owner)
        self.assertEqual(len(self.jobs), 1)
        self.assertFalse(self.owner.drr_bn_auto_chk.isChecked())

    def test_cold_managed_certificate_at_old_sg_restores_raw_without_tombstone(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        self.owner.drr_sg_window_spin.setValue(11)
        workflow.ensure_managed_bn_validation(self.owner)
        self.deliver()
        self.store.clear_activation.assert_called_once_with(('sample_a.csv', 'sample_b.csv'))
        self.store.deactivate.assert_not_called()
        self.assertEqual(self.owner.drr_baseline_files_manual, ['reference_BNbackground.csv'])
        self.owner._start_load.assert_called_once_with('DRR')
        self.assertFalse(self.payload['original_chosen'])

    def test_cold_managed_certificate_with_other_axis_restores_raw(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        self.owner._selected_y_axis_spec.return_value = 'TG'
        workflow.ensure_managed_bn_validation(self.owner)
        self.deliver()
        self.store.clear_activation.assert_called_once()
        self.assertEqual(self.owner.drr_baseline_files_manual, ['reference_BNbackground.csv'])

    def test_late_cold_certificate_cannot_restore_or_enable_another_sg(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        workflow.ensure_managed_bn_validation(self.owner)
        self.jobs[0].run()
        self.owner.drr_sg_window_spin.setValue(11)
        self.app.processEvents()
        self.assertFalse(workflow.managed_joint_validation_current(self.owner))
        self.store.clear_activation.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_cold_certificate_cancel_prevents_io_and_allows_new_check(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        workflow.ensure_managed_bn_validation(self.owner)
        workflow.cancel_bn_requests(self.owner)
        self.deliver()
        self.store.version_for_csv.assert_not_called()
        workflow.ensure_managed_bn_validation(self.owner)
        self.assertEqual(len(self.jobs), 2)
        self.deliver(1)
        self.assertTrue(workflow.managed_joint_validation_current(self.owner))

    def test_failed_manual_comparison_restores_raw_without_original_tombstone(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        payload = deepcopy(self.payload)
        payload['eligible'] = False
        payload['report']['joint_validation']['passed'] = False
        with patch('ui_qt.drr_bn_calibration_dialog.DrrBnCalibrationDialog') as factory:
            dialog = factory.return_value
            dialog._payload = payload
            dialog.exec.side_effect = lambda: dialog.comparisonCompleted.connect.call_args.args[0](payload)
            workflow._show_dialog(self.owner, workflow._context(self.owner),
                                  tuple(self.owner.drr_baseline_files_manual), 'all', 'BG', None)
        self.store.clear_activation.assert_called_once()
        self.store.deactivate.assert_not_called()
        self.assertFalse(payload['original_chosen'])
        self.assertEqual(self.owner.drr_baseline_files_manual, ['reference_BNbackground.csv'])
        dialog.set_application_state.assert_called_once()

    def test_previewing_rejected_saved_report_never_changes_current_recipe(self):
        self.owner.drr_baseline_files_manual = ['.drr_bn_calibration/version_1/background.csv']
        payload = deepcopy(self.payload)
        payload['eligible'] = False
        with patch('ui_qt.drr_bn_calibration_dialog.DrrBnCalibrationDialog') as factory:
            factory.return_value._payload = payload
            workflow._show_dialog(self.owner, workflow._context(self.owner),
                                  tuple(self.owner.drr_baseline_files_manual), 'all', 'BG', payload)
        self.store.clear_activation.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_sg_auto_recomparison_respects_explicit_original_choice(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        payload = deepcopy(self.payload)
        payload.update(original_chosen=True, settings={'derivative_window': 11, 'derivative_polyorder': 3},
                       report=accepted_joint_report(11, 3))
        self.owner.drr_sg_window_spin.setValue(11)
        with patch.object(workflow, '_compare_and_save', return_value=payload):
            workflow.derivative_settings_changed(self.owner)
            self.deliver()
            workflow.maybe_auto_compare(self.owner)
        self.assertEqual(len(self.jobs), 1)
        self.assertIsNone(getattr(self.owner, '_drr_bn_apply_request', None))
        self.store.activate_verified.assert_not_called()
        self.assertEqual(self.owner.drr_baseline_files_manual, ['reference_BNbackground.csv'])

    def test_non_bn_reference_does_not_start_automatic_calibration(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.owner.drr_baseline_files_manual = ['generated_PN_background.csv']
        workflow.update_bn_controls(self.owner)
        workflow.maybe_auto_compare(self.owner)
        self.assertFalse(self.owner.drr_bn_calibration_btn.isEnabled())
        self.assertEqual(self.jobs, [])

    def test_auto_compare_rejects_loaded_group_from_a_different_folder(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        self.owner.loaded.folder = str(self.root / "previous_experiment")
        workflow.maybe_auto_compare(self.owner)
        self.assertEqual(self.jobs, [])

    def test_disabling_auto_cancels_a_queued_comparison_without_running_it(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        with patch.object(workflow, "_compare_and_save", return_value=self.payload) as compare:
            workflow.maybe_auto_compare(self.owner)
            self.owner.drr_bn_auto_chk.setChecked(False)
            workflow.auto_toggled(self.owner, False)
            self.deliver()
            compare.assert_not_called()
        self.assertIsNone(self.owner._drr_bn_compare_request)
        self.store.activate_verified.assert_not_called()

    def test_stale_auto_result_cannot_start_application(self):
        self.owner.drr_bn_auto_chk.setChecked(True)
        with patch.object(workflow, "_compare_and_save", return_value=self.payload):
            workflow.maybe_auto_compare(self.owner)
            self.jobs[0].run()
            self.owner.drr_baseline_combine_combo.setCurrentIndex(0)
            self.app.processEvents()
        self.assertEqual(len(self.jobs), 1)
        self.store.activate_verified.assert_not_called()

    def test_cold_open_reads_saved_report_in_worker_before_showing_dialog(self):
        self.payload["settings"] = {"core_region": [1.9, 1.96], "gate_count": 8}
        with patch.object(workflow, "_show_dialog") as show:
            workflow.edit_bn_calibration(self.owner)
            self.assertEqual(len(self.jobs), 1)
            self.store.find_current.assert_not_called()
            self.assertFalse(self.owner.drr_bn_calibration_btn.isEnabled())
            self.jobs[0].run()
            self.store.find_current.assert_called_once_with(
                ("sample_a.csv", "sample_b.csv"), ("reference_BNbackground.csv",), "all")
            show.assert_not_called()
            self.app.processEvents()
            show.assert_called_once()
            payload = show.call_args.args[-1]
            self.assertEqual(payload["settings"], self.payload["settings"])
            self.assertEqual(payload["csv_path"], str(self.version.csv_path))
            self.assertIsNone(self.owner._drr_bn_open_request)
            self.assertTrue(self.owner.drr_bn_calibration_btn.isEnabled())
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_cold_open_without_saved_result_shows_empty_comparison(self):
        self.store.find_current.return_value = None
        with patch.object(workflow, "_show_dialog") as show:
            workflow.edit_bn_calibration(self.owner)
            self.deliver()
            self.assertIsNone(show.call_args.args[-1])
        self.store.load_result.assert_not_called()

    def test_managed_cold_open_loads_its_original_provenance(self):
        files = (".drr_bn_calibration/version_1/background.csv",)
        result = workflow._load_saved(str(self.root), tuple(self.owner.drr_selected_files),
                                      files, "last", "BG", progress=MagicMock(), log=MagicMock())
        self.store.version_for_csv.assert_called_once_with(files[0])
        self.store.find_current.assert_not_called()
        self.assertEqual(result["source_identity"]["baseline_files"],
                         [{"path": "reference_BNbackground.csv"}])
        self.assertEqual(result["source_identity"]["baseline_which"], "all")

    def test_managed_saved_report_from_another_group_is_rejected(self):
        self.version.source_identity["measurements"] = [{"path": "another_group.csv"}]
        with self.assertRaisesRegex(ValueError, "another measurement group"):
            workflow._load_saved(str(self.root), tuple(self.owner.drr_selected_files),
                                 (".drr_bn_calibration/version_1/background.csv",), "last", "BG",
                                 progress=MagicMock(), log=MagicMock())
        self.store.load_result.assert_not_called()

    def test_stale_saved_report_is_preview_only_including_candidate_rows(self):
        self.version.fresh = False
        self.payload["candidates"] = [{"name": "local_shift", "eligible": True},
                                      {"name": "local_shift_gain", "eligible": True}]
        result = workflow._load_saved(str(self.root), tuple(self.owner.drr_selected_files),
                                     ("reference_BNbackground.csv",), "all", "BG",
                                     progress=MagicMock(), log=MagicMock())
        self.assertFalse(result["eligible"])
        self.assertTrue(all(row["eligible"] is False for row in result["candidates"]))
        self.assertIn("stale", result["summary"])
        dialog = DrrBnCalibrationDialog(context_label="Saved report", initial_payload=result)
        self.dialogs.append(dialog)
        self.assertFalse(dialog.use_btn.isEnabled())
        self.store.activate_verified.assert_not_called()

    def test_other_axis_saved_report_is_preview_only_including_candidate_rows(self):
        self.payload["candidates"] = [{"name": "local_shift", "eligible": True}]
        result = workflow._load_saved(str(self.root), tuple(self.owner.drr_selected_files),
                                     ("reference_BNbackground.csv",), "all", "TG",
                                     progress=MagicMock(), log=MagicMock())
        self.assertFalse(result["eligible"])
        self.assertTrue(all(row["eligible"] is False for row in result["candidates"]))
        self.assertIn("axis", result["summary"].lower())

    def test_original_can_be_restored_after_gate_axis_changes(self):
        workflow.apply_bn_payload(self.owner, self.payload)
        self.deliver()
        self.owner._selected_y_axis_spec.return_value = "TG"
        self.owner._start_load.reset_mock()
        workflow.restore_original(self.owner, self.payload)
        self.store.deactivate.assert_called_once_with(
            ("sample_a.csv", "sample_b.csv"), ("reference_BNbackground.csv",), "all", restore_original=True)
        self.assertEqual(self.owner.drr_baseline_files_manual, ["reference_BNbackground.csv"])
        self.owner._start_load.assert_called_once_with("DRR")

    def test_cold_open_is_discarded_if_gate_axis_changes_before_delivery(self):
        with patch.object(workflow, "_show_dialog") as show:
            workflow.edit_bn_calibration(self.owner)
            self.jobs[0].run()
            self.owner._selected_y_axis_spec.return_value = "TG"
            self.app.processEvents()
            show.assert_not_called()
        self.assertIsNone(self.owner._drr_bn_open_request)
        self.assertTrue(self.owner.drr_bn_calibration_btn.isEnabled())

    def test_cold_open_cancellation_and_failure_restore_controls(self):
        with patch.object(workflow, "_show_dialog") as show:
            workflow.edit_bn_calibration(self.owner)
            workflow.cancel_bn_requests(self.owner)
            self.deliver()
            show.assert_not_called()
            self.store.find_current.assert_not_called()
            self.assertTrue(self.owner.drr_bn_calibration_btn.isEnabled())
            self.store.find_current.side_effect = OSError("Saved report cannot be read.")
            workflow.edit_bn_calibration(self.owner)
            self.deliver(1)
            show.assert_not_called()
        self.assertTrue(self.owner.drr_bn_calibration_btn.isEnabled())
        self.assertIn("Saved report cannot be read", self.owner._show_error.call_args.args[0])

    def test_warm_open_reuses_matching_payload_without_saved_report_io(self):
        workflow._remember(self.owner, self.payload)
        with patch.object(workflow, "_show_dialog") as show:
            workflow.edit_bn_calibration(self.owner)
            show.assert_called_once()
            self.assertIs(show.call_args.args[-1], self.payload)
        self.assertEqual(self.jobs, [])
        self.store.find_current.assert_not_called()

    def test_saved_settings_and_payload_are_passed_to_dialog_without_applying(self):
        self.payload["settings"] = {"core_region": [1.9, 1.96], "gate_count": 8}
        with patch("ui_qt.drr_bn_calibration_dialog.DrrBnCalibrationDialog") as factory:
            dialog = factory.return_value
            dialog._payload = self.payload
            workflow._show_dialog(self.owner, workflow._context(self.owner),
                                  ("reference_BNbackground.csv",), "all", "BG", self.payload)
            self.assertEqual(factory.call_args.kwargs["settings"],
                             {**self.payload["settings"], 'derivative_window': 21, 'derivative_polyorder': 3})
            self.assertIs(factory.call_args.kwargs["initial_payload"], self.payload)
            self.assertEqual(factory.call_args.kwargs["current_mode"], "original")
            self.assertTrue(callable(factory.call_args.kwargs["run_comparison"]))
            dialog.exec.assert_called_once()
        self.store.activate_verified.assert_not_called()
        self.owner._start_load.assert_not_called()

    def test_protection_help_distinguishes_raw_drr_from_derivative_support(self):
        dialog = self.dialog()
        self.assertIn("raw ΔR/R unchanged", dialog.protection_note.text())
        self.assertIn("interpolation", dialog.protection_note.text())
        self.assertIn("wide SG windows", dialog.protection_note.text())


if __name__ == "__main__":
    unittest.main()
