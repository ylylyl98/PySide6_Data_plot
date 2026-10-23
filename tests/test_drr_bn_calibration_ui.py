"""Application gating and worker-lifecycle checks for per-group BN comparison."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QScrollArea

from ui_qt.drr_bn_calibration_dialog import DrrBnCalibrationDialog
from ui_qt.theme import install_theme
from tests.bn_joint_fixtures import accepted_joint_report


def _payload(eligible=False):
    energy = np.linspace(1.6, 2.1, 101)
    original = 100 + np.sin(energy * 200)
    result = {
        "eligible": eligible,
        "summary": "Validated on held-out repeats." if eligible else "Uncertain interval overlaps correction.",
        "settings": {"derivative_window": 21, "derivative_polyorder": 2},
        "report": {"schema_version": 4, "optimization_policy": "d2_shrinkage_v1", "selection_frozen": True,
                   "status": "validated" if eligible else "preview_only",
                   "joint_validation": {"schema_version": 3, "policy": "d2_shrinkage_v1", "passed": eligible},
                   "derivative_settings": {"window_length": 21, "polyorder": 2, "used_window_length": 21}},
        "energy": energy, "original_background": original,
        "candidate_background": original * 1.001, "mean_raw": original * 1.1,
        "candidates": [
            {"name": "Original", "improvement_fraction": 0, "eligible": False, "reason": "Baseline"},
            {"name": "Local shift", "improvement_fraction": .35, "improvement_fraction_d2": .22, "eligible": eligible,
             "reason": "Passed" if eligible else "Independent feature evidence is missing."},
        ],
    }
    if eligible:
        result['report'] = accepted_joint_report()
        result['report']['final_confirmation'].update(
            mean_original_rms=.001, mean_candidate_rms=.0008, mean_improvement_fraction=.2)
        result['report']['final_confirmation_d2'].update(
            mean_original_rms=4, mean_candidate_rms=3, mean_improvement_fraction=.25)
    return result


def _strength_preview():
    payload = _payload(False)
    payload["candidates"] = payload["candidates"][:1]
    for family in ("local_shift", "local_shift_gain"):
        for percent in (25, 50, 75, 100):
            name = family if percent == 100 else f"{family}_s{percent}"
            payload["candidates"].append({
                "name": name, "improvement_fraction": .02,
                "improvement_fraction_d2": .15, "eligible": False,
                "parameters": {"family": family, "correction_strength": percent / 100},
                "reason": "training_feature_influence_not_confirmed",
            })
    payload["selected_candidate"] = "local_shift_gain_s50"
    payload["summary"] = "fallback: local_shift_gain_s50. training_feature_influence_not_confirmed"
    payload["report"].update(
        selected_method="local_shift_gain_s50",
        final_confirmation_d2={"repeat_indices": [4, 5], "mean_original_rms": 4,
                               "mean_candidate_rms": 3, "mean_improvement_fraction": .25})
    return payload


class BnCalibrationDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.original_font = cls.app.font()
        cls.test_font = -1
        cls.font_patch = None
        fallback_font = None
        system_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
        button_font = QFont(system_font)
        button_font.setWeight(QFont.Weight.DemiBold)
        if not all(QFontMetrics(font).inFont("A") for font in (system_font, button_font)):
            font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
            cls.test_font = QFontDatabase.addApplicationFont(str(font_path))
            fallback_font = QFont("DejaVu Sans", 9)
        cls.theme = install_theme(cls.app, mode="light")
        if fallback_font is not None:
            # Offscreen may list font families without a working system-font
            # engine. Keep an owned fallback through every theme application.
            cls.font_patch = patch.object(cls.theme, "_build_font", return_value=fallback_font)
            cls.font_patch.start()
            cls.theme.apply()

    @classmethod
    def tearDownClass(cls):
        if cls.font_patch is not None:
            cls.font_patch.stop()
        if cls.test_font >= 0:
            cls.app.setFont(cls.original_font)
            QFontDatabase.removeApplicationFont(cls.test_font)

    def setUp(self):
        self.dialogs = []

    def dialog(self, **kwargs):
        options = {"context_label": "4 measurement files · Rot In 94° · selected BN"}
        options.update(kwargs)
        dialog = DrrBnCalibrationDialog(**options)
        self.dialogs.append(dialog)
        return dialog

    def wait(self, dialog):
        deadline = time.monotonic() + 10
        while dialog._worker is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertIsNone(dialog._worker, dialog.status.text())
        self.app.processEvents()

    def tearDown(self):
        for dialog in self.dialogs:
            if dialog._worker is not None:
                dialog.cancel_comparison()
                self.wait(dialog)
            dialog.close()
            dialog.deleteLater()
        self.theme.set_mode("light")
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_empty_defaults_do_not_claim_trusted_energy(self):
        dialog = self.dialog()
        settings = dialog.settings()
        self.assertEqual(settings["trusted_regions"], [])
        self.assertEqual(settings["train_regions"], [])
        self.assertEqual(settings["uncertain_regions"], [[1.94, 2.02]])
        self.assertFalse(settings["auto_apply"])
        self.assertFalse(dialog.compare_btn.isEnabled())
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertTrue(dialog.original_btn.isEnabled())

    def test_training_bandwidth_failure_is_readable_and_preview_only(self):
        payload = _payload(False)
        payload['candidates'][1]['reason'] = 'insufficient_d2_training_bandwidth'
        dialog = self.dialog(initial_payload=payload)
        self.assertIn('too narrow for the current SG window', dialog.model.item(1, 4).text())
        self.assertFalse(dialog.use_btn.isEnabled())

    def test_saved_eligible_result_never_applies_on_open(self):
        calls = []
        dialog = self.dialog(initial_payload=_payload(True), current_mode="calibrated")
        dialog.applyRequested.connect(calls.append)
        dialog.show()
        self.app.processEvents()
        self.assertEqual(calls, [])
        self.assertTrue(dialog.use_btn.isEnabled())
        self.assertEqual(dialog.model.rowCount(), 2)
        self.assertEqual(len(dialog.figure.axes), 3)
        self.assertIn("validated BN candidate", dialog.mode_label.text())

    def test_only_new_validated_result_auto_requests_application(self):
        main_thread = threading.get_ident()
        worker_threads = []
        def compare(settings, progress, log, cancellation):
            worker_threads.append(threading.get_ident())
            progress(25)
            log("Checking held-out repeats…")
            return _payload(True)
        dialog = self.dialog(run_comparison=compare)
        self.assertFalse(dialog.auto_apply.isChecked())
        dialog.auto_apply.setChecked(True)
        calls = []
        dialog.applyRequested.connect(calls.append)
        dialog.start_comparison()
        self.wait(dialog)
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(worker_threads, [main_thread])
        # The request alone is not an acknowledgement of successful application.
        self.assertIn("original BN", dialog.mode_label.text())
        dialog.set_application_state("calibrated", "Saved and applied for this group.")
        self.assertIn("validated BN candidate", dialog.mode_label.text())
        self.assertEqual(dialog.status.text(), "Saved and applied for this group.")

    def test_only_completed_new_comparison_emits_completion_for_workflow_fallback(self):
        payload = _payload(False)
        dialog = self.dialog(initial_payload=payload, run_comparison=lambda *args: payload)
        results = []
        dialog.comparisonCompleted.connect(results.append)
        dialog.set_payload(payload)
        self.assertEqual(results, [])
        dialog.start_comparison()
        self.wait(dialog)
        self.assertEqual(results, [payload])

    def test_ineligible_result_cannot_apply_via_button_or_slot(self):
        dialog = self.dialog(run_comparison=lambda *args: _payload(False))
        calls = []
        dialog.applyRequested.connect(calls.append)
        dialog.start_comparison()
        self.wait(dialog)
        dialog._request_apply()
        dialog.use_btn.click()
        self.assertEqual(calls, [])
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertIn("Original BN recommended", dialog.status.text())

    def test_editing_parameters_invalidates_use_without_mutating_saved_report(self):
        payload = _payload(True)
        dialog = self.dialog(initial_payload=payload)
        calls = []
        dialog.applyRequested.connect(calls.append)
        dialog.ranges["core_region"][1].setValue(1.98)
        self.assertFalse(dialog.use_btn.isEnabled())
        dialog._request_apply()
        self.assertEqual(calls, [])
        self.assertTrue(payload["eligible"])

    def test_trusted_range_does_not_clear_uncertain_range(self):
        dialog = self.dialog()
        dialog.trusted.setText("1.89, 1.98; 2.03-2.08")
        settings = dialog.settings()
        self.assertEqual(settings["trusted_regions"], [[1.89, 1.98], [2.03, 2.08]])
        self.assertEqual(settings["uncertain_regions"], [[1.94, 2.02]])

    def test_optional_fit_ranges_round_trip_and_empty_keeps_single_fit_range(self):
        original = {"train_regions": ((1.90, 1.91), (1.92, 1.93)),
                    "derivative_window": 51, "derivative_polyorder": 3}
        dialog = self.dialog(settings=original)
        self.assertEqual(dialog.settings()["train_regions"], [[1.90, 1.91], [1.92, 1.93]])
        self.assertEqual(dialog.train_regions.accessibleName(), "Fit energy ranges (optional)")
        self.assertIn("replace", dialog.train_regions.toolTip())
        self.assertIn("physical evidence", dialog.train_regions.toolTip())
        dialog.train_regions.clear()
        settings = dialog.settings()
        self.assertEqual(settings["train_regions"], [])
        self.assertEqual(settings["train_region"], [1.90, 1.935])
        self.assertEqual(settings["trusted_regions"], [])
        self.assertEqual(settings["uncertain_regions"], [[1.94, 2.02]])
        self.assertEqual((settings["derivative_window"], settings["derivative_polyorder"]), (51, 3))
        self.assertEqual(original["train_regions"], ((1.90, 1.91), (1.92, 1.93)))

    def test_optional_fit_ranges_are_sent_to_worker_and_editing_withdraws_eligibility(self):
        received = []
        def compare(settings, *args):
            received.append(settings)
            return _payload(True)
        dialog = self.dialog(initial_payload=_payload(True), run_comparison=compare,
                             settings={"auto_apply": False})
        dialog.train_regions.setText("1.90, 1.91; 1.92-1.93")
        self.assertFalse(dialog.use_btn.isEnabled())
        dialog.start_comparison()
        self.wait(dialog)
        self.assertEqual(received[0]["train_regions"], [[1.90, 1.91], [1.92, 1.93]])
        self.assertEqual(received[0]["train_region"], [1.90, 1.935])
        self.assertEqual(received[0]["trusted_regions"], [])
        self.assertEqual(received[0]["uncertain_regions"], [[1.94, 2.02]])
        self.assertEqual((received[0]["derivative_window"], received[0]["derivative_polyorder"]), (21, 2))

    def test_invalid_optional_fit_ranges_are_accessible_and_do_not_start_worker(self):
        calls = []
        dialog = self.dialog(run_comparison=lambda *args: calls.append(args))
        dialog.train_regions.setText("1.93, 1.90")
        dialog.start_comparison()
        self.assertIsNone(dialog._worker)
        self.assertEqual(calls, [])
        self.assertTrue(dialog.train_regions.property("fluentInvalid"))
        self.assertIn("Fit energy ranges", dialog.status.text())
        self.assertIn("low < high", dialog.train_regions.accessibleDescription())
        dialog.train_regions.setText("1.90, 1.93")
        self.assertFalse(dialog.train_regions.property("fluentInvalid"))
        self.assertEqual(dialog.train_regions.accessibleDescription(), "")

    def test_invalid_ranges_do_not_start_background_work(self):
        calls = []
        dialog = self.dialog(run_comparison=lambda *args: calls.append(args))
        dialog.protected.setText("1.8, nan")
        dialog.start_comparison()
        self.assertIsNone(dialog._worker)
        self.assertEqual(calls, [])
        self.assertIn("finite positive", dialog.status.text())
        self.assertTrue(dialog.protected.property("fluentInvalid"))

    def test_worker_failure_preserves_original_and_allows_retry(self):
        def fail(*args):
            raise ValueError("Selected BN file changed.")
        dialog = self.dialog(run_comparison=fail)
        calls = []
        dialog.applyRequested.connect(calls.append)
        dialog.start_comparison()
        self.wait(dialog)
        self.assertEqual(calls, [])
        self.assertIn("Selected BN file changed", dialog.status.text())
        self.assertTrue(dialog.compare_btn.isEnabled())
        self.assertTrue(dialog.original_btn.isEnabled())

    def test_close_cancels_cooperatively_without_applying_late_success(self):
        started = threading.Event()
        def compare(settings, progress, log, cancellation):
            started.set()
            cancellation.wait(5)
            return _payload(True)
        dialog = self.dialog(run_comparison=compare)
        calls = []
        dialog.applyRequested.connect(calls.append)
        dialog.show()
        dialog.start_comparison()
        self.assertTrue(started.wait(2))
        self.assertFalse(dialog.original_btn.isEnabled())
        dialog.start_comparison()  # no duplicate worker
        start = time.monotonic()
        dialog.close()
        self.assertLess(time.monotonic() - start, .5)
        self.wait(dialog)
        self.assertEqual(calls, [])
        self.assertFalse(dialog.isVisible())

    def test_manual_mode_and_original_restore_emit_explicit_requests(self):
        dialog = self.dialog(run_comparison=lambda *args: _payload(True), settings={"auto_apply": False})
        calls, originals = [], []
        dialog.applyRequested.connect(calls.append)
        dialog.originalRequested.connect(lambda: originals.append(True))
        dialog.start_comparison()
        self.wait(dialog)
        self.assertEqual(calls, [])
        dialog.use_btn.click()
        self.assertEqual(len(calls), 1)
        dialog.set_application_busy(True)
        dialog._request_apply()
        dialog._request_original()
        self.assertEqual(len(calls), 1)
        self.assertEqual(originals, [])
        dialog.set_error("Saving failed; current version is unchanged.")
        dialog.original_btn.click()
        self.assertEqual(originals, [True])

    def test_theme_and_compact_layout_keep_controls_available(self):
        dialog = self.dialog(initial_payload=_payload(True))
        dialog.resize(1000, 650)
        dialog.show()
        for mode in ("light", "dark"):
            self.theme.set_mode(mode)
            self.app.processEvents()
            dialog.canvas.draw()
            self.assertEqual(dialog.styleSheet(), "")
            self.assertEqual(dialog.canvas.display_theme.figure_face,
                             str(self.theme.current_theme.value("canvas_background")))
            self.assertTrue(dialog.original_btn.isVisible())
            self.assertTrue(dialog.use_btn.isVisible())
            self.assertTrue(dialog.rect().contains(dialog.use_btn.geometry()))

    def test_strength_candidates_and_optional_fit_control_fit_at_1180_by_820(self):
        dialog = self.dialog(initial_payload=_strength_preview(),
                             settings={"train_regions": [[1.90, 1.91], [1.92, 1.93]]})
        dialog.resize(1180, 820)
        dialog.show()
        for mode in ("light", "dark"):
            with self.subTest(theme=mode):
                self.theme.set_mode(mode)
                self.app.processEvents()
                self.assertEqual(dialog.styleSheet(), "")
                self.assertTrue(dialog.train_regions.isVisible())
                self.assertTrue(dialog.train_regions.isEnabled())
                self.assertTrue(QFontMetrics(dialog.train_regions.font()).inFont("A"))
                self.assertGreaterEqual(dialog.train_regions.width(), dialog.train_regions.minimumSizeHint().width())
                self.assertGreaterEqual(dialog.train_regions.width(),
                                        QFontMetrics(dialog.train_regions.font()).horizontalAdvance(dialog.train_regions.text()) + 16)
                settings_scroll = dialog.findChild(QScrollArea, "bnSettingsScroll")
                self.assertEqual(settings_scroll.horizontalScrollBar().maximum(), 0)
                for button in (dialog.compare_btn, dialog.original_btn, dialog.use_btn):
                    self.assertTrue(dialog.rect().contains(button.geometry()))
                self.assertEqual(dialog.model.rowCount(), 9)
                for row in range(1, dialog.model.rowCount()):
                    label = dialog.model.item(row, 0).text()
                    self.assertNotIn("local_shift", label)
                    self.assertIn("% correction", label)
                    metrics = QFontMetrics(dialog.table.font())
                    for line in label.splitlines():
                        self.assertGreaterEqual(dialog.table.columnWidth(0), metrics.horizontalAdvance(line))

    def test_strength_is_readable_in_summary_candidate_rows_and_final_holdout(self):
        dialog = self.dialog(initial_payload=_strength_preview())
        self.assertIn("Local shift + amplitude · 50% correction", dialog.summary.text())
        self.assertIn("Local shift + amplitude · 50% correction", dialog.final_confirmation_d2.text())
        self.assertIn("separate from the amplitude", dialog.model.horizontalHeaderItem(0).toolTip())
        self.assertEqual(dialog.model.item(1, 0).text(), "Local fringe shift\n25% correction")
        self.assertEqual(dialog.model.item(4, 0).text(), "Local fringe shift\n100% correction")
        self.assertEqual(dialog.model.item(8, 0).text(), "Local shift + amplitude\n100% correction")
        self.assertIn("Weak-feature preservation", dialog.model.item(1, 4).text())
        self.assertFalse(dialog.use_btn.isEnabled())

    def test_unrecognized_reason_codes_are_readable_while_raw_report_is_preserved(self):
        payload = _payload(False)
        reason = "selection_feature_fidelity_not_confirmed"
        payload["summary"] = f"fallback: local_shift_s25. {reason}"
        payload["candidates"][1].update(name="local_shift_s25", reason=reason)
        payload["report"]["reasons"] = [reason]
        dialog = self.dialog(initial_payload=payload)
        self.assertNotIn("_", dialog.summary.text())
        self.assertEqual(dialog.model.item(1, 4).text(), "Selection feature fidelity not confirmed")
        self.assertIn(reason, dialog.report_text.toPlainText())

    def test_selection_sensitivity_rejection_has_clear_explanation(self):
        payload = _payload(False)
        reason = "pre_final_selection_sensitivity_failed"
        payload["summary"] = f"fallback: local_shift_s25. {reason}"
        payload["candidates"][1].update(name="local_shift_s25", reason=reason)
        payload["report"]["reasons"] = [reason]
        dialog = self.dialog(initial_payload=payload)
        self.assertIn("Candidate selection changes too much when weak test features are added", dialog.summary.text())
        self.assertIn("original BN retained", dialog.model.item(1, 4).text())
        self.assertNotIn(reason, dialog.summary.text())
        self.assertIn(reason, dialog.report_text.toPlainText())

    def test_close_waits_for_application_acknowledgement_without_blocking(self):
        dialog = self.dialog(initial_payload=_payload(True))
        dialog.show()
        dialog.set_application_busy(True)
        dialog.close()
        self.assertTrue(dialog.isVisible())
        self.assertTrue(dialog._close_pending)
        dialog.set_application_state("calibrated", "Applied")
        self.assertFalse(dialog.isVisible())
        self.assertFalse(dialog._application_busy)

    def test_destroying_parent_scope_cancels_pool_worker(self):
        started, stopped = threading.Event(), threading.Event()
        def compare(settings, progress, log, cancellation):
            started.set()
            if cancellation.wait(5):
                stopped.set()
            return _payload(True)
        dialog = self.dialog(run_comparison=compare)
        dialog.start_comparison()
        self.assertTrue(started.wait(2))
        self.dialogs.remove(dialog)
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.assertTrue(stopped.wait(2))
        self.app.processEvents()

    def test_selection_gain_and_final_holdout_are_distinct_and_reasons_are_readable(self):
        payload = _payload(False)
        payload["selected_candidate"] = "local_shift_gain"
        payload["summary"] = ("fallback: local_shift_gain. correction_overlaps_uncertain_region; "
                              "correction_support_not_explicitly_trusted")
        payload["candidates"][1].update(
            name="local_shift_gain", improvement_fraction=.35,
            reason="correction_overlaps_uncertain_region; correction_support_not_explicitly_trusted")
        payload["report"].update(
            selected_method="local_shift_gain",
            reasons=["correction_overlaps_uncertain_region"],
            final_confirmation={"repeat_indices": [2, 3], "mean_original_rms": .001,
                                "mean_candidate_rms": .0008, "mean_improvement_fraction": .20},
            split={"independent_final_confirmation_available": True})
        dialog = self.dialog(initial_payload=payload)
        self.assertEqual(dialog.model.horizontalHeaderItem(1).text(), "D2E selection gain")
        self.assertIn("selection", dialog.model.horizontalHeaderItem(1).toolTip())
        self.assertEqual(dialog.model.item(1, 2).text(), "+35.0%")
        self.assertIn("0.1000% → 0.0800%", dialog.final_confirmation.text())
        self.assertIn("gain +20.0%", dialog.final_confirmation.text())
        self.assertIn("Local shift + amplitude", dialog.final_confirmation.text())
        self.assertNotIn("35.0%", dialog.final_confirmation.text())
        self.assertEqual(dialog.model.item(1, 0).text(), "Local shift + amplitude\n100% correction")
        self.assertIn("uncertain spectral feature", dialog.summary.text())
        self.assertIn("outside the trusted calibration range", dialog.model.item(1, 4).text())
        self.assertNotIn("correction_overlaps_uncertain_region", dialog.summary.text())
        self.assertIn("correction_overlaps_uncertain_region", dialog.report_text.toPlainText())
        self.assertFalse(dialog.use_btn.isEnabled())

    def test_drr_and_d2_selection_and_final_holdout_are_separate(self):
        payload = _payload(True)
        payload['report']['split']['final'] = [4, 5]
        payload['report']['final_confirmation'] = {
            'repeat_indices': [4, 5], 'mean_original_rms': .001,
            'mean_candidate_rms': .0008, 'mean_improvement_fraction': .2}
        payload['report']['final_confirmation_d2'] = {
            'repeat_indices': [4, 5], 'mean_original_rms': 4,
            'mean_candidate_rms': 3, 'mean_improvement_fraction': .25}
        dialog = self.dialog(initial_payload=payload)
        self.assertEqual(dialog.model.item(1, 1).text(), '+22.0%')
        self.assertEqual(dialog.model.item(1, 2).text(), '+35.0%')
        self.assertIn('gain +20.0%', dialog.final_confirmation.text())
        self.assertIn('gain +25.0%', dialog.final_confirmation_d2.text())
        self.assertIn('4 → 3 eV⁻²', dialog.final_confirmation_d2.text())
        self.assertIn('joint validation passed', dialog.joint_validation.text())

    def test_d2_target_can_pass_with_zero_or_negative_drr_diagnostic_gain(self):
        for gain in (0., -.01):
            with self.subTest(drr_gain=gain):
                payload = _payload(True)
                payload['candidates'][1]['improvement_fraction'] = gain
                payload['report']['final_confirmation'].update(
                    mean_candidate_rms=.001 * (1 - gain), mean_improvement_fraction=gain)
                for role in ('selection', 'final'):
                    payload['report']['joint_validation'][role]['drr_benefit_passed'] = False
                dialog = self.dialog(initial_payload=payload)
                self.assertTrue(dialog.use_btn.isEnabled())
                self.assertEqual(dialog.model.item(1, 1).text(), '+22.0%')
                self.assertEqual(dialog.model.item(1, 2).text(), f'{gain:+.1%}')
                self.assertEqual(dialog.model.item(1, 3).text(), 'Passed')
                self.assertIn('DRR constraint passed', dialog.final_confirmation.text())
                self.assertIn('D2E target confirmed', dialog.final_confirmation_d2.text())

    def test_actual_drr_constraint_failure_rejects_even_with_d2_improvement(self):
        payload = _payload(True)
        payload['report']['joint_validation']['final']['drr_noninferiority_passed'] = False
        dialog = self.dialog(initial_payload=payload)
        requests = []
        dialog.applyRequested.connect(requests.append)
        dialog._request_apply()
        self.assertEqual(requests, [])
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertEqual(dialog.model.item(1, 3).text(), 'Preview only')
        self.assertIn('DRR constraint failed', dialog.final_confirmation.text())
        self.assertIn('D2E target confirmed', dialog.final_confirmation_d2.text())

    def test_training_projection_score_is_not_shown_as_display_holdout_gain(self):
        payload = _payload(True)
        training = payload['report']['candidates']['local_shift']['parameters']['training_objective']
        training.update(candidate_score=12345678., improvement_fraction=.9999)
        dialog = self.dialog(initial_payload=payload)
        self.assertIn('production residual RMS', dialog.final_confirmation_d2.text())
        self.assertIn('gain +25.0%', dialog.final_confirmation_d2.text())
        self.assertNotIn('12345678', dialog.final_confirmation_d2.text())
        self.assertNotIn('99.99%', dialog.final_confirmation_d2.text())
        self.assertEqual(dialog.model.item(1, 1).text(), '+22.0%')
        self.assertIn('separate training metric', dialog.model.horizontalHeaderItem(1).toolTip())

    def test_previous_joint_policy_is_preview_only_and_requests_new_comparison(self):
        payload = _payload(True)
        payload['report'].update(schema_version=2, optimization_policy='joint_benefit_v1')
        payload['report']['joint_validation'].update(schema_version=1, policy='joint_benefit_v1')
        dialog = self.dialog(initial_payload=payload, current_mode='calibrated')
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertIn('older optimization policy', dialog.joint_validation.text())
        self.assertIn('Compare again', dialog.joint_validation.text())
        self.assertIn('D2E as the target and DRR as a constraint', dialog.joint_validation.text())
        self.assertIn('compare again with current weak-feature checks', dialog.mode_label.text())

    def test_previous_d2_priority_policy_remains_viewable_but_cannot_apply(self):
        payload = _payload(True)
        payload['report'].update(schema_version=3, optimization_policy='d2_priority_v1')
        payload['report']['joint_validation'].update(schema_version=2, policy='d2_priority_v1')
        dialog = self.dialog(initial_payload=payload, current_mode='calibrated')
        calls = []
        dialog.applyRequested.connect(calls.append)
        dialog._request_apply()
        self.assertEqual(calls, [])
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertEqual(len(dialog.figure.axes), 3)
        self.assertIn('older optimization policy', dialog.joint_validation.text())
        self.assertTrue(dialog.report_text.toPlainText())

    def test_d2_target_and_drr_constraint_explanation_matches_table_order(self):
        from PySide6.QtWidgets import QLabel
        dialog = self.dialog(initial_payload=_payload(True))
        explanation = dialog.findChild(QLabel, 'bnOptimizationPolicy').text()
        self.assertIn('D2E improvement is the target', explanation)
        self.assertIn('DRR must remain within its allowed change', explanation)
        self.assertEqual(dialog.model.horizontalHeaderItem(1).text(), 'D2E selection gain')
        self.assertEqual(dialog.model.horizontalHeaderItem(2).text(), 'DRR diagnostic gain')
        self.assertIn('Zero or negative gain', dialog.model.horizontalHeaderItem(2).toolTip())

    def test_legacy_drr_only_report_cannot_apply(self):
        payload = _payload(True)
        payload['report'].pop('joint_validation')
        payload['report']['schema_version'] = 1
        dialog = self.dialog(initial_payload=payload)
        calls = []
        dialog.applyRequested.connect(calls.append)
        dialog._request_apply()
        self.assertEqual(calls, [])
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertIn('preview only', dialog.joint_validation.text())

    def test_report_candidate_mapping_is_supported_without_flattened_rows(self):
        payload = _payload(True)
        payload.pop('candidates')
        dialog = self.dialog(initial_payload=payload)
        self.assertEqual(dialog.model.rowCount(), 2)
        self.assertEqual(dialog.model.item(1, 0).text(), 'Local fringe shift\n100% correction')
        self.assertEqual(dialog.model.item(1, 1).text(), '+50.0%')
        self.assertEqual(dialog.model.item(1, 2).text(), '+50.0%')
        self.assertEqual(dialog.model.item(1, 3).text(), 'Passed')

    def test_sg_change_keeps_report_preview_and_requires_new_joint_comparison(self):
        payload = _payload(True)
        dialog = self.dialog(initial_payload=payload, run_comparison=lambda *args: _payload(True))
        dialog.set_derivative_settings(11, 3)
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertEqual(dialog.settings()['derivative_window'], 11)
        self.assertEqual(dialog.settings()['derivative_polyorder'], 3)
        self.assertIn('not been validated', dialog.joint_validation.text())
        self.assertEqual(dialog.model.item(1, 3).text(), 'Preview only')
        self.assertTrue(payload['eligible'])
        self.assertEqual(payload['report']['derivative_settings']['window_length'], 21)
        self.assertEqual(len(dialog.figure.axes), 3)

    def test_managed_report_opened_at_different_sg_does_not_claim_current_validation(self):
        dialog = self.dialog(initial_payload=_payload(True), current_mode='calibrated',
                             settings={'derivative_window': 11, 'derivative_polyorder': 3})
        self.assertFalse(dialog.use_btn.isEnabled())
        self.assertIn('current SG requires comparison', dialog.mode_label.text())
        self.assertIn('preview only', dialog.joint_validation.text())

    def test_missing_d2_final_does_not_reuse_drr_or_selection(self):
        payload = _payload(False)
        payload['report']['final_confirmation'] = {
            'repeat_indices': [4, 5], 'mean_original_rms': .001,
            'mean_candidate_rms': .0008, 'mean_improvement_fraction': .2}
        dialog = self.dialog(initial_payload=payload)
        self.assertIn('No final holdout', dialog.final_confirmation_d2.text())
        self.assertNotIn('+20.0%', dialog.final_confirmation_d2.text())
        self.assertNotIn('+22.0%', dialog.final_confirmation_d2.text())

    def test_d2_preview_matches_production_processing_on_nonuniform_energy(self):
        from core.loader import DataCube
        from core.processing import apply_sg_derivative_energy
        payload = _payload(True)
        payload['energy'] = np.linspace(1.6, 2.1, 101) ** 1.01
        payload['mean_raw'] = payload['original_background'] * (
            1 + .01 * np.sin(80 * payload['energy']))
        dialog = self.dialog(initial_payload=payload)
        ratio = np.stack([payload['mean_raw'] / payload[key] - 1
                          for key in ('original_background', 'candidate_background')])
        expected, _ = apply_sg_derivative_energy(
            DataCube(payload['energy'], np.arange(2), ratio, 'Preview', 'Mean DRR', 'DRR'),
            derivative=2, window_length=21, polyorder=2)
        for index in range(2):
            np.testing.assert_allclose(dialog.figure.axes[2].lines[index].get_ydata(), expected.Z[index])
            self.assertEqual(dialog.figure.axes[0].lines[index].get_color(),
                             dialog.figure.axes[2].lines[index].get_color())

    def test_missing_or_malformed_final_result_never_reuses_selection_gain(self):
        payload = _payload(False)
        dialog = self.dialog(initial_payload=payload)
        self.assertIn("No final holdout result available", dialog.final_confirmation.text())
        self.assertNotIn("35.0%", dialog.final_confirmation.text())
        payload["report"]["final_confirmation"] = {
            "repeat_indices": [], "mean_original_rms": .001,
            "mean_candidate_rms": .0008, "mean_improvement_fraction": .20}
        dialog.set_payload(payload)
        self.assertIn("No final holdout result available", dialog.final_confirmation.text())
        self.assertNotIn("+20.0%", dialog.final_confirmation.text())
        self.assertFalse(dialog.use_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
