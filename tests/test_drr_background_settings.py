"""Transactional source settings and explicit review of BN suggestions."""
import os
from math import ceil
import threading
import time
import unittest
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QThreadPool, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QScrollArea, QStyleOptionViewItem

from ui_qt.drr_background_settings import BackgroundDraft, DrrBackgroundSettings
from ui_qt.theme import install_theme
from tests.ui_test_helpers import dispose_owned_window


def plan(*, ready=True, warnings=(), heterogeneous=False):
    return SimpleNamespace(ready=ready, warnings=warnings,
        rows=(SimpleNamespace(measurement_file="sample.csv", baseline_file="BN.csv" if ready else None,
            timing_basis="acquisition", time_gap_seconds=90., warnings=warnings,
            unknown_conditions=(), evidence="Nearest preceding BN"),),
        resolution=SimpleNamespace(numerical_path="heterogeneous" if heterogeneous else "common"))


def pn_plan(candidates=("Pback.csv",)):
    return SimpleNamespace(kind="p", which="all", automatic_candidates=False,
        candidate_files=candidates, warnings=(), ready=True,
        rows=(SimpleNamespace(measurement_file="sample.csv", baseline_file=candidates[0],
            candidate_files=candidates, evidence="Matching acquisition conditions", warnings=()),))


def batch_plan(*, warnings=()):
    """Four verified runs: nine measurements use BN001, six use BN002."""
    prefix = "YZ383_p2n1_9T_1.67KREF_730nmc_0p3sx10_RotIn94deg_"
    rows, batches = [], []
    for run_index, count in enumerate((3, 3, 3, 6)):
        baseline = "nested/" + prefix + f"BNbackground_PIXIS_{1 if run_index < 3 else 2:03}.csv"
        files = []
        for index in range(count):
            name = "nested/" + prefix + f"TG−1BG=0_PIXIS_rep{run_index + 1:02}_{index + 1:03}.csv"
            files.append(name)
            rows.append(SimpleNamespace(measurement_file=name, baseline_file=baseline,
                timing_basis="capture_started_utc", time_gap_seconds=90., warnings=(),
                unknown_conditions=(), evidence="Unique per-file time suggestion agrees with the run",
                batch_id=f"batch-{run_index}", batch_status="consensus", anomaly=False))
        batches.append(SimpleNamespace(batch_id=f"batch-{run_index}", measurement_files=tuple(files),
            baseline_file=baseline, status="consensus", verified_run=True, run_id=f"run-{run_index}",
            experiment_id="experiment", warnings=(), anomaly_measurements=(),
            evidence="Verified capture run; all individual suggestions agree", capture_start=1000.,
            capture_end=1100., candidate_files=(baseline,), aggregate_time_gaps=((baseline, count * 90.),)))
    return SimpleNamespace(ready=True, warnings=warnings, rows=tuple(rows), batches=tuple(batches),
        resolution=SimpleNamespace(numerical_path="heterogeneous"))


class BackgroundSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        self.dialogs = []
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)

    def dialog(self, current=None, **kwargs):
        dialog = DrrBackgroundSettings(current=current or BackgroundDraft(kind="self", which="last"),
            folder=".", measurements=("sample.csv",), pool=self.pool, **kwargs)
        self.dialogs.append(dialog)
        return dialog

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.reject()
        self.pool.waitForDone(10000)
        self.app.processEvents()
        for dialog in self.dialogs:
            dispose_owned_window(dialog)

    def test_draft_is_immutable_and_cancel_keeps_applied_selection(self):
        original = BackgroundDraft(kind="self", which="last")
        dialog = self.dialog(original)
        with self.assertRaises(FrozenInstanceError):
            original.which = "first"
        dialog.self_frames.setCurrentIndex(dialog.self_frames.findData("first"))
        self.assertEqual(dialog.draft.which, "first")
        self.assertEqual(original.which, "last")
        self.assertIn("last", dialog.applied_summary.text())
        self.assertIsNone(dialog.selection)
        dialog.reject()
        self.assertIsNone(dialog.selection)

    def test_source_and_frame_change_applies_only_on_outer_button(self):
        selected = []
        def choose(_dialog, kind, files, title):
            selected.append((kind, files, title))
            return ("p1.csv", "p2.csv")
        dialog = self.dialog(select_files=choose)
        dialog._source_widgets["pn"].click()
        self.assertEqual(dialog.draft.kind, "p")
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData("common"))
        dialog.pn_files_btn.click()
        self.assertEqual(selected[0][0], "p")
        self.assertEqual(dialog.draft.files, ("p1.csv", "p2.csv"))
        self.assertIn("equally", dialog.repeat_note.text())
        self.assertIsNone(dialog.selection)
        dialog.apply_btn.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.selection.kind, "p")

    def test_initial_draft_is_distinct_from_the_currently_applied_background(self):
        applied = BackgroundDraft(kind="self", which="last")
        dialog = self.dialog(applied, initial_draft=BackgroundDraft(kind="bn"))
        self.assertEqual(dialog.current, applied)
        self.assertEqual(dialog.draft.kind, "bn")
        self.assertIn("Self", dialog.applied_summary.text())
        self.assertIn("BN", dialog.draft_summary.text())
        self.assertIn("86400", dialog.bn_timing_note.text())
        self.assertIn("60", dialog.bn_timing_note.text())
        self.assertIn("physical confidence", dialog.bn_timing_note.text())

    def test_p_n_and_composite_are_explicit_separate_methods(self):
        dialog = self.dialog(BackgroundDraft(kind="p", files=("p.csv",)))
        self.assertEqual([dialog.pn_method.itemData(i) for i in range(3)], ["p", "n", "pn"])
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData("n"))
        self.assertEqual(dialog.draft.kind, "n")
        self.assertEqual(dialog.draft.files, ())
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData("p"))
        self.assertEqual(dialog.draft.files, ("p.csv",))
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData("pn"))
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertTrue(dialog.pn_files_btn.isHidden())

    def test_new_pn_roles_start_with_an_empty_selected_matching_pool(self):
        for kind in ("p", "n", "pn"):
            dialog = self.dialog(BackgroundDraft(kind=kind), configure_composite=lambda *_: None)
            self.assertEqual(dialog.pn_assignment_method.currentData(), "matched")
            self.assertEqual(dialog.draft.pn_candidates, ())
            self.assertFalse(dialog.pn_matching.match_btn.isEnabled())
            self.assertFalse(dialog.apply_btn.isEnabled())
            self.assertIn("Choose background files", dialog.status.text())
            self.assertFalse(dialog.pn_composite_btn.isHidden())

    def test_changing_to_an_unconfigured_pn_role_starts_with_no_candidate_files(self):
        dialog = self.dialog(BackgroundDraft(kind="p", files=("Pback.csv",)))
        self.assertEqual(dialog.pn_assignment_method.currentData(), "common")
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData("n"))
        self.assertEqual(dialog.pn_assignment_method.currentData(), "matched")
        self.assertEqual(dialog.draft.pn_candidates, ())
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData("p"))
        self.assertEqual(dialog.pn_assignment_method.currentData(), "common")
        self.assertEqual(dialog.draft.files, ("Pback.csv",))

    def test_common_pn_draft_survives_visiting_the_empty_matching_pool(self):
        original = BackgroundDraft(kind="n", files=("Nback.csv",), which="first")
        dialog = self.dialog(original)
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData("matched"))
        self.assertEqual(dialog.draft.pn_candidates, ())
        self.assertEqual(dialog.draft.files, ())
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData("common"))
        self.assertEqual(dialog.draft, original)

    def test_pn_pool_change_is_private_and_cannot_expand_to_the_experiment(self):
        original = BackgroundDraft(kind="p", files=("common.csv",))
        dialog = self.dialog(original)
        dialog._pn_candidates_changed(("Pback.csv",))
        self.assertEqual(dialog.draft.pn_candidates, ("Pback.csv",))
        self.assertEqual(dialog.draft.files, ())
        self.assertEqual(dialog.pn_assignment_method.currentData(), "matched")
        self.assertEqual(dialog.current, original)
        dialog._pn_candidates_changed(None)
        self.assertEqual(dialog.draft.pn_candidates, ())
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.set_bn_busy(True)
        dialog._pn_candidates_changed(("other.csv",))
        self.assertEqual(dialog.draft.pn_candidates, ())

    def test_legacy_automatic_pn_plan_keeps_only_assigned_backgrounds_and_requires_rematch(self):
        matching = pn_plan(("Pback.csv", "previously_discovered.csv"))
        matching.automatic_candidates = True
        dialog = self.dialog(BackgroundDraft(kind="p", pn_plan=matching))
        self.assertIs(dialog.draft.pn_plan, matching)
        self.assertEqual(dialog.draft.pn_candidates, ("Pback.csv",))
        self.assertEqual(dialog.pn_assignment_method.currentData(), "matched")
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertIn("again", dialog.status.text())

    def test_unresolved_legacy_automatic_pn_plan_does_not_select_discovered_files(self):
        matching = pn_plan()
        matching.automatic_candidates = True
        matching.rows[0].baseline_file = ""
        dialog = self.dialog(BackgroundDraft(kind="p", pn_plan=matching))
        self.assertEqual(dialog.draft.pn_candidates, ())
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_pn_result_keeps_selected_files_including_unusable_choices(self):
        selected = ("unused.csv", "Pback.csv")
        dialog = self.dialog(BackgroundDraft(kind="p", pn_candidates=selected))
        matching = pn_plan((os.path.abspath("Pback.csv"),))
        dialog.set_pn_plan(matching)
        self.assertEqual(dialog.draft.pn_candidates, selected)
        self.assertIs(dialog.draft.pn_plan, matching)

    def test_pn_result_outside_the_selected_pool_is_not_adopted(self):
        dialog = self.dialog(BackgroundDraft(kind="p", pn_candidates=("Pback.csv",)))
        dialog.set_pn_plan(pn_plan(("other.csv",)))
        self.assertEqual(dialog.draft.pn_candidates, ("Pback.csv",))
        self.assertIsNone(dialog.draft.pn_plan)
        self.assertIn("outside the selected files", dialog.status.text())
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_pn_frame_change_rematches_only_the_selected_pool_after_returning_to_events(self):
        selected = ("Pback.csv",)
        dialog = self.dialog(BackgroundDraft(kind="p", pn_candidates=selected))
        dialog.set_pn_plan(pn_plan(selected))
        with patch.object(dialog.pn_matching, "match") as match:
            dialog.pn_frames.setCurrentIndex(dialog.pn_frames.findData("first"))
            self.assertIsNone(dialog.draft.pn_plan)
            match.assert_not_called()
            self.app.processEvents()
            match.assert_called_once_with()
        self.assertEqual(dialog.draft.pn_candidates, selected)

    def test_deferred_pn_frame_rematch_is_ignored_after_clearing_the_pool(self):
        dialog = self.dialog(BackgroundDraft(kind="p", pn_candidates=("Pback.csv",)))
        with patch.object(dialog.pn_matching, "match") as match:
            dialog.pn_frames.setCurrentIndex(dialog.pn_frames.findData("first"))
            dialog._pn_candidates_changed(())
            self.app.processEvents()
            match.assert_not_called()

    def test_child_result_is_private_until_outer_apply(self):
        version = SimpleNamespace(version_id="version-123")
        def configure(_dialog, draft):
            return replace(draft, kind="pn", version=version)
        dialog = self.dialog(BackgroundDraft(kind="pn"), configure_composite=configure)
        dialog.pn_composite_btn.click()
        self.assertIs(dialog.draft.version, version)
        self.assertIsNone(dialog.selection)
        self.assertEqual(dialog.current.version, None)
        dialog.apply_btn.click()
        self.assertIs(dialog.selection.version, version)

    def test_generated_composite_keeps_the_original_p_matching_draft_separate(self):
        original = BackgroundDraft(kind="p", pn_candidates=("Pback.csv",), pn_plan=pn_plan())
        version = SimpleNamespace(version_id="new-version")
        dialog = self.dialog(original, configure_composite=lambda _dialog, draft:
            replace(draft, kind="pn", version=version))
        dialog.pn_composite_btn.click()
        self.assertEqual(dialog.draft.kind, "pn")
        self.assertEqual(dialog.pn_assignment_method.currentData(), "common")
        self.assertIsNone(dialog.draft.pn_candidates)
        self.assertIsNone(dialog.selection)
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData("p"))
        self.assertEqual(dialog.draft, original)
        self.assertEqual(dialog.pn_assignment_method.currentData(), "matched")

    def test_generated_composite_can_return_to_its_previous_matching_draft(self):
        matching = pn_plan(("saved-composite.csv",))
        matching.kind = "pn"
        original = BackgroundDraft(kind="pn", pn_candidates=matching.candidate_files, pn_plan=matching)
        version = SimpleNamespace(version_id="new-version")
        dialog = self.dialog(original, configure_composite=lambda _dialog, draft: replace(draft, version=version))
        dialog.pn_composite_btn.click()
        self.assertEqual(dialog.pn_assignment_method.currentData(), "common")
        dialog.pn_assignment_method.setCurrentIndex(dialog.pn_assignment_method.findData("matched"))
        self.assertEqual(dialog.draft, original)

    def test_cancelled_child_picker_preserves_the_draft(self):
        original = BackgroundDraft(kind="n", files=("old.csv",), which="first")
        dialog = self.dialog(original, select_files=lambda *_: None)
        dialog.pn_files_btn.click()
        self.assertEqual(dialog.draft, original)

    def test_saved_composite_distinguishes_input_averaging_from_output_frame(self):
        version = SimpleNamespace(version_id="v1", p_which="all", n_which="all")
        original = BackgroundDraft(kind="pn", files=("saved.csv",), which="last", version=version)
        dialog = self.dialog(original)
        self.assertIn("Saved composite · single-frame background", dialog.pn_files_summary.text())
        self.assertEqual(dialog.pn_source_frames_note.text(), "P: all frames averaged; N: all frames averaged")
        self.assertTrue(dialog.pn_frames.isHidden())
        self.assertFalse(dialog.pn_source_frames_note.isHidden())
        self.assertEqual(dialog.draft, original)
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData("p"))
        self.assertFalse(dialog.pn_frames.isHidden())
        self.assertTrue(dialog.pn_source_frames_note.isHidden())
        dialog.pn_method.setCurrentIndex(dialog.pn_method.findData("pn"))
        self.assertEqual(dialog.draft, original)
        dialog.apply_btn.click()
        self.assertEqual(dialog.selection, original)

    def test_restored_composite_does_not_invent_unavailable_source_frame_choices(self):
        original = BackgroundDraft(kind="pn", files=("Processed Data/DRR/PN Backgrounds/saved.csv",), which="last")
        dialog = self.dialog(original)
        self.assertIn("single-frame background", dialog.pn_files_summary.text())
        self.assertEqual(dialog.pn_source_frames_note.text(), "P/N source averaging: see saved version")
        self.assertTrue(dialog.pn_frames.isHidden())
        self.assertEqual(dialog.draft, original)

    def test_candidate_pool_is_distinct_from_explicit_common_bn(self):
        calls = []
        def choose(_dialog, kind, *_):
            calls.append(kind)
            return ("candidate.csv",)
        dialog = self.dialog(BackgroundDraft(kind="bn"), select_files=choose)
        dialog.bn_candidates_btn.click()
        self.assertEqual(dialog.draft.bn_candidates, ("candidate.csv",))
        self.assertEqual(dialog.draft.files, ())
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.bn_common_btn.click()
        self.assertEqual(calls, ["bn_candidates", "bn"])
        self.assertEqual(dialog.draft.files, ("candidate.csv",))
        self.assertIsNone(dialog.draft.bn_plan)
        self.assertTrue(dialog.apply_btn.isEnabled())

    def test_bn_warning_must_be_reviewed_without_automatically_applying(self):
        dialog = self.dialog(BackgroundDraft(kind="bn"))
        mapping = plan(warnings=("Time comes from file modification metadata",))
        dialog.set_bn_plan(mapping)
        self.assertEqual(dialog.bn_model.rowCount(), 1)
        self.assertIn("sample.csv", dialog.bn_model.item(0, 0).text())
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertFalse(dialog.bn_review.isHidden())
        dialog.bn_review.setChecked(True)
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIsNone(dialog.selection)
        dialog.apply_btn.click()
        self.assertIs(dialog.selection.bn_plan, mapping)
        self.assertTrue(dialog.selection.bn_warnings_acknowledged)

    def test_exact_tie_cannot_be_overridden_with_review_checkbox(self):
        dialog = self.dialog(BackgroundDraft(kind="bn"))
        dialog.set_bn_plan(plan(ready=False, warnings=("Two candidates have the same acquisition time",)))
        dialog.bn_review.setChecked(True)
        self.assertFalse(dialog.bn_review.isEnabled())
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog._apply()
        self.assertIsNone(dialog.selection)

    def test_confirmed_unchanged_plan_keeps_review_and_is_reusable(self):
        mapping = plan(warnings=("Unknown temperature",))
        current = BackgroundDraft(kind="bn", bn_plan=mapping, bn_warnings_acknowledged=True)
        dialog = self.dialog(current)
        self.assertTrue(dialog.bn_review.isChecked())
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIs(dialog.draft.bn_plan, mapping)

    def test_unknown_conditions_are_reviewable_even_without_warning_strings(self):
        mapping = plan()
        mapping.rows[0].unknown_conditions = ("temperature",)
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertIn("temperature", dialog.bn_review.toolTip())
        dialog.bn_review.setChecked(True)
        self.assertTrue(dialog.apply_btn.isEnabled())

    def test_reused_restricted_plan_displays_its_original_candidate_pool(self):
        mapping = plan()
        mapping.automatic_candidates = False
        mapping.candidate_files = ("BN.csv", "other_BN.csv")
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        self.assertTrue(dialog.bn_restricted.isChecked())
        self.assertEqual(dialog.draft.bn_candidates, mapping.candidate_files)
        self.assertIn("2 candidate", dialog.bn_candidates_label.text())

    def test_multiple_files_are_a_candidate_pool_and_cannot_become_one_common_bn(self):
        dialog = self.dialog(BackgroundDraft(kind="bn"), select_files=lambda *_: ("one_BN.csv", "two_BN.csv"))
        dialog.bn_common_btn.click()
        self.assertEqual(dialog.draft.files, ())
        self.assertIn("one common BN", dialog.status.text())
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_main_bn_view_has_two_columns_and_keeps_details_and_strategy_collapsed(self):
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=plan(warnings=("Modification time fallback",))))
        dialog.show()
        self.app.processEvents()
        self.assertEqual(dialog.bn_summary_model.columnCount(), 2)
        self.assertEqual([dialog.bn_summary_model.headerData(i, Qt.Horizontal) for i in range(2)], ["Coverage", "BN file"])
        self.assertEqual([dialog.bn_method.itemText(i) for i in range(2)], ["Match BN by batch", "One BN for all"])
        self.assertTrue(dialog.bn_summary_table.isVisible())
        self.assertFalse(dialog.bn_table.isVisible())
        self.assertTrue(dialog.bn_auto.isHidden())
        self.assertTrue(dialog.bn_restricted.isHidden())
        self.assertFalse(dialog.group_strategy.isVisible())
        self.assertTrue(dialog.bn_details_panel.isHidden())
        self.assertTrue(dialog.draft_summary.isHidden())
        self.assertIn("timing", dialog.bn_plan_status.text())
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_scope_menu_changes_candidates_without_using_them_as_a_common_bn(self):
        dialog = self.dialog(BackgroundDraft(kind="bn"), select_files=lambda *_: ("one_BN.csv", "two_BN.csv"))
        dialog.bn_selected_scope_action.trigger()
        self.assertEqual(dialog.draft.bn_candidates, ("one_BN.csv", "two_BN.csv"))
        self.assertEqual(dialog.draft.files, ())
        self.assertEqual(dialog.bn_method.currentData(), "matched")
        dialog.bn_current_scope_action.trigger()
        self.assertIsNone(dialog.draft.bn_candidates)
        self.assertEqual(dialog.bn_candidates_label.text(), "Current experiment")

    def test_one_bn_method_is_separate_and_restores_the_matching_draft(self):
        mapping = plan()
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping), select_files=lambda *_: ("common_BN.csv",))
        dialog.bn_method.setCurrentIndex(dialog.bn_method.findData("common"))
        self.assertIsNone(dialog.draft.bn_plan)
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.bn_common_btn.click()
        self.assertEqual(dialog.draft.files, ("common_BN.csv",))
        dialog.bn_method.setCurrentIndex(dialog.bn_method.findData("matched"))
        self.assertIs(dialog.draft.bn_plan, mapping)
        self.assertEqual(dialog.draft.files, ())

    def test_match_details_contain_evidence_with_readable_timing_names(self):
        mapping = plan(warnings=("Modification time fallback",))
        mapping.rows[0].timing_basis = "filesystem_mtime_proxy"
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        self.assertIn("File modification time", dialog.bn_details_text.toPlainText())
        self.assertIn("90 s", dialog.bn_details_text.toPlainText())
        self.assertIn("Nearest preceding BN", dialog.bn_details_text.toPlainText())
        dialog.bn_details_button.click()
        self.assertFalse(dialog.bn_details_panel.isHidden())
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_long_basenames_wrap_fully_and_fit_rows_after_resize(self):
        mapping = batch_plan()
        background = os.path.basename(mapping.batches[0].baseline_file)
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        dialog.show()
        for width, height in ((860, 700), (720, 540)):
            dialog.resize(width, height)
            for _ in range(4):
                self.app.processEvents()
            self.assertEqual(dialog.bn_summary_model.item(0, 1).text(), background)
            self.assertEqual(dialog.bn_summary_table.textElideMode(), Qt.ElideNone)
            delegate = dialog.bn_summary_table.itemDelegate()
            for column in range(2):
                index = dialog.bn_summary_model.index(0, column)
                name = delegate._normalize_text(index.data())
                option = QStyleOptionViewItem()
                option.initFrom(dialog.bn_summary_table)
                option.rect = dialog.bn_summary_table.visualRect(index)
                delegate.initStyleOption(option, index)
                text_layout = delegate._layout_text(name, option.font, delegate._text_width(option))
                self.assertGreaterEqual(text_layout.lineCount(), 1)
                self.assertEqual(sum(text_layout.lineAt(i).textLength() for i in range(text_layout.lineCount())), len(name))
                needed = ceil(text_layout.boundingRect().height()) + 2 * delegate.VERTICAL_PADDING
                self.assertGreaterEqual(dialog.bn_summary_table.rowHeight(0), needed, (width, column))
                self.assertIn("nested", dialog.bn_summary_model.item(0, column).toolTip())
            scroll = dialog.findChildren(QScrollArea)[0]
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            self.assertEqual(scroll.verticalScrollBar().maximum(), 0)
        dialog.resize(860, 700)
        for _ in range(4):
            self.app.processEvents()
        self.assertLess(dialog.bn_summary_table.height(), 250)

    def test_batch_summary_keeps_all_measurements_and_distinct_verified_run_counts(self):
        mapping = batch_plan()
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        self.assertEqual(dialog.bn_summary_model.rowCount(), 2)
        self.assertEqual(dialog.bn_summary_model.item(0, 0).text(), "9 measurements\n3 verified runs")
        self.assertEqual(dialog.bn_summary_model.item(1, 0).text(), "6 measurements\n1 verified run")
        self.assertEqual(dialog.bn_model.rowCount(), 15)
        self.assertIn("15 measurements retained", dialog.bn_plan_status.text())
        self.assertIn("4 verified runs", dialog.bn_plan_status.text())
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIsNone(dialog.selection)

    def test_expanding_details_keeps_all_full_file_pairs_readable(self):
        mapping = batch_plan()
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        dialog.resize(720, 540)
        dialog.show()
        dialog.bn_details_button.click()
        for _ in range(4):
            self.app.processEvents()
        self.assertTrue(dialog.bn_table.isVisible())
        self.assertEqual(dialog.bn_model.rowCount(), 15)
        delegate = dialog.bn_table.itemDelegate()
        for row, source in enumerate(mapping.rows):
            for column, source_name in enumerate((source.measurement_file, source.baseline_file)):
                index = dialog.bn_model.index(row, column)
                self.assertEqual(index.data(), os.path.basename(source_name))
                option = QStyleOptionViewItem()
                option.initFrom(dialog.bn_table)
                option.rect = dialog.bn_table.visualRect(index)
                delegate.initStyleOption(option, index)
                layout = delegate._layout_text(index.data(), option.font, delegate._text_width(option))
                self.assertEqual(sum(layout.lineAt(i).textLength() for i in range(layout.lineCount())), len(index.data()))
                self.assertGreaterEqual(dialog.bn_table.rowHeight(row),
                    ceil(layout.boundingRect().height()) + 2 * delegate.VERTICAL_PADDING)

    def test_unknown_runs_and_legacy_plans_are_explicit_unverified_singles(self):
        for with_batches in (False, True):
            mapping = batch_plan()
            if with_batches:
                mapping.batches = tuple(SimpleNamespace(batch_id=f"single-{i}",
                    measurement_files=(row.measurement_file,), baseline_file=row.baseline_file,
                    status="unverified", verified_run=False, run_id="", experiment_id="",
                    warnings=(), evidence="No verified run", anomaly_measurements=(),
                    capture_start=None, capture_end=None) for i, row in enumerate(mapping.rows))
            else:
                del mapping.batches
            dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
            self.assertEqual(dialog.bn_summary_model.rowCount(), 2)
            self.assertEqual(dialog.bn_summary_model.item(0, 0).text(), "9 measurements\n9 unverified singles")
            self.assertEqual(dialog.bn_model.rowCount(), 15)
            self.assertNotIn("verified run", dialog.bn_plan_status.text())

    def test_conflicting_batch_keeps_individual_suggestions_without_majority_selection(self):
        mapping = batch_plan(warnings=("Run has conflicting per-file BN suggestions",))
        batch = mapping.batches[-1]
        batch.status, batch.baseline_file = "conflict", ""
        batch.anomaly_measurements = (batch.measurement_files[-1],)
        mapping.rows[-1].baseline_file = mapping.rows[0].baseline_file
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        self.assertIn("Conflict", dialog.bn_summary_model.item(1, 0).text())
        self.assertIn("Different BNs within this batch", dialog.bn_summary_model.item(1, 1).text())
        self.assertIn("1 batch conflict", dialog.bn_plan_status.text())
        self.assertIn("Review", dialog.bn_details_button.text())
        self.assertTrue(dialog.bn_details_panel.isHidden())
        self.assertEqual(dialog.bn_model.rowCount(), 15)
        self.assertEqual(dialog.bn_model.item(14, 1).text(), os.path.basename(mapping.rows[0].baseline_file))
        self.assertTrue(dialog.bn_review.isEnabled())
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.bn_review.setChecked(True)
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIn("keeps the reviewed per-file BN choices", dialog.status.text())
        dialog._apply()
        self.assertIs(dialog.selection.bn_plan, mapping)
        self.assertEqual(dialog.selection.bn_plan.rows[-1].baseline_file, mapping.rows[0].baseline_file)

    def test_unresolved_batch_cannot_apply_and_details_explain_run_evidence(self):
        mapping = batch_plan()
        mapping.batches[0].status = "unresolved"
        mapping.batches[0].baseline_file = ""
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        self.assertIn("unresolved", dialog.bn_plan_status.text())
        self.assertIn("No unambiguous BN", dialog.bn_summary_model.item(0, 1).text())
        self.assertIn("Batch status: unresolved", dialog.bn_details_text.toPlainText())
        self.assertIn("Last capture start (UTC)", dialog.bn_details_text.toPlainText())
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.bn_review.setChecked(True)
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_separate_abnormal_batches_do_not_become_a_single_conflict_batch(self):
        mapping = batch_plan()
        for batch in mapping.batches[:2]:
            batch.status, batch.baseline_file = "conflict", ""
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping))
        self.assertEqual(dialog.bn_summary_model.rowCount(), 4)
        for index in (0, 1):
            self.assertEqual(dialog.bn_summary_model.item(index, 0).text(),
                "3 measurements\n1 verified run\nConflict")
            self.assertEqual(dialog.bn_summary_model.item(index, 1).text(), "Per-file suggestions need review")
        self.assertFalse(dialog.apply_btn.isEnabled())
        dialog.bn_review.setChecked(True)
        self.assertTrue(dialog.apply_btn.isEnabled())

    def test_candidate_changes_clear_batch_result_and_previous_warning_approval(self):
        mapping = batch_plan(warnings=("Unknown conditions",))
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=mapping, bn_warnings_acknowledged=True),
            select_files=lambda *_: ("other_BN.csv",))
        dialog.bn_selected_scope_action.trigger()
        self.assertIsNone(dialog.draft.bn_plan)
        self.assertFalse(dialog.draft.bn_warnings_acknowledged)
        self.assertEqual(dialog.bn_summary_model.rowCount(), 0)
        self.assertEqual(dialog.bn_model.rowCount(), 0)
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_empty_busy_and_error_states_do_not_offer_stale_results(self):
        dialog = self.dialog(BackgroundDraft(kind="bn"))
        self.assertTrue(dialog.bn_summary_table.isHidden())
        self.assertIn("Match BN", dialog.status.text())
        dialog.set_bn_busy(True)
        self.assertFalse(dialog.bn_match_btn.isEnabled())
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertIn("Matching BN", dialog.status.text())
        dialog.set_bn_busy(False)
        dialog.set_bn_error("No compatible BN files in this experiment.")
        self.assertIn("No compatible BN", dialog.status.text())
        self.assertFalse(dialog.apply_btn.isEnabled())

    def test_replacing_calibrated_bn_clears_the_previous_recipe(self):
        stale = BackgroundDraft(kind="bn", files=("old_BN.csv",), calibration={"old": True},
                                version=object(), regions=(object(),))
        for action in ("common", "candidates", "scope", "match", "plan"):
            dialog = self.dialog(stale, select_files=lambda *_: ("new_BN.csv",),
                request_bn_match=lambda *_: plan())
            if action == "common":
                dialog.bn_common_btn.click()
            elif action == "candidates":
                dialog.bn_candidates_btn.click()
            elif action == "scope":
                dialog.bn_restricted.setChecked(True)
            elif action == "match":
                dialog.bn_match_btn.click()
            else:
                dialog.set_bn_plan(plan())
            self.assertIsNone(dialog.draft.calibration, action)
            self.assertIsNone(dialog.draft.version, action)
            self.assertEqual(dialog.draft.regions, (), action)
            if action == "common":
                dialog.apply_btn.click()
                self.assertEqual(dialog.selection.files, ("new_BN.csv",))
                self.assertIsNone(dialog.selection.calibration)

    def test_heterogeneous_bn_and_group_calibration_cannot_be_fixed(self):
        for current in (BackgroundDraft(kind="bn", bn_plan=plan(heterogeneous=True), fixed=True),
                        BackgroundDraft(kind="p", files=("p.csv",), calibration=object(), fixed=True),
                        BackgroundDraft(kind="bn", files=(".drr_bn_calibration/frozen.csv",), fixed=True),
                        BackgroundDraft(kind="fitted", fixed=True),
                        BackgroundDraft(kind="p", fixed=True)):
            dialog = self.dialog(current)
            self.assertFalse(dialog.draft.fixed)
            self.assertFalse(dialog.group_strategy.model().item(1).isEnabled())

    def test_common_bn_mapping_can_be_kept_fixed(self):
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=plan()))
        self.assertTrue(dialog.group_strategy.model().item(1).isEnabled())
        dialog.group_strategy.setCurrentIndex(1)
        self.assertTrue(dialog.draft.fixed)

    def test_bn_calibration_is_available_only_for_a_common_bn_source(self):
        for draft, available in ((BackgroundDraft(kind="self"), False),
                (BackgroundDraft(kind="p", files=("p.csv",)), False),
                (BackgroundDraft(kind="bn", bn_plan=plan(heterogeneous=True)), False),
                (BackgroundDraft(kind="bn", files=("BN.csv",)), True),
                (BackgroundDraft(kind="bn", bn_plan=plan()), True)):
            dialog = self.dialog(draft, configure_calibration=lambda *_: None)
            self.assertEqual(dialog.calibration_btn.isEnabled(), available)

    def test_custom_import_is_advanced_and_child_save_does_not_apply(self):
        dialog = self.dialog(import_custom=lambda _dialog, draft: replace(draft, kind="custom", files=("custom.csv",)))
        self.assertEqual(len(dialog.source_group.buttons()), 4)
        self.assertTrue(dialog.advanced_panel.isHidden())
        dialog.advanced_button.click()
        self.assertFalse(dialog.advanced_panel.isHidden())
        dialog.custom_btn.click()
        self.assertEqual(dialog.draft.kind, "custom")
        self.assertFalse(any(button.isChecked() for button in dialog.source_group.buttons()))
        self.assertIsNone(dialog.selection)
        self.assertTrue(dialog.apply_btn.isEnabled())

    def test_fitted_uses_current_settings_and_optional_configuration(self):
        dialog = self.dialog()
        dialog._source_widgets["fitted"].click()
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIn("current fit settings", dialog.draft_summary.text())
        self.assertIsNone(dialog.selection)

    def test_switching_to_fitted_retains_the_existing_fit_settings(self):
        existing = {"mode": "current configured sample fit"}
        dialog = self.dialog(BackgroundDraft(kind="self", which="last", fitted_result=existing))
        dialog._source_widgets["fitted"].click()
        self.assertEqual(dialog.draft.fitted_result, existing)
        dialog.apply_btn.click()
        self.assertEqual(dialog.selection.fitted_result, existing)

    def test_busy_match_blocks_duplicate_actions_and_cancel_cancels_worker(self):
        started, release = threading.Event(), threading.Event()
        captured = []
        def matching(_folder, _measurements, _candidates, stopped, **_kwargs):
            captured.append(stopped)
            started.set()
            release.wait(5)
            return plan()
        with patch("ui_qt.drr_background_settings._build_bn_plan", matching):
            dialog = self.dialog(BackgroundDraft(kind="bn"))
            dialog.bn_match_btn.click()
            self.assertTrue(started.wait(2))
            self.assertFalse(dialog.bn_match_btn.isEnabled())
            self.assertFalse(dialog.apply_btn.isEnabled())
            self.assertFalse(dialog._source_widgets["self"].isEnabled())
            cancellation = dialog._worker.cancellation
            dialog.reject()
            self.assertTrue(captured[0].is_set())
            self.assertTrue(cancellation.cancelled)
            release.set()
            self.pool.waitForDone(5000)
            self.app.processEvents()
            self.assertIsNone(dialog.draft.bn_plan)
            self.assertIsNone(dialog.selection)

    def test_successful_worker_reviews_plan_without_applying(self):
        mapping = plan()
        with patch("ui_qt.drr_background_settings._build_bn_plan", lambda *args, **kwargs: mapping):
            dialog = self.dialog(BackgroundDraft(kind="bn"))
            dialog.bn_match_btn.click()
            deadline = time.monotonic() + 5
            while dialog._worker is not None and time.monotonic() < deadline:
                self.app.processEvents()
                QTest.qWait(5)
            self.assertIsNone(dialog._worker)
            self.assertIs(dialog.draft.bn_plan, mapping)
            self.assertTrue(dialog.apply_btn.isEnabled())
            self.assertIsNone(dialog.selection)

    def test_confirmed_cache_reuses_review_without_applying_or_rebuilding(self):
        mapping = plan(warnings=("Previously accepted modification-time fallback",))
        cache = SimpleNamespace(find=lambda *_args, **_kwargs: mapping)
        with patch("core.drr_bn_matching.build_bn_match_plan") as build:
            dialog = self.dialog(BackgroundDraft(kind="bn"), confirmed_bn_cache=cache)
            dialog.bn_match_btn.click()
            deadline = time.monotonic() + 5
            while dialog._worker is not None and time.monotonic() < deadline:
                self.app.processEvents()
                QTest.qWait(5)
            self.assertIsNone(dialog._worker)
            build.assert_not_called()
            self.assertIs(dialog.draft.bn_plan, mapping)
            self.assertTrue(dialog.draft.bn_warnings_acknowledged)
            self.assertTrue(dialog.apply_btn.isEnabled())
            self.assertIsNone(dialog.selection)


if __name__ == "__main__":
    unittest.main()
