"""Manual BN selection remains discoverable when automatic matching fails."""
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QScrollArea

from tests.ui_test_helpers import dispose_owned_window
from ui_qt.drr_background_settings import BackgroundDraft, DrrBackgroundSettings
from ui_qt.theme import install_theme


def _plan(*, matches=0, warnings=()):
    rows = tuple(SimpleNamespace(measurement_file=f"sample_{i}.csv",
        baseline_file="sample_BNbackground_001.csv" if i < matches else "",
        timing_basis="capture_started_utc", time_gap_seconds=60., warnings=(),
        unknown_conditions=(), evidence="No compatible BN" if i >= matches else "Nearest BN")
        for i in range(2))
    return SimpleNamespace(ready=matches == 2, rows=rows, warnings=warnings,
                           resolution=SimpleNamespace(numerical_path="common"))


class BnManualEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_theme(cls.app, mode="light")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dialogs = []

    def tearDown(self):
        for dialog in self.dialogs:
            dispose_owned_window(dialog)
        self.tmp.cleanup()

    def dialog(self, draft=None, **kwargs):
        current = kwargs.pop("current", BackgroundDraft(kind="self", which="last"))
        dialog = DrrBackgroundSettings(current=current,
            initial_draft=draft or BackgroundDraft(kind="bn"),
            measurements=("sample_0.csv", "sample_1.csv"), folder=self.tmp.name, **kwargs)
        dialog.setAttribute(Qt.WA_DontShowOnScreen)
        self.dialogs.append(dialog)
        dialog.show()
        self.app.processEvents()
        return dialog

    def test_no_match_has_a_visible_direct_manual_action_without_opening_a_picker(self):
        picker = Mock(return_value=None)
        dialog = self.dialog(select_files=picker)
        dialog.set_bn_plan(_plan())
        self.app.processEvents()
        manual = dialog.findChild(QPushButton, "drr_bn_choose_manually")
        self.assertIsNotNone(manual)
        self.assertTrue(manual.isVisible())
        self.assertTrue(manual.isEnabled())
        self.assertEqual(manual.text(), "Choose BN manually…")
        self.assertFalse(dialog.apply_btn.isEnabled())
        self.assertTrue(dialog.bn_common_btn.isHidden())
        picker.assert_not_called()

    def test_partial_matches_and_worker_failure_keep_the_manual_action_available(self):
        picker = Mock(return_value=None)
        for mapping, error in ((_plan(matches=1, warnings=("One BN remains unresolved",)), ""),
                               (None, "No compatible BN files found.")):
            with self.subTest(error=error):
                dialog = self.dialog(select_files=picker)
                if mapping is not None:
                    dialog.set_bn_plan(mapping)
                else:
                    dialog.set_bn_error(error)
                self.assertTrue(dialog.bn_manual_btn.isVisible())
                self.assertTrue(dialog.bn_manual_btn.isEnabled())
                self.assertFalse(dialog.apply_btn.isEnabled())
        picker.assert_not_called()

    def test_cancelled_manual_picker_preserves_error_plan_and_review_draft(self):
        mapping = _plan(matches=1, warnings=("Review partial suggestions",))
        draft = BackgroundDraft(kind="bn", bn_plan=mapping, bn_warnings_acknowledged=True,
                                bn_candidates=("candidate.csv",))
        picker = Mock(return_value=None)
        dialog = self.dialog(draft, select_files=picker)
        dialog.set_bn_error("Matching could not complete.")
        original, status = dialog.draft, dialog.status.text()
        QTest.mouseClick(dialog.bn_manual_btn, Qt.LeftButton)
        self.assertIs(dialog.draft, original)
        self.assertEqual(dialog.status.text(), status)
        self.assertEqual(dialog.bn_method.currentData(), "matched")
        self.assertIsNone(dialog.selection)
        self.assertEqual(picker.call_count, 1)

    def test_manual_pick_stays_a_draft_until_outer_apply_and_restores_matching_recipe(self):
        mapping = _plan(matches=2, warnings=("Previously reviewed timing warning",))
        draft = BackgroundDraft(kind="bn", bn_plan=mapping, bn_warnings_acknowledged=True)
        picker = Mock(return_value=("chosen_BNbackground_002.csv",))
        dialog = self.dialog(draft, select_files=picker)
        QTest.mouseClick(dialog.bn_manual_btn, Qt.LeftButton)
        self.assertEqual(picker.call_args.args[1], "bn")
        self.assertEqual(dialog.draft.files, ("chosen_BNbackground_002.csv",))
        self.assertIsNone(dialog.draft.bn_plan)
        self.assertEqual(dialog.bn_method.currentData(), "common")
        self.assertTrue(dialog.bn_manual_btn.isHidden())
        self.assertTrue(dialog.bn_common_btn.isVisible())
        self.assertIsNone(dialog.selection)
        self.assertEqual(dialog.current.kind, "self")
        dialog.bn_method.setCurrentIndex(dialog.bn_method.findData("matched"))
        self.assertIs(dialog.draft.bn_plan, mapping)
        self.assertTrue(dialog.draft.bn_warnings_acknowledged)
        dialog.bn_method.setCurrentIndex(dialog.bn_method.findData("common"))
        dialog.apply_btn.click()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.assertEqual(dialog.selection.files, ("chosen_BNbackground_002.csv",))

    def test_no_match_to_manual_common_enables_apply_but_cancel_keeps_old_applied_plan(self):
        applied_plan = _plan(matches=2)
        current = BackgroundDraft(kind="bn", bn_plan=applied_plan)
        failed = _plan(matches=0)
        dialog = self.dialog(BackgroundDraft(kind="bn", bn_plan=failed), current=current,
                             select_files=Mock(return_value=("manual_BN.csv",)))
        self.assertFalse(dialog.apply_btn.isEnabled())
        QTest.mouseClick(dialog.bn_manual_btn, Qt.LeftButton)
        self.assertEqual(dialog.draft.files, ("manual_BN.csv",))
        self.assertIsNone(dialog.draft.bn_plan)
        self.assertTrue(dialog.apply_btn.isEnabled())
        self.assertIsNone(dialog.selection)
        dialog.reject()
        self.assertIs(dialog.current.bn_plan, applied_plan)
        self.assertIsNone(dialog.selection)

    def test_busy_and_missing_picker_states_disable_manual_selection(self):
        dialog = self.dialog(select_files=Mock(return_value=("BN.csv",)))
        dialog.set_bn_busy(True)
        self.assertTrue(dialog.bn_manual_btn.isVisible())
        self.assertFalse(dialog.bn_manual_btn.isEnabled())
        dialog.set_bn_busy(False)
        self.assertTrue(dialog.bn_manual_btn.isEnabled())
        unavailable = self.dialog()
        self.assertFalse(unavailable.bn_manual_btn.isEnabled())

    def test_common_mode_has_only_its_existing_manual_picker(self):
        dialog = self.dialog(BackgroundDraft(kind="bn", files=("BN.csv",)),
                             select_files=Mock(return_value=None))
        self.assertTrue(dialog.bn_manual_btn.isHidden())
        self.assertTrue(dialog.bn_common_btn.isVisible())
        self.assertEqual(sum(button.isVisible() for button in (dialog.bn_manual_btn, dialog.bn_common_btn)), 1)

    def test_failure_action_and_apply_fit_at_minimum_size(self):
        dialog = self.dialog(select_files=Mock(return_value=None))
        dialog.set_bn_error("No compatible BN files found for this group.")
        dialog.resize(720, 540)
        for _ in range(3):
            self.app.processEvents()
        manual = dialog.bn_manual_btn
        self.assertTrue(manual.isVisible())
        self.assertGreaterEqual(manual.width(), manual.sizeHint().width())
        self.assertTrue(dialog.apply_btn.isVisible())
        scroll = dialog.findChildren(QScrollArea)[0]
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
        self.assertEqual(scroll.verticalScrollBar().maximum(), 0)


if __name__ == "__main__":
    unittest.main()
