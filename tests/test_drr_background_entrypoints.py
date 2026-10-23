"""Ordinary background selection keeps application atomic around the modal picker.

The lightweight owner avoids persistent QSettings and a full MainWindow. Gate
assessment and file averaging use real temporary CSVs; only presentation,
loading dispatch and the modal picker are replaced.
"""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_sources import DrrMeasurementAssignment
from core.load_cancellation import LoadCancellation
from core.loader import build_external_baseline
from ui_qt.controllers_drr import DrrController
from ui_qt.drr_region_backgrounds import frame_method


class _Control:
    """Only the value/signal methods used by the controller under test."""

    def __init__(self, text="", checked=False):
        self.text = text
        self.checked = checked
        self.blocked = False
        self.enabled = True
        self.focused = False

    def currentText(self):
        return self.text

    def setCurrentText(self, text):
        self.text = text

    def isChecked(self):
        return self.checked

    def setChecked(self, checked):
        self.checked = bool(checked)

    def blockSignals(self, blocked):
        previous, self.blocked = self.blocked, bool(blocked)
        return previous

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setFocus(self):
        self.focused = True


class BackgroundEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.write("old.csv", -5, (90, 110))
        self.write("first.csv", -5, (10, 30))
        self.write("second.csv", -5, (80, 100, 120, 140))
        self.write("other_gate.csv", 5, (40, 60))
        self.write("sample.csv", 0, (200, 220))
        self.write("new_sample.csv", 0, (250, 270))
        region = DrrBackgroundRegion(1.61, 1.65, ("old.csv",), "last")
        assignment = DrrMeasurementAssignment(
            "sample.csv", "External", ("old.csv",), "first", regions=(region,))
        self.pending_auto = SimpleNamespace(cancelled=threading.Event())
        self.pending_pn = SimpleNamespace(cancellation=LoadCancellation())
        self.owner = SimpleNamespace(
            current_folder=str(self.root), drr_selected_files=["sample.csv"],
            drr_baseline_files_manual=["old.csv"], drr_baseline_files_found=["found.csv"],
            drr_baseline_combo=_Control("External"),
            drr_baseline_combine_combo=_Control("First frame from each file, then average"),
            drr_pin_baseline_chk=_Control(checked=False), drr_edit_baselines_btn=_Control(),
            _drr_background_regions=(region,), _drr_assignments=(assignment,),
            _drr_assignments_automatic=True, _drr_baseline_user_selected=False,
            _drr_external_auto_selected=True, _drr_background_guess="previous guess",
            _drr_external_request=self.pending_auto, _drr_pn_apply_request=self.pending_pn,
            _is_closing=False, _status=Mock(), _show_error=Mock(),
            _invalidate_drr_for_background_selection=Mock(),
        )
        self.controller = DrrController(self.owner)
        self.owner.drr_controller = self.controller
        self.loads = []
        self.owner._start_load = Mock(side_effect=self.capture_load)
        for name in ("_update_drr_selection_labels", "_update_drr_baseline_controls"):
            replacement = patch.object(DrrController, name, autospec=True)
            replacement.start()
            self.addCleanup(replacement.stop)

    def write(self, name, gate, values):
        rows = [f"{gate},{gate},{value},{value},{value}" for value in values]
        (self.root / name).write_text("\n".join(["Vbg,Vtg,740,760,780", *rows]) + "\n")

    def capture_load(self, mode):
        self.loads.append((mode, tuple(self.owner.drr_baseline_files_manual),
                           frame_method(self.owner), self.owner._drr_background_regions,
                           self.owner._drr_assignments))

    def recipe(self):
        owner = self.owner
        return (tuple(owner.drr_baseline_files_manual), tuple(owner.drr_baseline_files_found),
                owner._drr_background_regions, owner._drr_assignments,
                owner._drr_assignments_automatic, owner._drr_baseline_user_selected,
                owner._drr_background_guess, owner.drr_baseline_combo.currentText(),
                owner.drr_baseline_combine_combo.currentText(), owner.drr_pin_baseline_chk.isChecked())

    def choose(self, files, *, accepted=True, during_modal=None):
        previous = list(self.owner.drr_baseline_files_manual)

        def picker(controller, *, title, selected, baseline_mode, independent_background=False):
            self.assertIs(controller, self.controller)
            self.assertEqual(selected, previous)
            self.assertTrue(baseline_mode)
            self.assertFalse(independent_background, "Ordinary backgrounds must keep measurement-grid checks")
            self.assertTrue(self.pending_auto.cancelled.is_set())
            self.assertTrue(self.pending_pn.cancellation.cancelled)
            self.owner._drr_source_dialog_accepted = accepted
            if during_modal is not None:
                during_modal()
            return list(files)

        with patch.object(DrrController, "_open_drr_source_dialog", autospec=True, side_effect=picker) as opened:
            self.controller._edit_drr_baselines_dialog()
        opened.assert_called_once()
        self.assertTrue(self.owner.drr_edit_baselines_btn.focused)

    def test_cancel_keeps_recipe_frames_and_keep_setting(self):
        self.owner.drr_pin_baseline_chk.setChecked(True)
        self.owner._drr_assignments_automatic = False
        before = self.recipe()
        # The real picker returns its original selection when cancelled.
        self.choose(["old.csv"], accepted=False)
        self.assertEqual(self.recipe(), before)
        self.owner._start_load.assert_not_called()
        self.owner._invalidate_drr_for_background_selection.assert_not_called()
        self.owner._show_error.assert_not_called()

    def test_accepted_constant_gate_files_form_one_equal_file_average_and_clear_old_recipe(self):
        for keep in (False, True):
            with self.subTest(keep=keep):
                self.owner.drr_pin_baseline_chk.setChecked(keep)
                self.choose(["first.csv", "second.csv"])
                self.assertEqual(self.owner.drr_baseline_files_manual, ["first.csv", "second.csv"])
                self.assertEqual(self.owner.drr_baseline_files_found, [])
                self.assertEqual(self.owner._drr_background_regions, ())
                self.assertEqual(self.owner._drr_assignments, ())
                self.assertFalse(self.owner._drr_assignments_automatic)
                self.assertTrue(self.owner._drr_baseline_user_selected)
                self.assertFalse(self.owner._drr_external_auto_selected)
                self.assertIsNone(self.owner._drr_background_guess)
                self.assertEqual(self.owner.drr_baseline_combo.currentText(), "External")
                self.assertEqual(self.owner.drr_pin_baseline_chk.isChecked(), keep)
                self.assertEqual(self.loads[-1], ("DRR", ("first.csv", "second.csv"), "all", (), ()))
                # Unequal frame counts distinguish file averaging from pooling
                # every frame: ((10+30)/2 + (80+100+120+140)/4)/2 = 65.
                result = build_external_baseline(str(self.root), self.loads[-1][1], which=self.loads[-1][2])
                np.testing.assert_allclose(result["I0"], [65., 65., 65.], rtol=0, atol=0)
        self.assertEqual(self.owner._start_load.call_count, 2)
        self.owner._show_error.assert_not_called()

    def test_different_constant_gate_values_leave_old_recipe_and_plot_unchanged(self):
        before = self.recipe()
        self.choose(["first.csv", "other_gate.csv"])
        self.assertEqual(self.recipe(), before)
        self.owner._start_load.assert_not_called()
        self.owner._invalidate_drr_for_background_selection.assert_not_called()
        self.owner._show_error.assert_called_once()
        self.assertIn("different constant gate values", self.owner._show_error.call_args.args[0])

    def test_measurement_change_during_picker_does_not_apply_selection(self):
        before = self.recipe()
        self.choose(["first.csv"], during_modal=lambda: setattr(self.owner, "drr_selected_files", ["new_sample.csv"]))
        self.assertEqual(self.owner.drr_selected_files, ["new_sample.csv"])
        self.assertEqual(self.recipe(), before)
        self.owner._start_load.assert_not_called()
        self.owner._invalidate_drr_for_background_selection.assert_not_called()
        self.assertIn("current selection changed", self.owner._status.call_args.args[0])

    def test_accepting_empty_selection_clears_recipe_and_invalidates_old_plot(self):
        self.owner.drr_pin_baseline_chk.setChecked(True)
        self.choose([])
        self.assertEqual(self.owner.drr_baseline_files_manual, [])
        self.assertEqual(self.owner.drr_baseline_files_found, [])
        self.assertEqual(self.owner._drr_background_regions, ())
        self.assertEqual(self.owner._drr_assignments, ())
        self.assertFalse(self.owner._drr_assignments_automatic)
        self.assertFalse(self.owner.drr_pin_baseline_chk.isChecked())
        self.owner._start_load.assert_not_called()
        self.owner._invalidate_drr_for_background_selection.assert_called_once()
        self.assertIn("cleared", self.owner._invalidate_drr_for_background_selection.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
