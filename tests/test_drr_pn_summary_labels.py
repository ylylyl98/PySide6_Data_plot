"""Sidebar labels distinguish a saved spectrum from its P/N input frames."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.drr_background_regions import DrrBackgroundRegion
from core.drr_sources import DrrMeasurementAssignment
from ui_qt.controllers_drr import DrrController
from ui_qt.drr_background_settings import BackgroundDraft
from ui_qt.drr_background_settings_workflow import FRAME_LABELS, _root_key


class _Control:
    def __init__(self, text=""):
        self.value = text

    def currentText(self):
        return self.value


class _Label:
    def __init__(self):
        self.value = ""
        self.tooltip = ""

    def setText(self, text):
        self.value = text

    def text(self):
        return self.value

    def setToolTip(self, text):
        self.tooltip = text

    def toolTip(self):
        return self.tooltip


class _SummaryController(DrrController):
    """Keep unrelated widget updates and missing-file checks outside this test."""

    def _drr_missing_sources(self, sources):
        return []

    def _update_drr_baseline_controls(self):
        pass

    def _repopulate_drr_yaxis(self):
        pass


class PnSummaryLabelTests(unittest.TestCase):
    def setUp(self):
        self.pn = "Processed Data/DRR/PN Backgrounds/PN_saved.csv"
        self.bn = "reference_BNbackground.csv"
        self.owner = SimpleNamespace(
            current_folder="summary-label-experiment",
            drr_selected_files=["sample.csv"],
            drr_baseline_files_manual=[self.pn],
            drr_baseline_files_found=[],
            drr_baseline_combo=_Control("External"),
            drr_baseline_combine_combo=_Control(FRAME_LABELS["last"]),
            drr_baseline_summary=_Label(),
            drr_measurement_summary=_Label(),
            _drr_assignments=(), _drr_assignments_automatic=False,
            _drr_external_auto_selected=False, _drr_background_regions=(),
            _drr_background_intent=None, _drr_background_intent_root=None,
        )
        self.controller = _SummaryController(self.owner)

    def intent(self, *, files=None, version=None, kind="pn"):
        self.owner._drr_background_intent = BackgroundDraft(
            kind=kind, files=tuple(files or [self.pn]), which="last", version=version)
        self.owner._drr_background_intent_root = _root_key(self.owner)

    def recipe(self):
        owner = self.owner
        return (
            tuple(owner.drr_selected_files), tuple(owner.drr_baseline_files_manual),
            tuple(owner.drr_baseline_files_found), owner._drr_assignments,
            owner._drr_assignments_automatic, owner._drr_external_auto_selected,
            owner._drr_background_regions, owner.drr_baseline_combo.currentText(),
            owner.drr_baseline_combine_combo.currentText(), owner._drr_background_intent,
        )

    def refresh(self):
        before = self.recipe()
        with patch("core.drr_pn_store.DrrPnStore", side_effect=AssertionError("GUI opened P/N store")), \
                patch("core.drr_composite_background.read_composite_background",
                      side_effect=AssertionError("GUI read composite")):
            self.controller._update_drr_selection_labels()
        self.assertEqual(self.recipe(), before, "Label refresh changed the processing recipe")
        return self.owner.drr_baseline_summary

    def assert_saved_composite(self, label):
        self.assertIn("Saved composite · single-frame background", label.text())
        self.assertNotIn("mode: last", label.text())
        self.assertIn("one generated background spectrum", label.toolTip())

    def test_applied_version_reports_recorded_source_frames_without_changing_last_adapter(self):
        version = SimpleNamespace(p_which="all", n_which="first")
        for name in (self.pn, "saved_reference.csv"):
            with self.subTest(name=name):
                self.owner.drr_baseline_files_manual = [name]
                self.intent(files=[name], version=version)
                label = self.refresh()
                self.assert_saved_composite(label)
                self.assertIn("P: all frames averaged; N: first frame", label.text())
                self.assertEqual(self.owner.drr_baseline_combine_combo.currentText(), FRAME_LABELS["last"])

    def test_restored_managed_path_without_version_does_not_invent_source_averaging(self):
        self.intent()
        self.assertIsNone(self.owner._drr_background_intent.version)
        label = self.refresh()
        self.assert_saved_composite(label)
        self.assertIn("P/N source averaging: see saved version", label.text())
        self.assertNotIn("P: last frame", label.text())

    def test_legacy_managed_path_without_intent_is_still_a_saved_composite(self):
        self.owner.drr_baseline_files_manual = [self.pn.replace("/", "\\")]
        label = self.refresh()
        self.assert_saved_composite(label)
        self.assertIn("P/N source averaging: see saved version", label.text())

    def test_common_automatic_composite_keeps_measurement_count(self):
        self.owner.drr_selected_files = ["sample.csv", "repeat.csv"]
        self.owner._drr_assignments = tuple(
            DrrMeasurementAssignment(name, "External", (self.pn,), "last")
            for name in self.owner.drr_selected_files)
        self.owner._drr_external_auto_selected = True
        self.intent()
        label = self.refresh()
        self.assert_saved_composite(label)
        self.assertIn("all 2 measurements → one average", label.text())
        self.assertNotIn("background files (last)", label.text())

    def test_mixed_mapping_describes_pn_and_bn_frames_independently(self):
        self.owner.drr_selected_files = ["sample.csv", "repeat.csv"]
        self.owner.drr_baseline_files_manual = [self.pn, self.bn]
        self.owner._drr_assignments = (
            DrrMeasurementAssignment("sample.csv", "External", (self.pn,), "last"),
            DrrMeasurementAssignment("repeat.csv", "External", (self.bn,), "first"),
        )
        self.owner._drr_assignments_automatic = True
        self.intent(files=[self.pn, self.bn], kind="custom")
        label = self.refresh()
        self.assertIn("all 2 measurements → one average", label.text())
        self.assertIn(f"{self.pn} [Saved composite · single-frame background]", label.toolTip())
        self.assertIn(f"{self.bn} [first]", label.toolTip())
        self.assertNotIn(f"{self.pn} [last]", label.toolTip())

    def test_region_composite_does_not_relabel_default_raw_bn(self):
        self.owner.drr_baseline_files_manual = [self.bn]
        self.owner._drr_background_regions = (
            DrrBackgroundRegion(1.61, 1.65, (self.pn,), "last"),
            DrrBackgroundRegion(1.66, 1.7, (self.bn,), "first"),
        )
        label = self.refresh()
        self.assertIn("mode: last", label.text())
        self.assertIn("PN_saved.csv [Saved composite · single-frame background]", label.toolTip())
        self.assertIn(f"{self.bn} [first]", label.toolTip())

    def test_stale_version_metadata_is_not_attributed_to_a_different_managed_path(self):
        self.intent(files=["Processed Data/DRR/PN Backgrounds/old.csv"],
                    version=SimpleNamespace(p_which="first", n_which="first"))
        label = self.refresh()
        self.assert_saved_composite(label)
        self.assertIn("P/N source averaging: see saved version", label.text())
        self.assertNotIn("P: first frame", label.text())

    def test_raw_bn_and_p_only_keep_their_meaningful_frame_modes(self):
        for kind, name in (("bn", self.bn), ("p", "reference_Pbackground.csv")):
            for which, expected in (("last", "last"), ("first", "first"), ("all", "all frames")):
                with self.subTest(kind=kind, which=which):
                    self.owner.drr_baseline_files_manual = [name]
                    self.owner.drr_baseline_combine_combo.value = FRAME_LABELS[which]
                    self.intent(files=[name], kind=kind)
                    label = self.refresh()
                    self.assertIn(f"1 files (mode: {expected})", label.text())
                    self.assertNotIn("Saved composite", label.text())

    def test_self_reference_keeps_its_measurement_frame_description(self):
        for frame in ("first", "last"):
            with self.subTest(frame=frame):
                self.owner.drr_baseline_combo.value = f"Self ({frame} frame)"
                label = self.refresh()
                self.assertEqual(label.text(), f"Using {frame} frame")


if __name__ == "__main__":
    unittest.main()
