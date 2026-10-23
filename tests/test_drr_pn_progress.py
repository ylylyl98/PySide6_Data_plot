"""Progress follows real stage events without a Qt application or core fits."""
import unittest

from ui_qt.drr_pn_progress import (
    ReconstructionProgress, STAGES, STAGE_PROGRESS_NOTE, TOTAL_STAGES,
)


class SignalRecorder:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(value)


class ReconstructionProgressTests(unittest.TestCase):
    def setUp(self):
        self.progress = SignalRecorder()
        self.log = SignalRecorder()
        self.reporter = ReconstructionProgress(self.progress, self.log)

    def test_complete_sequence_reserves_final_success_for_gui(self):
        self.assertEqual(TOTAL_STAGES, 19)
        self.assertEqual(STAGES, (
            "verify_inputs", "load_raw", "v12", "v20", "v21", "v24", "v29", "v31", "v32",
            "regions", "render_full", "render_first_half", "render_last_half",
            "raw_common_texture", "bounded_texture", "apply_bounded_texture",
            "save", "preview", "ready",
        ))
        for stage in STAGES:
            self.reporter.start(stage)
            self.reporter.complete(stage)
        self.assertEqual(self.progress.values, list(range(19)))
        self.assertTrue(all(type(value) is int for value in self.progress.values))
        self.assertNotIn(TOTAL_STAGES, self.progress.values)
        self.assertIn("not an estimate of time remaining", STAGE_PROGRESS_NOTE)

    def test_nested_repeated_render_events_do_not_advance_or_replace_label(self):
        self.reporter.start("regions")
        for stage in ("v35", "v36", "v45", "v50", "v57_selection"):
            self.reporter.start(stage)
            self.reporter.complete(stage)
        self.assertEqual(self.progress.values, [9])
        self.assertEqual(self.log.values, ["Checking sample-supported regions…"])
        for outer in ("render_full", "render_first_half", "render_last_half"):
            self.reporter.start(outer)
            expected_progress = self.progress.values.copy()
            expected_log = self.log.values.copy()
            for nested in ("cubic_reference", "excursion_selection", "excursion_closure",
                           "region_000_shape", "conditioned_compact_000"):
                self.reporter.event(nested, "start")
                self.reporter.event(nested, "complete")
            self.assertEqual(self.progress.values, expected_progress)
            self.assertEqual(self.log.values, expected_log)
            self.reporter.complete(outer)
        self.assertEqual(self.progress.values, [9, 10, 11, 12, 13])
        self.assertEqual(self.log.values[-1], "Checking last-half repeatability…")

    def test_source_cache_hit_skips_fits_and_keeps_later_work_pending(self):
        self.reporter.start("load_raw")
        self.reporter.event("source_preparation", "start")
        self.reporter.event("source_preparation_cache", "hit")
        self.reporter.event("source_preparation", "complete")
        self.assertEqual(self.progress.values, [1, 2, 9])
        self.assertIn("Reusing verified P/N source fits", self.log.values[-1])
        self.reporter.start("regions")
        self.reporter.complete("regions")
        self.assertEqual(self.progress.values[-1], 10)
        self.assertEqual(self.log.values[-1], "Checking sample-supported regions…")

    def test_late_known_events_neither_regress_progress_nor_current_label(self):
        self.reporter.start("preview")
        expected = (self.progress.values.copy(), self.log.values.copy())
        self.reporter.start("v12")
        self.reporter.complete("v32")
        self.reporter.event("source_preparation_cache", "hit")
        self.assertEqual((self.progress.values, self.log.values), expected)
        self.reporter.complete("preview")
        self.reporter.start("save")
        self.assertEqual(self.progress.values, [17, 18])
        self.assertEqual(self.log.values[-1], "Loading background preview…")

    def test_unknown_stage_or_event_does_not_emit(self):
        self.reporter.event("mystery", "start")
        self.reporter.event("v12", "iteration")
        self.reporter.event("source_preparation", "iteration")
        self.reporter.complete("mystery")
        self.assertEqual(self.progress.values, [])
        self.assertEqual(self.log.values, [])

    def test_saved_reuse_jumps_to_preview_without_claiming_fit_or_save(self):
        self.reporter.start("verify_inputs")
        self.reporter.complete("verify_inputs")
        self.reporter.start("preview")
        self.reporter.complete("preview")
        self.assertEqual(self.progress.values, [0, 1, 17, 18])
        self.assertEqual(self.log.values, [
            "Checking selected files and saved versions…", "Loading background preview…"])
        self.reporter.complete("ready")
        self.reporter.complete("ready")
        self.assertEqual(self.progress.values[-1], 18)

    def test_source_alias_and_direct_callbacks_share_the_same_milestones(self):
        self.reporter.event("source_preparation", "start")
        self.reporter.event("source_v12", "start")
        self.assertEqual(self.log.values[-1], "Fitting P/N source spectra (1/7)…")
        self.reporter.event("source_v12", "complete")
        self.reporter.start("v20")
        self.assertEqual(self.log.values[-1], "Fitting P/N source spectra (2/7)…")
        self.reporter.event("source_preparation", "complete")
        self.assertEqual(self.progress.values, [2, 3, 9])


if __name__ == "__main__":
    unittest.main()
