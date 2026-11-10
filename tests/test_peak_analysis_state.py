from copy import deepcopy
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core.peak_analysis_state import analysis_state, invalidate_fit
from ui_qt.quick.peak_analysis_workflow import analyze_dataset, export_analysis, send_overlay


def dataset():
    return dict(kind="PL", cube=SimpleNamespace(gate=[0]), row=0, analysis_scope="spectrum",
                settings={}, branches=[dict(id="a", enabled=True), dict(id="b", enabled=False)],
                candidates={"points": [1]}, result={"points": [
                    dict(branch_id="a", row_index=0, status="accepted", energy=1., fit_center=1.1,
                         fit_center_fixed=True),
                    dict(branch_id="b", row_index=0, status="accepted", energy=2.)],
                    "fits": [{"row_index": 0}]})


class PeakAnalysisStateTests(unittest.TestCase):
    def test_invalidate_preserves_candidates_constraints_and_source(self):
        d = dataset()
        previous, candidates = d["result"], d["candidates"]
        invalidate_fit(d)
        self.assertTrue(previous["fits"])
        self.assertEqual(previous["points"][0]["fit_center"], 1.1)
        self.assertIs(d["candidates"], candidates)
        self.assertFalse(d["result"]["fits"])
        self.assertTrue(d["result"]["points"][0]["fit_center_fixed"])
        self.assertNotIn("fit_center", d["result"]["points"][0])
        self.assertEqual(d["overlay_position"], "detected")

    def test_fit_uses_reviewed_enabled_points_without_mutating_source(self):
        d = dataset()
        original = deepcopy(d["result"])
        def fit(ds, result, settings, cancelled, progress):
            self.assertEqual([p["branch_id"] for p in result["points"]], ["a"])
            self.assertEqual(result["points"][0]["energy"], 1.)
            result["points"][0]["fit_status"] = "ok"
            return result
        with patch("core.pl_peak_analysis.fit", side_effect=fit):
            fitted = analyze_dataset(d, "fit")
        self.assertEqual(d["result"], original)
        self.assertEqual(fitted["result"]["points"][0]["fit_status"], "ok")
        self.assertEqual(len(fitted["result"]["points"]), 2)
        state = analysis_state(fitted)
        self.assertEqual((state["total_count"], state["review_count"], state["fitted_count"]), (2, 1, 1))

    def test_dirty_blocks_fit_filter_export_and_overlay(self):
        d = dataset()
        d["detection_dirty"] = True
        self.assertFalse(analysis_state(d)["can_fit"])
        self.assertFalse(analysis_state(d)["can_export"])
        for action in ("fit", "filter", "auto-filter"):
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, "Find peaks again"):
                analyze_dataset(d, action)
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "pending changes"):
                export_analysis(folder, d)
            with self.assertRaisesRegex(ValueError, "pending changes"):
                send_overlay(d, folder, "server")
        d["detection_dirty"] = False
        d["filters_pending"] = True
        self.assertFalse(analysis_state(d)["can_fit"])
        self.assertFalse(analysis_state(d)["can_export"])
        with self.assertRaisesRegex(ValueError, "Apply filters"):
            analyze_dataset(d, "fit")


if __name__ == "__main__":
    unittest.main()
