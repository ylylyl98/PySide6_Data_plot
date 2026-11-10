"""Power law area selection keeps the legacy series intact."""
from types import SimpleNamespace
import unittest

from core.power_law_analysis import evaluate_ranges, peak_series


class PowerLawAreaTests(unittest.TestCase):
    def test_explicit_full_area_changes_law_and_preserves_legacy_default(self):
        fits = [SimpleNamespace(power_uw=float(p), components=[SimpleNamespace(
            area=float(p), fit_area_full=float(p*p), status="ok")]) for p in (1, 2, 4, 8, 16)]
        results = {"Power": fits}
        ranges = [dict(channel="Power", peak=1, lower=1., upper=16.)]
        old = evaluate_ranges(results, {}, ranges)[0]
        full = evaluate_ranges(results, {}, ranges, "full")[0]
        self.assertAlmostEqual(old["alpha"], 1.)
        self.assertAlmostEqual(full["alpha"], 2.)
        self.assertEqual(old["intensity_metric"], "area")
        self.assertEqual(full["intensity_metric"], "fit_area_full")
        self.assertEqual(peak_series(results)[0]["intensity"].tolist(), [1., 2., 4., 8., 16.])

    def test_missing_full_values_not_replaced_with_legacy(self):
        fits = [SimpleNamespace(power_uw=float(p), components=[SimpleNamespace(area=float(p), status="ok")]) for p in (1, 2, 4, 8, 16)]
        with self.assertRaises(ValueError):
            evaluate_ranges({"Power": fits}, {}, [dict(channel="Power", peak=1, lower=1, upper=8)], "full")

    def test_kept_law_captures_original_inputs(self):
        import numpy as np
        from ui_qt.quick.power_analysis import PowerAnalysisMixin
        fits = [SimpleNamespace(power_uw=float(p), components=[SimpleNamespace(area=float(p), status="ok")]) for p in (1, 2, 4, 8, 16)]
        results = {"Power": fits}
        law = evaluate_ranges(results, {}, [dict(channel="Power", peak=1, lower=2, upper=16)])[0]
        owner = SimpleNamespace(_law_timer=SimpleNamespace(isActive=lambda: False), _running=set(), _law_fits=[law],
            _analysis_current_key="key", _analysis_cache={"key": {"fingerprint": "abc"}},
            _law_series=peak_series(results), _products={"records": {}}, _analysis_results=results,
            _law_state={}, _analysis_save=lambda: None)
        PowerAnalysisMixin._analysis_keep_law(owner)
        kept = owner._analysis_cache["key"]["lawHistory"][0]
        self.assertEqual([row["inclusion_reason"] for row in kept["input_rows"]],
                         ["outside interval", "included", "included", "included", "included"])
        self.assertEqual(kept["input_fingerprint"], "abc")
        fits[1].components[0].area = 999.
        self.assertEqual(kept["input_rows"][1]["intensity"], 2.)
