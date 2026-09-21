import json
import unittest
from dataclasses import replace

import numpy as np
from scipy.signal import savgol_filter

from core.loader import DataCube
from core.drr_peak_analysis import PeakAnalysisSettings, analyze_drr_peaks


class PeakAnalysisTests(unittest.TestCase):
    def cube(self, x, rows, y=None):
        rows = np.asarray(rows, float)
        return DataCube(np.asarray(x, float), np.arange(len(rows), dtype=float) if y is None else np.asarray(y, float), rows, "Gate (V)", "Synthetic", "DR/R")

    def settings(self, **kwargs):
        return replace(PeakAnalysisSettings(1.0, 1.1, 0.0, 9.0, source="raw", min_distance_mev=0.0), **kwargs)

    def test_detects_both_extrema_and_retains_selected_empty_rows(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.03)/0.004)**2) - 0.7*np.exp(-((x-1.07)/0.004)**2)
        result = analyze_drr_peaks(self.cube(x, [z, z*0, z], [-1, 0, 1]), self.settings())
        points = result["products"]["raw"]["points"]
        self.assertEqual(result["y_values"], [0.0, 1.0])
        self.assertEqual([(p["row_index"], p["polarity"]) for p in points], [(2, "peak"), (2, "dip")])
        self.assertAlmostEqual(points[0]["energy"], 1.03)
        self.assertAlmostEqual(points[1]["energy"], 1.07)
        self.assertEqual(result["y_label"], "Gate (V)")
        json.dumps(result, allow_nan=False)

    def test_range_crops_candidates_and_does_not_create_edge_peaks(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.03)/0.004)**2) + np.exp(-((x-1.08)/0.004)**2)
        points = analyze_drr_peaks(self.cube(x, [z]), self.settings(x_min=1.03, x_max=1.09, polarity="peaks"))["products"]["raw"]["points"]
        self.assertEqual([p["energy"] for p in points], [1.08])

    def test_physical_minimum_distance_on_nonuniform_axis(self):
        x = np.array([1, 1.010, 1.011, 1.012, 1.013, 1.020, 1.060, 1.090, 1.1])
        z = [0, 0, 2, 0, 1, 0, 3, 0, 0]
        points = analyze_drr_peaks(self.cube(x, [z]), self.settings(polarity="peaks", min_distance_mev=5))["products"]["raw"]["points"]
        self.assertEqual([p["energy"] for p in points], [1.011, 1.060])

    def test_descending_axes_preserve_original_row_indices(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.05)/0.004)**2)
        cube = self.cube(x[::-1], [z[::-1]]*3, [2, 1, 0])
        result = analyze_drr_peaks(cube, self.settings(y_min=0, y_max=1, polarity="peaks"))
        self.assertEqual(result["y_values"], [1., 0.])
        self.assertEqual([p["row_index"] for p in result["products"]["raw"]["points"]], [1, 2])

    def test_adjacent_tracking_and_missing_row_break(self):
        x = np.linspace(1, 1.1, 1001)
        rows = [np.exp(-((x-c)/.002)**2) for c in [1.04, 1.041, 1.042, 1.043]]
        rows[2][:] = np.nan
        points = analyze_drr_peaks(self.cube(x, rows), self.settings(polarity="peaks"))["products"]["raw"]["points"]
        self.assertEqual([p["row_index"] for p in points], [0, 1, 3])
        self.assertEqual(points[0]["track_id"], points[1]["track_id"])
        self.assertNotEqual(points[1]["track_id"], points[2]["track_id"])

    def test_competing_candidates_are_uncertain_and_disconnected(self):
        x = np.linspace(1, 1.1, 1001)
        first = np.exp(-((x-1.05)/.0004)**2)
        split = np.exp(-((x-1.048)/.0004)**2)+np.exp(-((x-1.052)/.0004)**2)
        points = analyze_drr_peaks(self.cube(x, [first, split, first]), self.settings(polarity="peaks", max_shift_mev=3))["products"]["raw"]["points"]
        self.assertEqual(len(points), 4)
        self.assertEqual(len({p["track_id"] for p in points}), 4)
        self.assertTrue(all(p["status"] == "uncertain" for p in points[:3]))

    def test_tracking_never_matches_opposite_polarities(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.05)/.004)**2)
        points = analyze_drr_peaks(self.cube(x, [z, -z]), self.settings())["products"]["raw"]["points"]
        self.assertEqual([p["polarity"] for p in points], ["peak", "dip"])
        self.assertNotEqual(points[0]["track_id"], points[1]["track_id"])

    def test_second_derivative_uses_full_spectrum_before_crop(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.05)/.006)**2)
        # The selected interval is shorter than the 21-point SG window.
        points = analyze_drr_peaks(self.cube(x, [z]), self.settings(x_min=1.047, x_max=1.053, source="second", polarity="dips"))["products"]["second"]["points"]
        self.assertEqual(len(points), 1)
        expected = savgol_filter(z, 21, 2, deriv=2, delta=.001)[50]
        self.assertAlmostEqual(points[0]["amplitude"], expected, places=6)
        self.assertAlmostEqual(points[0]["energy"], 1.05)

    def test_nonuniform_derivative_and_missing_data_are_json_safe(self):
        x = 1 + np.linspace(0, 1, 151)**1.1 * .1
        z = np.exp(-((x-1.05)/.006)**2)
        z[10] = np.nan
        cube = self.cube(x, [z, np.full(x.shape, np.nan)])
        original = cube.Z.copy()
        result = analyze_drr_peaks(cube, self.settings(source="both"))
        self.assertEqual(set(result["products"]), {"raw", "second"})
        self.assertTrue(result["products"]["second"]["points"])
        self.assertTrue(all(p["row_index"] == 0 for v in result["products"].values() for p in v["points"]))
        np.testing.assert_equal(cube.Z, original)
        json.dumps(result, allow_nan=False)

    def test_peak_limit_keeps_most_prominent(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.02)/.002)**2)+3*np.exp(-((x-1.08)/.002)**2)
        points = analyze_drr_peaks(self.cube(x, [z]), self.settings(polarity="peaks", max_peaks=1))["products"]["raw"]["points"]
        self.assertEqual([p["energy"] for p in points], [1.08])

    def test_rejects_bad_settings_axes_and_shapes(self):
        cube = self.cube(np.linspace(1, 1.1, 11), [np.zeros(11)])
        for changes in ({"x_min":1.2}, {"x_max":float("nan")}, {"y_min":10}, {"prominence":-1}, {"prominence":1.1}, {"min_distance_mev":-1}, {"max_shift_mev":float("inf")}, {"max_peaks":0}, {"source":"other"}, {"polarity":"other"}, {"source":"second", "sg_polyorder":1}, {"source":"both", "sg_window":2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                analyze_drr_peaks(cube, self.settings(**changes))
        for bad_cube in (replace(cube, energy=np.zeros(11)), replace(cube, energy=np.array([1., np.nan]+[1.1]*9)), replace(cube, Z=np.zeros((2, 11))), replace(cube, gate=np.array([np.inf])), replace(cube, energy=cube.energy.reshape(1, -1))):
            with self.subTest(cube=bad_cube), self.assertRaises(ValueError):
                analyze_drr_peaks(bad_cube, self.settings())

    def test_cancellation_stops_without_partial_result(self):
        cube = self.cube(np.linspace(1, 1.1, 11), [np.zeros(11)])
        with self.assertRaisesRegex(RuntimeError, "[Cc]ancel"):
            analyze_drr_peaks(cube, self.settings(), cancelled=lambda: True)

    def test_cancellation_is_checked_between_rows(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.05)/.004)**2)
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls >= 4

        with self.assertRaisesRegex(RuntimeError, "[Cc]ancel"):
            analyze_drr_peaks(self.cube(x, [z]*20), self.settings(source="both"), cancelled)

    def test_constant_spectra_do_not_produce_derivative_roundoff_peaks(self):
        x = np.linspace(1, 1.1, 101)
        result = analyze_drr_peaks(self.cube(x, [np.ones(101)]), self.settings(source="both"))
        self.assertEqual(result["products"]["raw"]["points"], [])
        self.assertEqual(result["products"]["second"]["points"], [])

    def test_missing_sample_does_not_create_or_bridge_a_raw_peak(self):
        x = np.linspace(1, 1.1, 11)
        z = [0, 0, 0, 1, 2, np.nan, 2, 1, 0, 0, 0]
        result = analyze_drr_peaks(self.cube(x, [z]), self.settings())
        self.assertEqual(result["products"]["raw"]["points"], [])

    def test_large_shift_starts_new_track(self):
        x = np.linspace(1, 1.1, 101)
        rows = [np.exp(-((x-c)/.004)**2) for c in [1.03, 1.07]]
        points = analyze_drr_peaks(self.cube(x, rows), self.settings(polarity="peaks"))["products"]["raw"]["points"]
        self.assertNotEqual(points[0]["track_id"], points[1]["track_id"])

    def test_second_derivative_requires_at_least_five_energy_samples(self):
        cube = self.cube(np.linspace(1, 1.1, 4), [[0, 1, 1, 0]])
        with self.assertRaises(ValueError):
            analyze_drr_peaks(cube, self.settings(source="second"))

    def test_descending_nonuniform_derivative_locates_same_feature(self):
        x = 1 + np.linspace(0, 1, 151)**1.1 * .1
        z = np.exp(-((x-1.05)/.006)**2)
        result = analyze_drr_peaks(self.cube(x[::-1], [z[::-1]]), self.settings(source="second", polarity="dips"))
        points = result["products"]["second"]["points"]
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0]["energy"], 1.05, delta=.001)

    def test_normalizes_numpy_scalars_for_json(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.05)/.004)**2)
        result = analyze_drr_peaks(self.cube(x, [z]), self.settings(max_peaks=np.int64(2), x_min=np.float32(1)))
        json.dumps(result, allow_nan=False)

    def test_duplicate_y_requires_separate_or_aggregated_sweeps(self):
        x = np.linspace(1, 1.1, 11)
        cube = self.cube(x, [np.zeros(11)]*3, [0, 1, 0])
        with self.assertRaisesRegex(ValueError, "[Rr]epeated.*separated.*aggregated"):
            analyze_drr_peaks(cube, self.settings())

    def test_equal_y_bounds_select_one_row(self):
        x = np.linspace(1, 1.1, 11)
        cube = self.cube(x, [np.zeros(11)]*3, [0, 1, 2])
        result = analyze_drr_peaks(cube, self.settings(y_min=1, y_max=1))
        self.assertEqual(result["y_values"], [1.0])

    def test_raw_detection_ignores_irrelevant_sg_constraints(self):
        x = np.linspace(1, 1.1, 101)
        z = np.exp(-((x-1.05)/.004)**2)
        cube = self.cube(x, [z])
        expected = analyze_drr_peaks(cube, self.settings())["products"]["raw"]["points"]
        for overrides in ({"sg_polyorder": 1}, {"sg_window": 2}, {"sg_window": 3, "sg_polyorder": 5}):
            with self.subTest(overrides=overrides):
                result = analyze_drr_peaks(cube, self.settings(**overrides))
                self.assertEqual(result["products"]["raw"]["points"], expected)


if __name__ == "__main__":
    unittest.main()
