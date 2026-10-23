"""Synthetic local-feature contracts, not physical truth or source validation."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_reconstruction.feature_detection import detect_local_features


def profiles(e, p, n=None):
    if n is None:
        n = p
    stationary = np.array([[p, n], [p, n]])
    return stationary, np.zeros_like(stationary)


class LocalFeatureDetectionTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1., 2., 2001)

    def run_profiles(self, p, n=None, noise=.2):
        stationary, dynamic = profiles(self.e, p, n)
        return detect_local_features(self.e, stationary, dynamic, [noise, noise], [noise, noise])

    def test_weak_peak_next_to_strong_keeps_own_center_and_observations(self):
        e = self.e
        p = 100*np.exp(-((e-1.5)/.0015)**2)+10*np.exp(-((e-1.507)/.0015)**2)
        arrays, report = self.run_profiles(p)
        positive = [r for r in report["regions"] if r["branch"] == "stationary" and r["polarity"] == 1]
        self.assertEqual(len(positive), 2)
        np.testing.assert_allclose([r["center_ev"] for r in positive], [1.5, 1.507], atol=.0005)
        self.assertTrue(all(len(r["observations"]) == 2 for r in positive))
        self.assertTrue(all(r["shares_nominal_SG21_window_with_neighbor"] for r in positive))
        self.assertGreater(positive[1]["observations"][0]["prominence_snr"], 5)
        self.assertLess(positive[0]["core_bounds_eV"][1], positive[1]["center_ev"])
        self.assertLess(positive[0]["bounds_eV"][1], positive[1]["bounds_eV"][1])
        self.assertTrue(arrays["feature_core_mask"].any())

    def test_processed_unresolved_pair_is_not_invented_as_two_extrema(self):
        from core.processing import _sg_derivative_energy_rows
        e = np.linspace(1., 2., 1001)
        raw = np.exp(-((e-1.498)/.002)**2)+np.exp(-((e-1.502)/.002)**2)
        processed = _sg_derivative_energy_rows(raw[None], e, 21, 2, 2)[0]
        stationary, dynamic = profiles(e, processed)
        _, report = detect_local_features(e, stationary, dynamic, [50., 50.], [50., 50.])
        central = [r for r in report["regions"] if r["branch"] == "stationary" and r["polarity"] == -1 and abs(r["center_ev"]-1.5) < .006]
        self.assertEqual(len(central), 1)
        self.assertGreater(report["resolution"]["SG21_full_span_ev"], .004)
        self.assertIn("not deblended", report["resolution"]["interpretation"])

    def test_stationary_opposite_directions_are_ambiguous_not_selected(self):
        p = 12*np.exp(-((self.e-1.5)/.004)**2)
        _, report = self.run_profiles(p, -p)
        centered = [row for row in report["regions"] if abs(row["center_ev"]-1.5) < .001]
        self.assertTrue(centered)
        for row in centered:
            self.assertEqual(row["status"], "ambiguous_stationary")
            self.assertEqual(row["source_directions"]["P"]["direction"], "positive")
            self.assertEqual(row["source_directions"]["N"]["direction"], "negative")
            self.assertIsNone(row["suggested_source"])
            self.assertEqual(np.shape(row["repeat_source_signed_contrasts"]), (2, 2))
        self.assertFalse(report["scientifically_validated"])
        self.assertFalse(report["automatic_regions"])

    def test_source_swap_preserves_geometry_and_swaps_channel_evidence(self):
        p = 80*np.exp(-((self.e-1.4)/.003)**2)+15*np.exp(-((self.e-1.65)/.003)**2)
        n = 60*np.exp(-((self.e-1.4)/.003)**2)-12*np.exp(-((self.e-1.65)/.003)**2)
        _, a = self.run_profiles(p, n)
        _, b = self.run_profiles(n, p)
        self.assertEqual(len(a["regions"]), len(b["regions"]))
        for left, right in zip(a["regions"], b["regions"]):
            for field in ("bounds_eV", "core_bounds_eV", "center_ev", "branch", "polarity", "status"):
                self.assertEqual(left[field], right[field])
            self.assertEqual(left["source_directions"]["P"], right["source_directions"]["N"])
            self.assertEqual(left["source_directions"]["N"], right["source_directions"]["P"])

    def test_neighbor_source_deduplication_does_not_choose_stronger_center(self):
        p = 200*np.exp(-((self.e-1.5005)/.003)**2)
        n = 20*np.exp(-((self.e-1.5)/.003)**2)
        _, report = self.run_profiles(p, n)
        positive = [r for r in report["regions"] if r["polarity"] == 1]
        self.assertEqual(len(positive), 1)
        self.assertEqual(positive[0]["center_ev"], 1.5)
        self.assertEqual({r["center_index"] for r in positive[0]["observations"]}, {1000, 1001})
        for observation in positive[0]["observations"]:
            center = observation["center_index"]
            lo, hi = observation["anchor_indices"]
            fraction = (self.e[center]-self.e[lo])/(self.e[hi]-self.e[lo])
            expected = p[center]-(p[lo]+fraction*(p[hi]-p[lo]))
            self.assertAlmostEqual(observation["repeat_source_signed_contrasts"][0][0], expected)

    def test_flat_top_retains_extent_and_one_center(self):
        e = self.e
        p = np.maximum(0, np.minimum((e-1.48)/.01, (1.52-e)/.01))
        p = 10*np.minimum(1, p)
        _, report = self.run_profiles(p)
        positive = [r for r in report["regions"] if r["polarity"] == 1]
        self.assertEqual(len(positive), 1)
        for observation in positive[0]["observations"]:
            self.assertGreater(observation["plateau_bounds_eV"][1]-observation["plateau_bounds_eV"][0], .015)
        self.assertLess(abs(positive[0]["center_ev"]-1.5), .0006)

    def test_edge_low_prominence_and_narrow_candidates_are_retained_rejected(self):
        p = 10*np.exp(-((self.e-1.001)/.001)**2)
        p += .3*np.exp(-((self.e-1.4)/.003)**2)
        p[1400] += 20
        _, report = self.run_profiles(p, noise=.2)
        self.assertTrue(report["raw_candidates"])
        reasons = {reason for row in report["raw_candidates"] for reason in row["rejection_reasons"]}
        self.assertIn("nominal_SG21_footprint_touches_axis", reasons)
        self.assertIn("local_prominence_below_5_noise", reasons)
        self.assertIn("half_prominence_width_below_two_sampling_steps", reasons)
        self.assertFalse(report["regions"])

    def test_local_anchors_never_span_remote_strong_and_weak_features(self):
        p = 1000*np.exp(-((self.e-1.25)/.01)**2)+8*np.exp(-((self.e-1.8)/.003)**2)
        _, report = self.run_profiles(p)
        radius = report["resolution"]["anchor_search_radius_ev"]
        for row in report["raw_candidates"]:
            self.assertLessEqual(row["bounds_eV"][1]-row["bounds_eV"][0], 2*radius+1e-12)
        weak = [r for r in report["regions"] if r["polarity"] == 1 and abs(r["center_ev"]-1.8) < .001]
        self.assertEqual(len(weak), 1)
        self.assertTrue(weak[0]["anchor_truncated"])
        self.assertLess(weak[0]["bounds_eV"][1]-weak[0]["bounds_eV"][0], .05)

    def test_dynamic_rms_and_repeat_evidence_without_truth_claim(self):
        stationary, dynamic = profiles(self.e, np.zeros(len(self.e)))
        dynamic[:] = 10*np.exp(-((self.e-1.6)/.004)**2)
        dynamic[1] *= .8
        arrays, report = detect_local_features(self.e, stationary, dynamic, [.1, .1], [.1, .1])
        rows = report["regions"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["branch"], "dynamic")
        self.assertEqual(rows[0]["status"], "dynamic_candidate_requires_review")
        self.assertIsNone(rows[0]["suggested_source"])
        np.testing.assert_allclose(arrays["dynamic_mean_PN_rms_d2"], np.sqrt(np.mean(dynamic**2, axis=0)), atol=1e-14)
        self.assertIs(report["regions"], report["local_features"])
        json.dumps(report, allow_nan=False)

    def test_readonly_inputs_no_io_and_cancellation(self):
        stationary, dynamic = profiles(self.e, 10*np.exp(-((self.e-1.5)/.003)**2))
        data = [self.e, stationary, dynamic, np.array([.2, .2]), np.array([.2, .2])]
        copies = [a.copy() for a in data]
        for a in data: a.flags.writeable = False
        with patch("builtins.open", side_effect=AssertionError("No file input")), patch("io.open", side_effect=AssertionError("No file input")):
            detect_local_features(*data)
        for a, b in zip(data, copies): np.testing.assert_array_equal(a, b)
        with self.assertRaises(InterruptedError): detect_local_features(*data, cancelled=lambda: True)
        calls = []
        def cancel():
            calls.append(1)
            return len(calls) == 4
        with self.assertRaises(InterruptedError): detect_local_features(*data, cancelled=cancel)

    def test_invalid_profiles_noise_energy_and_empty_signal(self):
        stationary, dynamic = profiles(self.e, np.zeros(len(self.e)))
        for e, s, d, sn, dn in [(self.e[::-1], stationary, dynamic, [1, 1], [1, 1]),
                               (self.e, stationary[:1], dynamic, [1, 1], [1, 1]),
                               (self.e, stationary, dynamic-1, [1, 1], [1, 1]),
                               (self.e, stationary, dynamic, [0, 1], [1, 1]),
                               (self.e, stationary, dynamic, [1, 1], [1, float("nan")])]:
            with self.assertRaises(ValueError): detect_local_features(e, s, d, sn, dn)
        arrays, report = detect_local_features(self.e, stationary, dynamic, [1, 1], [1, 1])
        self.assertFalse(report["raw_candidates"])
        self.assertFalse(report["regions"])
        self.assertFalse(arrays["feature_support_mask"].any())


if __name__ == "__main__":
    unittest.main()
