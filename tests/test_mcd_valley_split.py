import unittest
from dataclasses import replace

import numpy as np

from core.mcd_peak_shift import PeakPoint, PeakShiftResult, PeakTrack
from core.mcd_valley_split import compute_valley_splitting


def _analysis(pos_fields, neg_fields, *, missing_pos=None, extra_neg=False):
    pos_fields = np.asarray(pos_fields, float)
    neg_fields = np.asarray(neg_fields, float)
    missing_pos = set(missing_pos or ())

    def track(channel, fields, offset, peak_id):
        points = []
        for field in fields:
            energy = 1.62 - 0.001 * field if channel == "pos" else 1.6203 + 0.001 * field
            status = "missing" if channel == "pos" and float(field) in missing_pos else "tracked"
            points.append(PeakPoint(float(field), "B sweep", None if status == "missing" else energy, None, status))
        return PeakTrack(peak_id, "B sweep", tuple(points), None, None, "unavailable")

    tracks = [track("pos", pos_fields, 0.0, 7), track("neg", neg_fields, 0.0, 3)]
    if extra_neg:
        tracks.append(PeakTrack(9, "B sweep", tuple(PeakPoint(float(field), "B sweep", 1.62031 + 0.001 * field, None, "tracked") for field in neg_fields), None, None, "unavailable"))
    fields = np.asarray(sorted(set(pos_fields.tolist() + neg_fields.tolist())), float)
    branches = np.full(fields.size, "B sweep")
    return {
        "pos": PeakShiftResult(fields, branches, tuple(), (tracks[0],), "raw pos"),
        "neg": PeakShiftResult(fields, branches, tuple(), tuple(tracks[1:]), "raw neg"),
    }


class ValleySplitTests(unittest.TestCase):
    def test_matches_different_field_grids_and_preserves_negative_field_identity(self):
        analysis = _analysis([-1.0, 0.0, 1.0], [-0.8, 0.2, 1.2])
        result = compute_valley_splitting({"Raw spectrum": analysis}, method="Raw spectrum", selected_channel="pos", branch="B sweep", target_energy_ev=1.62)
        self.assertEqual(result.status, "ok")
        values = {round(point.field_t, 3): point.splitting_ev for point in result.points if point.splitting_ev is not None}
        self.assertAlmostEqual(values[-0.8], -0.0013, places=6)
        self.assertAlmostEqual(values[1.0], 0.0023, places=6)

    def test_missing_point_does_not_bridge_interpolation_gap(self):
        analysis = _analysis([-1.0, 0.0, 1.0], [-0.8, 0.2, 1.2], missing_pos={0.0})
        result = compute_valley_splitting({"Raw spectrum": analysis}, method="Raw spectrum", selected_channel="pos", branch="B sweep", target_energy_ev=1.62)
        self.assertTrue(any(point.status == "missing" for point in result.points))

    def test_ambiguous_counterpart_is_explicit(self):
        analysis = _analysis([-1.0, 0.0, 1.0], [-0.8, 0.2, 1.2], extra_neg=True)
        result = compute_valley_splitting({"Raw spectrum": analysis}, method="Raw spectrum", selected_channel="pos", branch="B sweep", target_energy_ev=1.62)
        self.assertEqual(result.status, "ambiguous")

    def test_peak_does_not_pair_with_dip_counterpart(self):
        analysis = _analysis([-1.0, 0.0, 1.0], [-0.8, 0.2, 1.2])
        analysis["neg"] = replace(analysis["neg"], tracks=(replace(analysis["neg"].tracks[0], feature_kind="dip"),))
        result = compute_valley_splitting({"Raw spectrum": analysis}, method="Raw spectrum", selected_channel="pos", branch="B sweep", target_energy_ev=1.62, feature_kind="peak")
        self.assertEqual(result.status, "unmatched")

    def test_cross_channel_matching_prefers_true_e0_over_measured_energy(self):
        analysis = _analysis([-1.0, 0.0, 1.0], [-0.8, 0.2, 1.2])
        pos_track = replace(analysis["pos"].tracks[0], reference_energy_ev=1.7000, reference_method="exact 0 T")
        neg_track = replace(analysis["neg"].tracks[0], reference_energy_ev=1.7002, reference_method="interpolated near-zero")
        analysis["pos"] = replace(analysis["pos"], tracks=(pos_track,))
        analysis["neg"] = replace(analysis["neg"], tracks=(neg_track,))
        result = compute_valley_splitting(
            {"Raw spectrum": analysis}, method="Raw spectrum", selected_channel="pos",
            branch="B sweep", target_energy_ev=1.7001, tolerance_ev=0.005,
            allow_energy_fallback=False,
        )
        self.assertEqual(result.status, "ok")

    def test_strict_e0_matching_does_not_fall_back_to_missing_reference(self):
        analysis = _analysis([-1.0, 0.0, 1.0], [-0.8, 0.2, 1.2])
        pos_track = replace(analysis["pos"].tracks[0], reference_energy_ev=1.7000, reference_method="exact 0 T")
        analysis["pos"] = replace(analysis["pos"], tracks=(pos_track,))
        result = compute_valley_splitting(
            {"Raw spectrum": analysis}, method="Raw spectrum", selected_channel="pos",
            branch="B sweep", target_energy_ev=1.7000, tolerance_ev=0.005,
            allow_energy_fallback=False,
        )
        self.assertEqual(result.status, "unmatched")


if __name__ == "__main__":
    unittest.main()
