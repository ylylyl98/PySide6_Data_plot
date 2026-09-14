from types import SimpleNamespace
import unittest

import numpy as np

from core.mcd_analysis import FeatureMeasurement, detect_analysis_features


def _synthetic():
    energy = np.linspace(1.5, 1.9, 1001)
    b = np.linspace(-2, 2, 17)
    fields = np.tile(b, 2)
    labels = np.repeat(["B increasing", "B decreasing"], 17)
    stable = .1 * np.exp(-((energy - 1.56) / .004) ** 2)
    moving = .02 * np.exp(-((energy - 1.74) / .004) ** 2)
    rng = np.random.default_rng(812)
    values = stable + fields[:, None] / 2 * moving + rng.normal(0, .0002, (34, 1001))
    return SimpleNamespace(energy_ev=energy, pair_b=fields, pair_labels=labels,
        pair_mcd_corrected=values, pair_raw_pos=np.tile(stable + moving, (34, 1)),
        pair_raw_neg=np.tile(stable + moving, (34, 1)))


class McdCenterRemediationTests(unittest.TestCase):
    def test_measurements_are_json_primitives_and_support_rows_are_unique(self):
        result = detect_analysis_features(_synthetic(), spectral_sources=())
        mcd = [f for f in result.features if f.domain == "mcd"]
        self.assertTrue(mcd)
        self.assertTrue(all(isinstance(p, FeatureMeasurement) for f in mcd for p in f.measured_points))
        self.assertTrue(all(f.support_count == len(f.measured_points) for f in mcd))
        self.assertTrue(all(len({round(p.field_t, 10) for p in f.measured_points}) == f.support_count for f in mcd))
        self.assertIsInstance(result.to_dict()["features"][0]["measured_points"], list)

    def test_static_texture_is_not_recommended_but_field_response_is(self):
        result = detect_analysis_features(_synthetic(), spectral_sources=())
        groups = {}
        for feature in result.features:
            groups.setdefault(feature.metadata_dict["recommendation_group"], []).append(feature)
        stable = [f for f in result.features if abs(f.energy_ev - 1.56) < .015]
        response = [f for f in result.features if abs(f.energy_ev - 1.74) < .015]
        self.assertTrue(stable and response)
        self.assertTrue(all(not f.metadata_dict["recommended"] for f in stable))
        self.assertTrue(any(f.metadata_dict["recommended"] for f in response))
        self.assertGreater(max(f.metadata_dict["response_snr"] for f in response),
                           max(f.metadata_dict["response_snr"] for f in stable) * 3)
        self.assertTrue(all("response_snr" in f.metadata_dict for f in result.features))
        self.assertTrue(all("scoring_interval_ev" in f.metadata_dict for f in result.features))

    def test_static_map_has_no_recommendation_and_even_b2_response_is_retained(self):
        base = _synthetic()
        energy = base.energy_ev
        shape = np.exp(-((energy - 1.72) / .004) ** 2)
        base.pair_mcd_corrected = .25 * np.ones((34, energy.size)) + (base.pair_b[:, None] ** 2) / 4 * shape
        b2 = detect_analysis_features(base, spectral_sources=())
        self.assertTrue(any(f.metadata_dict["recommended"] for f in b2.features))
        static = _synthetic(); static.pair_mcd_corrected = np.ones_like(static.pair_mcd_corrected) * .2
        flat = detect_analysis_features(static, spectral_sources=())
        self.assertFalse(any(f.metadata_dict["recommended"] for f in flat.features))

    def test_reversing_equivalent_energy_storage_preserves_every_score_and_flag(self):
        left_result = _synthetic(); left = detect_analysis_features(left_result)
        right_result = _synthetic(); right_result.energy_ev = right_result.energy_ev[::-1]
        right_result.pair_mcd_corrected = right_result.pair_mcd_corrected[:, ::-1]
        right_result.pair_raw_pos = right_result.pair_raw_pos[:, ::-1]
        right_result.pair_raw_neg = right_result.pair_raw_neg[:, ::-1]
        right = detect_analysis_features(right_result)
        la = {f.id: f for f in left.features if f.domain == "mcd"}
        ra = {f.id: f for f in right.features if f.domain == "mcd"}
        self.assertEqual(la.keys(), ra.keys())
        for key in la:
            lm, rm = la[key].metadata_dict, ra[key].metadata_dict
            self.assertAlmostEqual(lm["recommendation_score"], rm["recommendation_score"], places=8)
            self.assertEqual(lm["recommended"], rm["recommended"])
            self.assertEqual(lm["quality_flags"], rm["quality_flags"])


if __name__ == "__main__":
    unittest.main()
