from dataclasses import replace
import unittest

from core.drr_pn_pairing import DrrPnPair, describe_manual_pair, detect_pn_role, recommend_pn_pairs
from core.drr_sources import DrrSource, measurement_group_key


def source(role, angle=49, repeat=1, *, gate=None, **overrides):
    token = "data" if role == "sample" else role
    name = f"YZ383_p2n1_1.67K_9T_RotIn{angle}deg_730nmc_0.3sx10_{token}_rep{repeat}.csv"
    gate = (-20 if role.lower().startswith("pback") else 20) if gate is None else gate
    item = DrrSource(
        name, name, measurement_group_key(name), "2026-10-20", 100., role != "sample",
        gate_varies=False, frame_count=2, wavelength_center_nm=730.,
        gate_grid=((gate, 0., 0.), (gate, 0., 0.)),
        spectral_grid=(700., 710., 720.), gate_labels=("Vbg", "Vtg", "Vbias"), grid_complete=True,
    )
    return replace(item, **overrides)


class PnPairingTests(unittest.TestCase):
    def test_explicit_role_tokens_only(self):
        for name in ("Pbackground.csv", "device_Pback_rep1.csv", "x_pbackground2.csv"):
            self.assertEqual(detect_pn_role(name), "P")
        for name in ("Nbackground.csv", "device_Nback_1.csv", "x_nbackground12.csv"):
            self.assertEqual(detect_pn_role(name), "N")
        for name in ("BNbackground.csv", "x_BNback.csv", "p2n1.csv", "background.csv",
                     "sample_Pbackground_Nbackground.csv", "notNback.csv", "PbackgroundNoise.csv"):
            self.assertIsNone(detect_pn_role(name), name)

    def test_opposite_gate_values_pair_and_repeats_remain_separate_roles(self):
        p1, p2, n1, n2 = (source(role, repeat=repeat) for role, repeat in
                           (("Pbackground", 1), ("Pbackground", 2), ("Nbackground", 1), ("Nbackground", 2)))
        result = recommend_pn_pairs([source("sample")], [p1, p2, n1, n2])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].pair.p_files, (p1.source, p2.source))
        self.assertEqual(result[0].pair.n_files, (n1.source, n2.source))
        self.assertFalse(result[0].condition_differences)
        self.assertIn("rotation", result[0].matched_conditions)

    def test_arbitrary_angles_and_conflicts_inside_pair_are_not_combined(self):
        catalog = [source(role, angle) for angle in (17.5, 132)
                   for role in ("Pbackground", "Nbackground")]
        result = recommend_pn_pairs([source("sample", 132)], catalog)
        self.assertEqual(len(result), 2)
        self.assertIn("132deg", result[0].pair.p_files[0])
        self.assertFalse(result[0].condition_differences)
        self.assertTrue(any("rotation" in item for item in result[1].condition_differences))
        self.assertIn("17.5deg", result[1].pair.n_files[0])

    def test_internal_known_sample_position_exposure_and_optics_conflicts_excluded(self):
        p, n = source("Pbackground"), source("Nbackground")
        for old, new in (("YZ383", "YZ384"), ("p2n1", "p2n2"), ("1.67K", "4K"),
                         ("9T", "8T"), ("0.3sx10", "0.4sx10"), ("x10", "x20"),
                         ("RotIn49deg", "RotIn73deg")):
            other = replace(n, filename=n.filename.replace(old, new))
            with self.subTest(new=new):
                self.assertFalse(recommend_pn_pairs([source("sample")], [p, other]))

    def test_unknown_fields_explicit_and_multiple_pairs_not_automatically_selected(self):
        p = replace(source("Pbackground"), filename="Pbackground.csv", group_key="Pbackground")
        n = replace(source("Nbackground"), filename="Nbackground.csv", group_key="Nbackground")
        result = recommend_pn_pairs([source("sample")], [p, n])
        self.assertEqual(len(result), 1)
        self.assertIn("rotation", result[0].unknown_conditions)
        self.assertTrue(result[0].requires_review)
        self.assertIn("unknown", result[0].reason)

    def test_ordered_grid_required_and_BN_never_becomes_N(self):
        p, n = source("Pbackground"), source("Nbackground")
        self.assertFalse(recommend_pn_pairs([source("sample")], [p, source("BNbackground")]))
        self.assertFalse(recommend_pn_pairs([source("sample")], [p, replace(n, spectral_grid=(720., 710., 700.))]))
        self.assertFalse(recommend_pn_pairs([source("sample", spectral_grid=(1., 2., 3.))], [p, n]))
        self.assertFalse(recommend_pn_pairs([], [p, replace(n, spectral_grid=())]))

    def test_same_role_different_gate_conditions_are_not_averaged_as_repeats(self):
        p1, p2, n = source("Pbackground"), source("Pbackground", repeat=2, gate=-10), source("Nbackground")
        results = recommend_pn_pairs([], [p1, p2, n])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(item.pair.p_files) == 1 for item in results))

    def test_pair_validation(self):
        for p, n in (((), ("n.csv",)), (("p.csv", "p.csv"), ("n.csv",)),
                     (("p.csv",), ("P.csv",))):
            with self.assertRaises(ValueError):
                DrrPnPair(p, n)

    def test_manual_pair_displays_arbitrary_angle_conflict_and_measurement_difference(self):
        p, n, measurement = source("Pbackground", 17.5), source("Nbackground", 132), source("sample", 17.5)
        text = describe_manual_pair(DrrPnPair((p.filename,), (n.filename,)), (measurement.filename,))
        self.assertIn("rotation: P [rotin=17.5], N [rotin=132]", text)
        self.assertIn("N rotation: [rotin=132], measurement [rotin=17.5]", text)
        self.assertIn("Unknown", text)
        self.assertIn("no automatic P/N averaging or fallback", text)

    def test_manual_pair_ignores_gate_difference_and_never_reads_files(self):
        from unittest.mock import patch
        p = source("Pbackground", 73).filename.replace("_Pbackground", "_Vbg-20V_Pbackground")
        n = source("Nbackground", 73).filename.replace("_Nbackground", "_Vbg+20V_Nbackground")
        with patch("builtins.open", side_effect=AssertionError("Filename-only helper must not read files")):
            text = describe_manual_pair(DrrPnPair((p,), (n,)))
        self.assertIn("No known filename-condition conflict within the pair", text)
        self.assertNotIn("Known P/N conflicts:", text)
        self.assertNotIn("gate:", text)

    def test_manual_unknown_and_same_role_conflict_are_explicit(self):
        first, second = source("Pbackground", 17.5), source("Pbackground", 132, repeat=2)
        text = describe_manual_pair(DrrPnPair((first.filename, second.filename), ("Nbackground.csv",)))
        self.assertIn("P repeat rotation: [rotin=17.5, rotin=132]", text)
        self.assertIn("N rotation", text)
        self.assertIn("not a confirmed match or a blocker", text)


if __name__ == "__main__":
    unittest.main()
