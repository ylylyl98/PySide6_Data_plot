"""Generation contracts; scientific acceptance is not inferred from these tests."""
import builtins
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline, PPoly

from core import drr_pn_generator as G
from core.drr_composite_background import (
    compose_background, save_composite_background, read_composite_background,
)
from core.load_cancellation import LoadCancelled
from core.loader import build_external_baseline


class PnArrayGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1.4, 2.05, 651)
        self.logbase = 9 + .4*(self.e-1.7) - .7*(self.e-1.7)**2
        self.base = np.exp(self.logbase)

    def test_shared_texture_is_kept_and_method_is_not_claimed_as_v64(self):
        raw = self.base*np.exp(.002*np.sin(self.e*1500))
        before = raw.copy()
        raw.setflags(write=False)
        arrays, report = G.generate_pn_arrays(self.e, raw, raw)
        np.testing.assert_allclose(arrays["background"], raw, rtol=3e-14, atol=0)
        np.testing.assert_array_equal(raw, before)
        self.assertEqual(report["algorithm_version"], G.ALGORITHM_VERSION)
        self.assertFalse(report["v64_equivalent"])
        self.assertFalse(report["scientifically_validated"])
        self.assertFalse(report["BN_used"])
        self.assertFalse(report["sample_used"])
        self.assertTrue(np.all(arrays["source_weights"][0] == 1))

    def test_complementary_peak_and_valley_choose_other_native_source(self):
        p = self.base*np.exp(-.4*np.exp(-((self.e-1.61)/.003)**2))
        n = self.base*np.exp(.3*np.exp(-((self.e-1.74)/.004)**2))
        arrays, report = G.generate_pn_arrays(self.e, p, n)
        for center, source in ((1.61, 1), (1.74, 0)):
            index = int(np.argmin(abs(self.e-center)))
            self.assertEqual(arrays["source_weights"][source, index], 1)
        poly = PPoly(arrays["background_log_polynomial_coefficients"], self.e)
        np.testing.assert_allclose(poly(self.e), arrays["background_log"], rtol=0, atol=1e-12)
        self.assertLess(max(report["C2_join_error"]), 1e-6)
        self.assertTrue(report["transitions"])
        for row in report["source_segments"]:
            self.assertLess(row["shape_residual_max"], 1e-10)

    def test_shared_peak_is_unresolved_not_proven_clean_or_removed(self):
        raw = self.base*np.exp(.2*np.exp(-((self.e-1.72)/.004)**2))
        arrays, report = G.generate_pn_arrays(self.e, raw, raw)
        self.assertTrue(any(a <= 1.72 <= b for a, b in report["unresolved_regions"]))
        np.testing.assert_allclose(arrays["background"], raw, rtol=3e-14)

    def test_one_moment_bridge_retains_nonuniform_pure_shapes(self):
        e = 1.4 + .65*np.linspace(0, 1, 301)**1.1
        logs = np.array([9+.3*e+.002*np.sin(e*1200),
                         9.1+.25*e+.002*np.sin(e*1200)])
        chosen = np.where(e < 1.7, 0, 1)
        w, supports = G._transition_weights(e, chosen, .008)
        poly, _, records, _, reference = G._curvature_bridges(e, logs, w, supports)
        self.assertTrue(records)
        self.assertLess(max(G._jumps(poly)), 1e-6)
        for source, pure in ((0, w == 1), (1, w == 0)):
            delta = poly(e[pure])-logs[source, pure]
            self.assertLess(np.ptp(delta), 1e-11)
        cell = np.gradient(e)
        self.assertAlmostEqual(float(cell@(poly(e)-reference)), 0., places=12)

    def test_invalid_data_and_settings_are_rejected(self):
        for settings in ({"transition_halfwidth_ev": 0}, {"feature_threshold": -1},
                         {"widths_ev": [.07, .035]}, {"transition_halfwidth_ev": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                G.generate_pn_arrays(self.e, self.base, self.base, settings=settings)
        bad = self.base.copy()
        bad[50] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            G.generate_pn_arrays(self.e, bad, self.base)
        with self.assertRaisesRegex(ValueError, "increasing"):
            G.generate_pn_arrays(self.e[::-1], self.base, self.base)


class PreparedV64RendererTests(unittest.TestCase):
    def test_explicit_arbitrary_N_region_and_parent_exterior(self):
        e = np.linspace(1.2, 2.3, 251)
        A = np.array([9+.2*e+.002*np.sin(e*700), 9.1+.18*e+.002*np.sin(e*700)])
        parent_log = A[1] + .008*np.exp(-((e-1.7)/.15)**2)
        parent = CubicSpline(e, parent_log)
        recipe = G.V64RegionRecipe(1, 70, 90, 140, 160)
        original = [a.copy() for a in (e, A, parent_log, parent.c)]
        for a in (e, A, parent_log, parent.c):
            a.setflags(write=False)
        with patch.object(builtins, "open", side_effect=AssertionError("No IO")):
            arrays, report = G.render_v64_region(e, aligned_source_log=A,
                parent_background_log=parent_log, parent_polynomial_coefficients=parent.c,
                recipe=recipe)
        for value, old in zip((e, A, parent_log, parent.c), original):
            np.testing.assert_array_equal(value, old)
        outside = np.r_[np.arange(71), np.arange(160, len(e))]
        np.testing.assert_array_equal(arrays["background_log"][outside], parent_log[outside])
        delta = arrays["background_log"][90:141]-A[1, 90:141]
        np.testing.assert_allclose(delta, arrays["regional_log_constant"], rtol=0, atol=2e-15)
        self.assertLess(max(report["C2_join_error"]), 1e-7)
        self.assertFalse(report["raw_PN_preparation_performed"])
        self.assertEqual(report["algorithm"], G.PREPARED_V64_VERSION)

    def test_invalid_prepared_recipe_does_not_run_raw_fallback(self):
        e = np.linspace(1.4, 2., 101)
        A = np.array([9+e, 9+e])
        parent = CubicSpline(e, A[0])
        with self.assertRaises(ValueError):
            G.render_v64_region(e, aligned_source_log=A, parent_background_log=A[0],
                parent_polynomial_coefficients=parent.c,
                recipe=G.V64RegionRecipe(0, 10, 20, 20, 30))


class PnFileGeneratorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.energy = np.linspace(1.4, 2.05, 201)
        self.base = 1000*np.exp(.2*self.energy)
        self.write("angle17_P.csv", [self.base, self.base*1.1])
        self.write("angle17_N.csv", [self.base, self.base*1.1])
        self.pair = {"p_files": ("angle17_P.csv",), "n_files": ("angle17_N.csv",)}

    def write(self, name, values, shift=0):
        with (self.folder/name).open("w", newline="", encoding="utf8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Vbg", "Vtg", "Vbias", *[format(x+shift, ".17g") for x in self.energy]])
            for index, row in enumerate(values):
                writer.writerow([index, 0, 0, *row])

    def test_frame_policies_snapshots_and_real_external_roundtrip(self):
        result = G.generate_pn_background(self.folder, self.pair, "last", "last")
        self.assertEqual([s.key for s in result.document.sources[:2]], ["P", "N"])
        self.assertEqual([s.which for s in result.document.sources[:2]], ["last", "last"])
        self.assertEqual(len(result.document.sources[0].source_hashes), 1)
        generated = result.document.sources[-1]
        metadata = json.loads(generated.generation_json)
        self.assertEqual(metadata["algorithm_version"], G.ALGORITHM_VERSION)
        self.assertEqual(metadata["settings"], G.default_settings())
        energy, intensity, seams = compose_background(result.document)
        np.testing.assert_allclose(intensity, self.base*1.1, rtol=3e-14)
        self.assertEqual(seams, ())
        destination = self.folder/"generated.csv"
        save_composite_background(destination, result.document)
        restored = read_composite_background(destination)
        np.testing.assert_array_equal(compose_background(restored)[1], intensity)
        loaded = build_external_baseline(str(self.folder), [destination.name], which="all")
        np.testing.assert_array_equal(loaded["energy"], energy)
        np.testing.assert_allclose(loaded["I0"], intensity, rtol=1e-14)

    def test_grid_mismatch_and_cancellation_leave_inputs_unchanged(self):
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        with self.assertRaises(LoadCancelled):
            G.generate_pn_background(self.folder, self.pair, cancelled=lambda: True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.folder.iterdir()})
        self.write("shifted.csv", [self.base], shift=.001)
        with self.assertRaisesRegex(ValueError, "same ordered"):
            G.generate_pn_background(self.folder, {"p_files": self.pair["p_files"], "n_files": ("shifted.csv",)})

    def test_generation_and_version_store_bind_same_canonical_request(self):
        from core.drr_pn_pairing import DrrPnPair
        from core.drr_pn_store import DrrPnStore

        pair = DrrPnPair(**self.pair)
        settings = G.default_settings()
        store = DrrPnStore(self.folder)
        key = store.prepare(pair, p_which="first", n_which="last",
                            settings=settings, algorithm_version=G.ALGORITHM_VERSION,
                            dependencies=G.dependency_paths())
        result = G.generate_pn_background(self.folder, pair, "first", "last", settings)
        self.assertEqual(result.report["settings"], settings)
        version = store.save(result.document, key)
        self.assertEqual(version.algorithm_version, G.ALGORITHM_VERSION)
        self.assertTrue(version.fresh)
        self.assertEqual(store.find_reusable(key).version_id, version.version_id)
        restored = read_composite_background(version.csv_path)
        np.testing.assert_array_equal(compose_background(restored)[1], result.arrays["background"])

    def test_frame_and_path_alias_validation_precede_generation(self):
        with self.assertRaisesRegex(ValueError, "Frame choice"):
            G.generate_pn_background(self.folder, self.pair, "FIRST", "last")
        with self.assertRaisesRegex(ValueError, "aliases"):
            G.generate_pn_background(self.folder, {
                "p_files": ("angle17_P.csv",), "n_files": ("./angle17_P.csv",)})


if __name__ == "__main__":
    unittest.main()
