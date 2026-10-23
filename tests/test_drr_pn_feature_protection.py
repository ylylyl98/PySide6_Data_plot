"""Construction invariants, not scientific acceptance of selected sources."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline, PPoly

from core.drr_pn_reconstruction import feature_protection as protection


class FeatureProtectionTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1., 2., 1001)
        self.parent = np.exp(5+.04*(self.e-1.5)+.1*(self.e-1.5)**2)
        p = self.parent*np.exp(.3*(self.e-1.5)**2+.004*np.exp(-((self.e-1.45)/.007)**2))
        n = self.parent*np.exp(-.2*(self.e-1.5)**2+.003*np.exp(-((self.e-1.7)/.009)**2))
        self.pn = np.array([p, n])
        self.regions = [dict(xmin_ev=1.43, xmax_ev=1.47, source="P")]

    def apply(self, **kwargs):
        return protection.apply_feature_protection(self.e, self.parent, self.pn, self.regions, **kwargs)

    def test_disabled_is_bitwise_parent_without_spline_construction(self):
        parent = self.parent.astype(np.float32)
        with patch.object(protection, "CubicSpline", side_effect=AssertionError("disabled")):
            out, arrays, report = protection.apply_feature_protection(self.e, parent, self.pn, [])
        self.assertEqual(out.dtype, parent.dtype)
        self.assertEqual(out.tobytes(), parent.tobytes())
        self.assertFalse(np.shares_memory(out, parent))
        self.assertFalse(report["enabled"])
        self.assertEqual(report["whole_axis_gain_minmax"], [1., 1.])
        self.assertFalse(arrays["curvature_support_mask"].any())

    def test_quadratic_source_exact_curvature_and_polynomial_integration(self):
        pn = np.array([self.parent*np.exp(.3*(self.e-1.5)**2), self.pn[1]])
        out, arrays, report = protection.apply_feature_protection(self.e, self.parent, pn, self.regions)
        np.testing.assert_allclose(arrays["curvature_delta"], .6*arrays["weights"][0], atol=2e-8, rtol=0)
        pp = PPoly(arrays["delta_log_polynomial_coefficients"], arrays["curvature_knots"])
        np.testing.assert_allclose(pp(self.e), arrays["delta_log"], atol=2e-16, rtol=1e-13)
        np.testing.assert_allclose(pp(self.e, 2), arrays["curvature_delta"], atol=2e-15, rtol=1e-12)
        np.testing.assert_allclose(np.log(out/self.parent), arrays["delta_log"], atol=1e-15, rtol=0)
        self.assertLess(np.max(abs(arrays["boundary_log_jet_residuals"])), 1e-10)
        self.assertTrue(report["convex_log_curvature_mixture"])

    def test_raw_source_plateau_shape_and_exterior_affine_change(self):
        out, arrays, report = self.apply()
        plateau = arrays["plateau_mask"]
        offset, tilt, center = arrays["source_affine_parameters"][0]
        np.testing.assert_allclose(np.log(out[plateau]/self.pn[0, plateau]),
                                   offset+tilt*(self.e[plateau]-center), atol=3e-15, rtol=0)
        left, right = report["regions"][0]["support_bounds_eV"]
        for mask in (self.e < left, self.e > right):
            difference = np.log(out[mask]/self.parent[mask])
            affine = np.polyfit(self.e[mask], difference, 1)
            np.testing.assert_allclose(difference, np.polyval(affine, self.e[mask]), atol=3e-15, rtol=0)
        self.assertFalse(report["outside_support_native_parent_exact"])
        np.testing.assert_array_equal(arrays["curvature_delta"][~arrays["curvature_support_mask"]], 0.)
        np.testing.assert_allclose(report["whole_axis_gain_minmax"], [np.min(out/self.parent), np.max(out/self.parent)])
        json.dumps(report, allow_nan=False)
        self.assertFalse(report["scientifically_validated"])
        self.assertIn("training samples", report["parent_indirect_sample_dependence"])

    def test_convex_curvature_envelope_between_knots_and_c2_joins(self):
        _, arrays, _ = self.apply()
        x = np.unique(np.r_[self.e, np.linspace(self.e[0], self.e[-1], 4321)])
        pp = PPoly(arrays["delta_log_polynomial_coefficients"], arrays["curvature_knots"])
        parent_curvature = CubicSpline(self.e, np.log(self.parent))(x, 2)
        source_curvature = CubicSpline(self.e, np.log(self.pn[0]))(x, 2)
        final_curvature = parent_curvature+pp(x, 2)
        self.assertLess(np.max(np.minimum(parent_curvature, source_curvature)-final_curvature), 2e-9)
        self.assertLess(np.max(final_curvature-np.maximum(parent_curvature, source_curvature)), 2e-9)
        # Check every join of the integrated polynomial independently.
        for degree in range(3):
            d = pp.derivative(degree)
            left = np.array([np.polyval(d.c[:, k], h) for k, h in enumerate(np.diff(d.x)[:-1])])
            np.testing.assert_allclose(left, d.c[-1, 1:], atol=2e-10, rtol=1e-12)

    def test_multiple_sources_and_declared_geometry(self):
        regions = self.regions+[dict(xmin_ev=1.68, xmax_ev=1.72, source="N")]
        out, arrays, report = protection.apply_feature_protection(self.e, self.parent, self.pn, regions)
        span = 20*(self.e[-1]-self.e[0])/(len(self.e)-1)
        for index, (region, row) in enumerate(zip(regions, report["regions"])):
            np.testing.assert_allclose(row["plateau_bounds_eV"], [region["xmin_ev"]-span, region["xmax_ev"]+span])
            mask = arrays["weights"][index] == 1
            offset, tilt, center = arrays["source_affine_parameters"][index]
            np.testing.assert_allclose(np.log(out[mask]/self.pn[index, mask]), offset+tilt*(self.e[mask]-center), atol=4e-15)

    def test_full_constants_frozen_while_half_uses_own_source_and_parent(self):
        full, full_arrays, report = self.apply()
        frozen = copy.deepcopy(report["frozen_parameters"])
        half_pn = self.pn*np.exp(.002*np.sin(80*self.e))[None, :]
        half_parent = self.parent*np.exp(.003*np.cos(19*self.e))
        half, arrays, detail = protection.apply_feature_protection(self.e, half_parent, half_pn, self.regions, frozen_parameters=frozen)
        np.testing.assert_array_equal(arrays["global_affine_parameters"], full_arrays["global_affine_parameters"])
        np.testing.assert_array_equal(arrays["weights"], full_arrays["weights"])
        expected = arrays["weights"][0]*(CubicSpline(self.e, np.log(half_pn[0]))(self.e, 2)-CubicSpline(self.e, np.log(half_parent))(self.e, 2))
        np.testing.assert_allclose(arrays["curvature_delta"], expected, atol=1e-12, rtol=1e-12)
        self.assertGreater(np.max(abs(half-full)), .01)
        self.assertTrue(detail["affine_parameters_reused"])
        self.assertEqual(frozen, report["frozen_parameters"])
        _, unfrozen_arrays, _ = protection.apply_feature_protection(self.e, half_parent, half_pn, self.regions)
        self.assertGreater(np.max(abs(unfrozen_arrays["global_affine_parameters"]-arrays["global_affine_parameters"])), 1e-5)

    def test_inputs_readonly_and_no_file_io(self):
        copies = [v.copy() for v in (self.e, self.parent, self.pn)]
        for value in (self.e, self.parent, self.pn):
            value.flags.writeable = False
        regions = copy.deepcopy(self.regions)
        with patch("builtins.open", side_effect=AssertionError("No I/O")), patch("io.open", side_effect=AssertionError("No I/O")):
            self.apply()
        for value, before in zip((self.e, self.parent, self.pn), copies):
            np.testing.assert_array_equal(value, before)
        self.assertEqual(self.regions, regions)

    def test_normalization_and_invalid_regions(self):
        self.assertEqual(protection.normalize_feature_protection(None), [])
        rows = [dict(xmin_ev=1.7, xmax_ev=1.8, source=" n "), dict(xmin_ev=1.3, xmax_ev=1.4, source="p")]
        self.assertEqual([r["source"] for r in protection.normalize_feature_protection(rows)], ["P", "N"])
        invalid = [{}, dict(xmin_ev=1.4, xmax_ev=1.3, source="P"), dict(xmin_ev=float("nan"), xmax_ev=1.5, source="P"),
                   dict(xmin_ev=True, xmax_ev=1.5, source="P"), dict(xmin_ev=1.3, xmax_ev=1.5, source="BN"),
                   dict(xmin_ev=1.3, xmax_ev=1.5, source="P", unknown=1)]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                protection.normalize_feature_protection([row])
        with self.assertRaises(ValueError):
            protection.normalize_feature_protection(self.regions+[dict(xmin_ev=1.47, xmax_ev=1.5, source="N")])

    def test_bad_arrays_and_insufficient_geometry_rejected(self):
        for energy, parent, pn in [(self.e[::-1], self.parent, self.pn), (self.e, self.parent[:-1], self.pn),
                                  (self.e, self.parent, self.pn[:1]), (self.e, -self.parent, self.pn),
                                  (self.e, self.parent, self.pn*float("nan")), (self.e[:20], self.parent[:20], self.pn[:, :20])]:
            with self.assertRaises(ValueError):
                protection.apply_feature_protection(energy, parent, pn, self.regions)
        for regions in [[dict(xmin_ev=1.01, xmax_ev=1.1, source="P")],
                        [dict(xmin_ev=1.4301, xmax_ev=1.4302, source="P")],
                        self.regions+[dict(xmin_ev=1.5, xmax_ev=1.53, source="N")]]:
            with self.assertRaises(ValueError):
                protection.apply_feature_protection(self.e, self.parent, self.pn, regions)

    def test_frozen_identity_and_nonfinite_output_rejected(self):
        _, _, report = self.apply()
        for key, value in [("energy_sha256", "changed"), ("global_affine_parameters", [0., 1.]),
                           ("global_affine_parameters", [0., 1., 0.]), ("global_affine_parameters", [float("nan"), 1., 1.5])]:
            frozen = copy.deepcopy(report["frozen_parameters"])
            frozen[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.apply(frozen_parameters=frozen)
        frozen = copy.deepcopy(report["frozen_parameters"])
        frozen["global_affine_parameters"][0] = 1000.
        with self.assertRaises(ValueError):
            self.apply(frozen_parameters=frozen)

    def test_support_exactly_touching_axis_endpoint_is_rejected(self):
        for lo, hi in [(1.04, 1.08), (1.92, 1.96)]:
            with self.subTest(bounds=(lo, hi)), self.assertRaisesRegex(ValueError, "strictly inside"):
                protection.apply_feature_protection(self.e, self.parent, self.pn,
                    [dict(xmin_ev=lo, xmax_ev=hi, source="P")])

    def test_cancellation_before_and_during_integration(self):
        with self.assertRaises(InterruptedError):
            self.apply(cancelled=lambda: True)
        calls = []
        def cancelled():
            calls.append(1)
            return len(calls) == 3
        with self.assertRaises(InterruptedError):
            self.apply(cancelled=cancelled)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
