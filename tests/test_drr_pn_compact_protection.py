"""Analytic construction checks; these do not certify a physical P/N source."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline, CubicSpline, PPoly

from core.drr_pn_reconstruction import compact_protection as compact


class CompactProtectionTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1., 2., 1001)
        self.parent = np.exp(5+.04*(self.e-1.5)+.1*(self.e-1.5)**2)
        self.regions = [dict(xmin_ev=1.43, xmax_ev=1.47, source="P")]
        self.bounds = [1.39, 1.51]
        self.h = self.bump(self.e, *self.bounds)
        self.pn = np.array([self.parent*np.exp(self.h), self.parent*np.exp(-self.h)])

    @staticmethod
    def bump(e, left, right, amplitude=.64):
        u = (e-left)/(right-left)
        result = np.zeros_like(e)
        inside = (u > 0) & (u < 1)
        # An analytic degree-six C2 compact bump, independent of the quintic
        # construction basis. Its maximum is amplitude/64.
        result[inside] = amplitude*(u[inside]*(1-u[inside]))**3
        return result

    def apply(self, **kwargs):
        return compact.apply_compact_protection(self.e, self.parent, self.pn,
                                                 self.regions, **kwargs)

    @staticmethod
    def spline(row):
        return BSpline(row["knots_u"], row["coefficients"], row["degree"],
                       extrapolate=False)

    def test_disabled_is_bitwise_parent_without_splines_or_solver(self):
        parent = self.parent.astype(np.float32)
        with patch.object(compact, "CubicSpline", side_effect=AssertionError("disabled")), \
                patch.object(compact, "lsq_linear", side_effect=AssertionError("disabled")):
            out, arrays, report = compact.apply_compact_protection(self.e, parent, self.pn, [])
        self.assertEqual(out.dtype, parent.dtype)
        self.assertEqual(out.tobytes(), parent.tobytes())
        self.assertFalse(np.shares_memory(out, parent))
        self.assertTrue(report["fit_accepted"])
        self.assertFalse(report["enabled"])
        self.assertEqual(report["whole_axis_gain_minmax"], [1., 1.])
        self.assertFalse(arrays["support_mask"].any())

    def test_nonzero_affine_log_source_exact_noop_on_uniform_and_uneven_axes(self):
        uneven = self.e+.00015*np.sin(np.linspace(0., 20*np.pi, len(self.e)))
        for e in (self.e, uneven):
            parent = np.exp(5+.2*(e-1.5)**2)
            pn = parent[None, :]*np.exp(np.array([.3+.4*(e-1.5), -.2+.1*(e-1.5)]))
            for source in ("P", "N"):
                region = [{**self.regions[0], "source": source}]
                with patch.object(compact, "CubicSpline", side_effect=AssertionError("affine noop")), \
                        patch.object(compact, "lsq_linear", side_effect=AssertionError("affine noop")):
                    out, arrays, report = compact.apply_compact_protection(e, parent, pn, region)
                self.assertEqual(out.tobytes(), parent.tobytes())
                np.testing.assert_array_equal(arrays["delta_log"], 0.)
                self.assertTrue(report["regions"][0]["affine_roundoff_exact_noop"])
                self.assertTrue(report["fit_accepted"])

    def test_compatible_analytic_compact_source_is_reconstructed(self):
        out, arrays, report = self.apply()
        self.assertTrue(report["fit_accepted"])
        np.testing.assert_allclose(arrays["delta_log"], self.h, rtol=0, atol=3e-8)
        np.testing.assert_allclose(out, self.pn[0], rtol=3e-8, atol=0)
        for row in report["regions"][0]["curvature_fit_metrics"].values():
            self.assertLess(row["relative_rms_error"], .001)
        self.assertEqual(report["regions"][0]["active_bound_count"], 0)

    def test_incompatible_constant_curvature_is_rejected_with_honest_residual(self):
        pn = np.array([self.parent*np.exp(.3*(self.e-1.5)**2), self.pn[1]])
        out, arrays, report = compact.apply_compact_protection(self.e, self.parent, pn, self.regions)
        self.assertTrue(report["numeric_valid"])
        self.assertFalse(report["fit_accepted"])
        # Compact closure imposes integral h''=0. A nonzero constant curvature
        # has an irreducible full-support relative residual of one.
        metrics = report["regions"][0]["curvature_fit_metrics"]
        self.assertAlmostEqual(metrics["support"]["relative_rms_error"], 1., places=12)
        self.assertEqual(len(report["regions"][0]["fit_rejection_reasons"]), 3)
        self.assertLess(np.max(abs(arrays["delta_log"])), 1e-12)
        self.assertTrue(np.isfinite(out).all())

    def test_rejected_cap_limited_fit_keeps_actual_candidate_and_continuous_bounds(self):
        pn = np.array([self.parent*np.exp(40*self.h), self.pn[1]])
        out, arrays, report = compact.apply_compact_protection(self.e, self.parent, pn, self.regions)
        row = report["regions"][0]
        self.assertFalse(report["fit_accepted"])
        self.assertGreater(row["active_bound_count"], 0)
        self.assertGreater(np.max(abs(arrays["delta_log"])), .01)
        self.assertFalse(np.array_equal(out, self.parent))
        spline = self.spline(row)
        derivative = PPoly.from_spline(spline.derivative())
        roots = derivative.roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= 1)]
        extrema = spline(np.r_[0., 1., roots])
        self.assertLessEqual(np.max(abs(extrema)), np.log(1.05)+2e-15)
        self.assertGreaterEqual(np.min(out/self.parent), 1/1.05-2e-15)
        self.assertLessEqual(np.max(out/self.parent), 1.05+2e-15)
        coefficients = np.asarray(row["coefficients"])
        np.testing.assert_array_equal(coefficients[:3], 0.)
        np.testing.assert_array_equal(coefficients[-3:], 0.)

    def test_exterior_is_bitwise_parent_and_log_c2_jets_close(self):
        out, arrays, report = self.apply()
        row = report["regions"][0]
        outside = ~arrays["support_mask"]
        self.assertEqual(out[outside].tobytes(), self.parent[outside].tobytes())
        self.assertTrue(report["outside_support_native_parent_exact"])
        spline = self.spline(row)
        for derivative in range(3):
            np.testing.assert_array_equal(spline([0., 1.], derivative), 0.)
        np.testing.assert_array_equal(arrays["boundary_log_jet_residuals"], 0.)
        # Check every interior polynomial join independently of boundary flags.
        pp = PPoly.from_spline(spline)
        for derivative in range(3):
            d = pp.derivative(derivative)
            for k in range(len(d.x)-2):
                if d.x[k+1] <= d.x[k] or d.x[k+2] <= d.x[k+1]:
                    continue
                left = np.polyval(d.c[:, k], d.x[k+1]-d.x[k])
                self.assertAlmostEqual(left, d.c[-1, k+1], delta=2e-9)

    def test_native_uneven_axis_and_uniform_knot_spacing(self):
        e = self.e+.00015*np.sin(np.linspace(0., 20*np.pi, len(self.e)))
        parent = np.exp(3+.1*(e-1.5)**2)
        h = self.bump(e, *self.bounds)
        pn = np.array([parent*np.exp(h), parent])
        out, arrays, report = compact.apply_compact_protection(e, parent, pn, self.regions)
        self.assertTrue(report["fit_accepted"])
        np.testing.assert_allclose(arrays["delta_log"], h, rtol=0, atol=5e-8)
        row = report["regions"][0]
        width = np.diff(row["support_bounds_eV"])[0]
        spacing = np.diff(np.unique(row["knots_u"]))*width
        self.assertGreaterEqual(np.min(spacing), .5*row["SG21_full_uniform_span_ev"]-2e-15)
        self.assertLess(np.max(spacing)-np.min(spacing), 2e-15)
        np.testing.assert_array_equal(out[~arrays["support_mask"]], parent[~arrays["support_mask"]])

    def test_native_high_frequency_target_has_no_quadrature_alias(self):
        # Near-Nyquist source structure must remain in the residual even though
        # the coarse construction basis cannot reproduce it.
        source_log_ratio = .01*np.sin(2*np.pi*(self.e-self.e[0])/.0027)
        pn = np.array([self.parent*np.exp(source_log_ratio), self.parent])
        _, _, report = compact.apply_compact_protection(self.e, self.parent, pn, self.regions)
        row = report["regions"][0]
        self.assertFalse(row["fit_accepted"])
        self.assertGreater(row["curvature_fit_metrics"]["support"]["relative_rms_error"], .99)
        spline = self.spline(row)
        left, right = row["support_bounds_eV"]
        width = right-left
        target = CubicSpline(self.e, np.log(pn[0])-np.log(self.parent))
        breaks = np.unique(np.r_[row["knots_u"],
            (self.e[(self.e > left) & (self.e < right)]-left)/width])
        integral = 0.
        # Exact piecewise Taylor polynomial integration is independent of the
        # implementation's Gauss quadrature and catches omitted native breaks.
        for a, b in zip(breaks[:-1], breaks[1:]):
            mid, half = (a+b)/2, (b-a)/2
            energy = left+width*mid
            residual = Polynomial([spline(mid, 2)-width**2*target(energy, 2),
                spline(mid, 3)-width**3*target(energy, 3),
                spline(mid, 4)/2, spline(mid, 5)/6])
            anti = (residual*residual).integ()
            integral += anti(half)-anti(-half)
        self.assertAlmostEqual(integral/(2*row["solver"]["cost"]), 1., delta=2e-12)

    def test_near_zero_core_target_does_not_hide_generated_curvature(self):
        parent = np.ones_like(self.e)
        # Put structure in the left transition only. The coarse fit generates
        # core curvature where the source target is zero to machine precision.
        q = self.bump(self.e, 1.392, 1.406, amplitude=.2)
        pn = np.array([np.exp(q), parent])
        _, _, report = compact.apply_compact_protection(self.e, parent, pn, self.regions)
        metric = report["regions"][0]["curvature_fit_metrics"]["core"]
        self.assertTrue(metric["near_zero_target"])
        self.assertIsNone(metric["relative_rms_error"])
        self.assertGreater(metric["residual_rms_u2"], .01)
        self.assertGreater(metric["residual_rms_u2"], metric["machine_absolute_tolerance_u2"])
        self.assertFalse(metric["accepted"])
        self.assertFalse(report["fit_accepted"])

    def test_pn_swap_and_multiple_disjoint_regions(self):
        swapped = self.pn[::-1].copy()
        out, _, report = self.apply()
        other, _, _ = compact.apply_compact_protection(self.e, self.parent, swapped,
            [{**self.regions[0], "source": "N"}])
        np.testing.assert_array_equal(other, out)
        second_region = dict(xmin_ev=1.68, xmax_ev=1.72, source="N")
        h2 = -self.bump(self.e, 1.64, 1.76, amplitude=.32)
        pn = np.array([self.pn[0], self.parent*np.exp(h2)])
        combined, arrays, detail = compact.apply_compact_protection(self.e, self.parent,
            pn, [second_region, self.regions[0]])
        self.assertEqual([r["source"] for r in detail["regions"]], ["P", "N"])
        self.assertTrue(detail["fit_accepted"])
        np.testing.assert_allclose(arrays["delta_log"], self.h+h2, rtol=0, atol=3e-8)
        np.testing.assert_array_equal(combined[~arrays["support_mask"]], self.parent[~arrays["support_mask"]])

    def test_inputs_unchanged_no_io_and_json_and_numeric_arrays_are_finite(self):
        copies = [v.copy() for v in (self.e, self.parent, self.pn)]
        regions = copy.deepcopy(self.regions)
        for value in (self.e, self.parent, self.pn):
            value.flags.writeable = False
        with patch("builtins.open", side_effect=AssertionError("No I/O")), \
                patch("io.open", side_effect=AssertionError("No I/O")):
            _, arrays, report = self.apply()
        for value, before in zip((self.e, self.parent, self.pn), copies):
            np.testing.assert_array_equal(value, before)
        self.assertEqual(regions, self.regions)
        for name, value in arrays.items():
            self.assertTrue(np.isfinite(value).all(), name)
            self.assertNotEqual(value.dtype, object)
        json.dumps(report, allow_nan=False)
        self.assertFalse(report["sample_used"])
        self.assertFalse(report["BN_used"])
        self.assertFalse(report["scientifically_validated"])
        self.assertFalse(report["production_default_allowed"])

    def test_invalid_inputs_geometry_and_cancel_callable(self):
        for e, parent, pn in ((self.e[::-1], self.parent, self.pn),
                (self.e, self.parent[:-1], self.pn), (self.e, self.parent, self.pn[:1]),
                (self.e, -self.parent, self.pn), (self.e, self.parent, self.pn*np.nan),
                (self.e[:20], self.parent[:20], self.pn[:, :20])):
            with self.subTest(shapes=(e.shape, parent.shape, pn.shape)), self.assertRaises(ValueError):
                compact.apply_compact_protection(e, parent, pn, self.regions)
        invalid_regions = [[dict(xmin_ev=1.01, xmax_ev=1.1, source="P")],
            [dict(xmin_ev=1.4301, xmax_ev=1.4302, source="P")],
            self.regions+[dict(xmin_ev=1.5, xmax_ev=1.53, source="N")],
            [dict(xmin_ev=True, xmax_ev=1.5, source="P")],
            [dict(xmin_ev=1.43, xmax_ev=1.47, source="BN")]]
        for regions in invalid_regions:
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                compact.apply_compact_protection(self.e, self.parent, self.pn, regions)
        with self.assertRaises(TypeError):
            self.apply(cancelled=True)

    def test_cancellation_and_numerical_or_unexpected_failure_propagate(self):
        with self.assertRaises(InterruptedError):
            self.apply(cancelled=lambda: True)
        calls = []
        def cancelled():
            calls.append(1)
            return len(calls) == 5
        with self.assertRaises(InterruptedError):
            self.apply(cancelled=cancelled)
        self.assertEqual(len(calls), 5)
        failed = SimpleNamespace(success=False, x=np.zeros(1))
        with patch.object(compact, "lsq_linear", return_value=failed), \
                self.assertRaisesRegex(ArithmeticError, "did not converge"):
            self.apply()
        with patch.object(compact, "lsq_linear", side_effect=RuntimeError("unexpected")), \
                self.assertRaisesRegex(RuntimeError, "unexpected"):
            self.apply()


if __name__ == "__main__":
    unittest.main()
