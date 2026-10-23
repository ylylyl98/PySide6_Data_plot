"""Numerical/provenance checks, not scientific acceptance of P/N backgrounds."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

from core.drr_pn_reconstruction.regions_helpers import local_quadratic
from core.drr_pn_texture_compatibility import (
    analyze_raw_common_texture,
    apply_common_fine_delta,
)


def synthetic_inputs(n=720, shift=0.38, gain=1.12, local_mismatch=False):
    energy = np.linspace(1.7, 2.4, n)
    pixel = np.arange(n, dtype=float)
    texture = 0.002 * np.sin(2 * np.pi * pixel * 0.29)
    texture += 0.001 * np.cos(2 * np.pi * pixel * 0.371 + 0.4)
    curve = CubicSpline(pixel, texture)
    registered = gain * curve(pixel + shift)
    if local_mismatch:
        edges = np.linspace(energy[0], energy[-1], 7)
        block = (energy >= edges[2]) & (energy < edges[3])
        registered[block] = 0.95 * curve(pixel[block] - 0.65)
    source = np.stack([
        np.exp(7 + texture + sign * 0.03 * (energy - 2.0))
        for sign in (-1, 1)
    ])
    frames = source[:, None, :] * np.array([0.998, 1.002, 0.999, 1.001])[None, :, None]
    gates = np.stack([
        np.exp(7.1 + registered + 0.02 * i * (energy - 2.0) ** 2)
        for i in range(9)
    ])
    samples = gates[None, :, :] * np.array([0.999, 1.001])[:, None, None]
    eligible = np.ones(n, dtype=bool)
    eligible[:20] = eligible[-20:] = False
    return energy, frames, samples, eligible


class RawCommonTextureTests(unittest.TestCase):
    def test_recovers_known_shift_gain_and_leaves_inputs_unchanged(self):
        inputs = synthetic_inputs()
        originals = [value.copy() for value in inputs]
        arrays, report = analyze_raw_common_texture(*inputs)
        np.testing.assert_allclose(arrays["parameters"], [0.38, 1.12], atol=1e-7)
        for value, original in zip(inputs, originals):
            np.testing.assert_array_equal(value, original)
        self.assertTrue(report["candidate_available"])
        self.assertTrue(report["support"]["all_optimizers_successful"])
        self.assertFalse(report["scientifically_validated"])
        self.assertFalse(report["automatic_application"])
        self.assertTrue(report["protocol"]["cv"]["conditional_on_eligibility_mask"])
        self.assertFalse(report["protocol"]["cv"]["blind_validation"])
        self.assertLess(report["training_errors"]["eligible"]["after"], 1e-9)
        json.dumps(report, allow_nan=False)

    def test_block_cv_exposes_local_mismatch_despite_global_improvement(self):
        arrays, report = analyze_raw_common_texture(*synthetic_inputs(local_mismatch=True))
        self.assertLess(report["training_errors"]["eligible"]["relative_RMS_change"], 0)
        self.assertEqual(len(report["folds"]), 6)
        bad = report["folds"][2]
        self.assertTrue(bad["supported"])
        self.assertTrue(bad["optimizer_success"])
        for domain in ("full", "eligible"):
            self.assertGreater(bad["domains"][domain]["relative_RMS_change"], 0.1)
        self.assertFalse(report["scientifically_validated"])
        self.assertEqual(arrays["fold_parameters"].shape, (6, 2))

    def test_same_frozen_transform_checks_and_halves(self):
        energy, pn, samples, mask = synthetic_inputs()
        # A disagreement limited to the first fixed gate group must remain
        # visible even though the majority-gate target still fits well.
        alternate = synthetic_inputs(shift=-0.65, gain=0.95)[2]
        samples[:, :3] = alternate[:, :3]
        arrays, report = analyze_raw_common_texture(energy, pn, samples, mask)
        shift, gain = arrays["parameters"]
        pixel = np.arange(energy.size)
        expected = gain * CubicSpline(
            pixel, arrays["half_raw_common_fine"], axis=-1,
        )(pixel + shift) - arrays["half_raw_common_fine"]
        np.testing.assert_array_equal(arrays["half_proposed_delta_log"], expected)
        checks = report["same_transform_checks"]
        self.assertFalse(checks["refitted"])
        self.assertEqual(set(checks["sources"]), {"P", "N"})
        self.assertEqual(len(checks["pn_halves"]), 2)
        self.assertEqual(len(checks["training_repeats"]), 2)
        self.assertEqual(len(checks["gate_groups"]), 3)
        self.assertEqual(checks["gate_groups"][0]["gate_indices"], [0, 1, 2])
        for block in checks["gate_groups"][0]["domains"]["blocks"]:
            self.assertGreater(block["eligible"]["relative_RMS_change"], 0)
        self.assertLess(checks["gate_groups"][1]["domains"]["eligible"]["after"], 1e-9)

    def test_averages_intensity_before_log_and_reports_leakage(self):
        energy, pn, samples, mask = synthetic_inputs()
        x = np.arange(energy.size)
        pn[:, 0] *= np.exp(0.02 * np.sin(x))
        samples[0] *= np.exp(0.02 * np.cos(x))
        arrays, report = analyze_raw_common_texture(energy, pn, samples, mask)
        source_log = np.log(pn.mean(axis=1))
        expected = source_log - local_quadratic(energy, source_log, 0.004)
        np.testing.assert_allclose(arrays["raw_source_fine"], expected, rtol=0, atol=1e-14)
        logs = np.log(samples.mean(axis=0))
        target = np.median(logs - local_quadratic(energy, logs, 0.004), axis=0)
        np.testing.assert_allclose(arrays["target_train_fine"], target, rtol=0, atol=1e-14)
        q4 = local_quadratic(energy, arrays["proposed_delta_log"], 0.004)
        q35 = local_quadratic(energy, arrays["proposed_delta_log"], 0.035)
        np.testing.assert_array_equal(arrays["Q4_delta_leakage"], q4)
        np.testing.assert_array_equal(arrays["medium_delta_leakage"], q4 - q35)
        self.assertGreater(report["leakage"]["Q4_delta_RMS"], 0)

    def test_support_and_zero_denominator_are_explicit_and_json_safe(self):
        inputs = list(synthetic_inputs(n=30))
        inputs[1][:] = 1.0
        inputs[2][:] = 1.0
        inputs[3][:] = False
        arrays, report = analyze_raw_common_texture(*inputs)
        self.assertFalse(report["candidate_available"])
        self.assertEqual(report["optimizer"]["status"], "insufficient_support")
        np.testing.assert_array_equal(arrays["proposed_delta_log"], np.zeros(30))
        np.testing.assert_array_equal(arrays["half_proposed_delta_log"], np.zeros((2, 30)))
        self.assertIsNone(report["training_errors"]["eligible"]["before"])
        self.assertIsNone(report["training_errors"]["full"]["relative_RMS_change"])
        self.assertIsNone(report["leakage"]["Q4_delta_RMS_over_delta_RMS"])
        self.assertTrue(np.isnan(arrays["fold_parameters"]).all())
        self.assertTrue(all(not fold["supported"] for fold in report["folds"]))
        json.dumps(report, allow_nan=False)

    def test_full_domain_is_reported_when_eligible_test_lacks_support(self):
        inputs = list(synthetic_inputs())
        inputs[3][120:240] = False
        _, report = analyze_raw_common_texture(*inputs)
        fold = report["folds"][1]
        self.assertFalse(fold["supported"])
        self.assertTrue(fold["domains"]["full"]["supported"])
        self.assertIsNotNone(fold["domains"]["full"]["after"])
        self.assertFalse(fold["domains"]["eligible"]["supported"])
        self.assertIsNone(fold["domains"]["eligible"]["after"])

    def test_failed_optimization_is_never_an_available_candidate(self):
        failed = SimpleNamespace(x=np.array([0.0, 1.0]), success=False, nfev=300,
                                 status=0, message="Evaluation budget exhausted", cost=0.1,
                                 optimality=0.2)
        with patch("core.drr_pn_texture_compatibility.least_squares", return_value=failed):
            _, report = analyze_raw_common_texture(*synthetic_inputs())
        self.assertFalse(report["candidate_available"])
        self.assertFalse(report["optimizer"]["success"])
        self.assertEqual(report["optimizer"]["status"], "not_converged")
        self.assertFalse(report["support"]["all_optimizers_successful"])

    def test_rejects_invalid_input(self):
        good = synthetic_inputs(n=210)
        cases = []
        for index in (1, 2):
            for invalid in (0, -1, np.nan, np.inf):
                values = [item.copy() for item in good]
                values[index].flat[0] = invalid
                cases.append(values)
        for energy in (good[0][::-1], np.full(210, 1.0), good[0].reshape(2, 105)):
            cases.append([energy, *good[1:]])
        nonfinite = good[0].copy()
        nonfinite[0] = np.nan
        cases.append([nonfinite, *good[1:]])
        cases.extend([
            [good[0], good[1][:, :3], good[2], good[3]],
            [good[0], good[1][0], good[2], good[3]],
            [good[0], good[1], good[2][:, :2], good[3]],
            [good[0], good[1], good[2][:1], good[3]],
            [*good[:3], good[3].astype(int)],
            [*good[:3], good[3][:-1]],
            [good[0][:20], good[1][..., :20], good[2][..., :20], good[3][:20]],
        ])
        for index, values in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                analyze_raw_common_texture(*values)

    def test_cancellation_interrupts_kernel_and_does_not_leak_scope(self):
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls >= 12

        with self.assertRaises(InterruptedError):
            analyze_raw_common_texture(*synthetic_inputs(), cancelled=cancelled)
        self.assertEqual(calls, 12)
        # A later invocation must not inherit a cancelled context.
        _, report = analyze_raw_common_texture(*synthetic_inputs(n=30))
        self.assertFalse(report["candidate_available"])


class ApplyCommonDeltaTests(unittest.TestCase):
    def test_zero_delta_is_exact_copy(self):
        parent = np.array([1.0000001, 103.12345, 0.000231], dtype=np.float32)
        result = apply_common_fine_delta(parent, np.zeros(3))
        self.assertTrue(np.array_equal(result, parent))
        self.assertEqual(result.dtype, parent.dtype)
        self.assertFalse(np.shares_memory(result, parent))

    def test_explicit_application_preserves_parent_and_validates_result(self):
        parent = np.array([0.5, 7.0, 300.0])
        old = parent.copy()
        delta = np.array([0.01, -0.02, 0.0])
        result = apply_common_fine_delta(parent, delta)
        np.testing.assert_array_equal(parent, old)
        np.testing.assert_allclose(result, parent * np.exp(delta), rtol=0, atol=0)
        self.assertTrue(np.all(result > 0))
        for bad_parent, bad_delta in [
            ([0.0], [0.0]), ([np.nan], [0.0]), ([1.0], [np.nan]),
            ([1.0], [np.inf]), ([1.0], [1000]), ([1.0], [-1000]),
            ([1.0, 2.0], [0.0]), ([], []), ([[1.0]], [[0.0]]),
        ]:
            with self.subTest(parent=bad_parent, delta=bad_delta), self.assertRaises(ValueError):
                apply_common_fine_delta(bad_parent, bad_delta)


if __name__ == "__main__":
    unittest.main()
