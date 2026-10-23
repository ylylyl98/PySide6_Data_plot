"""Boundary/support regressions for frozen experimental corrections."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_texture_compatibility import analyze_raw_common_texture
from core.drr_pn_texture_support import BOUNDED_TEXTURE_PROTOCOL, bound_raw_common_texture
from tests.test_drr_pn_texture_compatibility import synthetic_inputs


class BoundedTextureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = synthetic_inputs()
        cls.raw_arrays, cls.raw_report = analyze_raw_common_texture(*cls.inputs)

    def test_freezes_parameters_and_uses_same_weight_for_halves(self):
        old_arrays = {name:value.copy() for name,value in self.raw_arrays.items()}
        old_report = copy.deepcopy(self.raw_report)
        with patch("scipy.optimize.least_squares", side_effect=AssertionError("Must not refit")):
            bounded, report = bound_raw_common_texture(self.inputs[0], self.raw_arrays, self.raw_report)
        for name in ("parameters", "fold_parameters"):
            np.testing.assert_array_equal(bounded[name], old_arrays[name])
        for name,value in self.raw_arrays.items():
            np.testing.assert_array_equal(value, old_arrays[name])
        self.assertEqual(old_report, self.raw_report)
        np.testing.assert_array_equal(bounded["proposed_delta_log"],
                                      bounded["support_weight"]*old_arrays["proposed_delta_log"])
        np.testing.assert_array_equal(bounded["half_proposed_delta_log"],
                                      bounded["support_weight"][None]*old_arrays["half_proposed_delta_log"])
        self.assertTrue(report["selection"]["proposal_eligible"])
        self.assertTrue(report["selection"]["at_least_one_fold_strictly_improves"])
        self.assertFalse(report["scientifically_validated"])
        self.assertFalse(report["refitted"])
        json.dumps(report, allow_nan=False)

    def test_operator_footprints_stay_in_same_run_and_holes_never_fill(self):
        inputs = list(synthetic_inputs(n=1200))
        inputs[3][390:440] = False
        raw, raw_report = analyze_raw_common_texture(*inputs)
        bounded, report = bound_raw_common_texture(inputs[0], raw, raw_report)
        weight = bounded["support_weight"]
        self.assertTrue(np.all(weight[~inputs[3]] == 0))
        self.assertTrue(np.all((weight >= 0) & (weight <= 1)))
        for i in np.flatnonzero(weight):
            left = bounded["Q4_shift_footprint_left"][i]
            stop = bounded["Q4_shift_footprint_stop"][i]
            self.assertTrue(inputs[3][left:stop].all())
        self.assertEqual(len(report["contiguous_runs"]), 2)
        self.assertTrue(any(r["status"] == "unsupported" for r in report["region_applicability"]["regions"]))
        for run in report["contiguous_runs"]:
            if run["status"] == "supported_candidate":
                start, stop = run["guarded_indices"]
                self.assertEqual(weight[start], 0.)
                self.assertEqual(weight[stop], 0.)
                plateau = (inputs[0] >= run["plateau_energy_bounds"][0]) & (inputs[0] <= run["plateau_energy_bounds"][1])
                self.assertTrue(np.all(weight[plateau] == 1))

    def test_short_runs_remain_uncovered_and_are_not_reported_as_benefit(self):
        inputs = list(synthetic_inputs())
        inputs[3][:] = False
        # Many narrow runs have enough aggregate points for a global fit but
        # do not support a continuous correction and two transitions.
        for start in range(20, 690, 30):
            inputs[3][start:start+15] = True
        raw, raw_report = analyze_raw_common_texture(*inputs)
        bounded, report = bound_raw_common_texture(inputs[0], raw, raw_report)
        self.assertTrue(report["selection"]["original_proposal_eligible"])
        self.assertFalse(report["selection"]["proposal_eligible"])
        self.assertFalse(report["selection"]["at_least_one_fold_strictly_improves"])
        np.testing.assert_array_equal(bounded["support_weight"], np.zeros(720))
        np.testing.assert_array_equal(bounded["applied_delta_log"], np.zeros(720))
        self.assertTrue(all(run["status"] == "insufficient_contiguous_support"
                            for run in report["contiguous_runs"]))
        for fold in report["folds"]:
            for domain in fold["domains"].values():
                self.assertFalse(domain["has_benefit"])
                self.assertFalse(domain["supported"])
                self.assertEqual(domain["before"], domain["after"])

    def test_original_rejection_forces_exact_identity_despite_bounded_improvement(self):
        original = copy.deepcopy(self.raw_report)
        original["folds"][2]["optimizer_success"] = False
        bounded, report = bound_raw_common_texture(self.inputs[0], self.raw_arrays, original)
        self.assertFalse(report["selection"]["proposal_eligible"])
        self.assertFalse(report["selection"]["original_proposal_eligible"])
        self.assertTrue(np.any(bounded["proposed_delta_log"] != 0))
        self.assertTrue(np.all(bounded["applied_delta_log"] == 0))
        self.assertTrue(np.all(bounded["half_applied_delta_log"] == 0))
        self.assertTrue(any(r["status"] == "proposal_rejected" for r in report["region_applicability"]["regions"]))
        self.assertFalse(any(r["correction_applied"] for r in report["region_applicability"]["regions"]))

    def test_endpoint_guard_uses_uniform_energy_span_on_uneven_native_axis(self):
        e = self.inputs[0].copy()
        t = np.linspace(0, 1, e.size)
        e = e[0]+(e[-1]-e[0])*(0.2*t+0.8*t*t)
        raw = {name:value.copy() for name,value in self.raw_arrays.items()}
        raw["energy"] = e.copy()
        bounded, report = bound_raw_common_texture(e, raw, self.raw_report)
        width = 20*(e[-1]-e[0])/(len(e)-1)
        self.assertAlmostEqual(report["operator_guard"]["SG21_complete_window_span_ev"], width)
        edges = (e < e[0]+width) | (e > e[-1]-width)
        self.assertTrue(np.all(bounded["support_weight"][edges] == 0))
        self.assertGreater(np.count_nonzero(e < e[0]+width), 21)
        self.assertEqual(report["region_applicability"]["scope"], "optional_raw_common_fine_correction")

    def test_cancellation_and_invalid_frozen_parameters(self):
        with self.assertRaises(InterruptedError):
            bound_raw_common_texture(self.inputs[0], self.raw_arrays, self.raw_report, cancelled=lambda:True)
        bad = {name:value.copy() for name,value in self.raw_arrays.items()}
        bad["fold_parameters"][0, 0] += .01
        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            bound_raw_common_texture(self.inputs[0], bad, self.raw_report)
        bad = {name:value.copy() for name,value in self.raw_arrays.items()}
        bad["training_mask"] = bad["training_mask"].astype(int)
        with self.assertRaisesRegex(ValueError, "boolean"):
            bound_raw_common_texture(self.inputs[0], bad, self.raw_report)

    def test_bounded_cv_is_recomputed_instead_of_inheriting_success(self):
        # Full transform improves the high-amplitude guarded samples; its
        # remaining correction worsens the supported interiors. This must
        # reject despite all original folds passing.
        n = 1200
        e = np.linspace(1., 1.6, n)
        mask = np.ones(n, bool)
        for boundary in range(0, n, 200):
            mask[boundary:boundary+12] = False
        common = np.full(n, .001)
        target = common*.9
        for start in range(12, n, 200):
            common[start:start+9] = .1
            target[start:start+9] = common[start:start+9]*1.2
        delta = .1*common
        raw = dict(energy=e, raw_common_fine=common, target_train_fine=target,
                   training_mask=mask, parameters=np.array([0.,1.1]), proposed_delta_log=delta,
                   half_raw_common_fine=np.stack([common, common]),
                   half_proposed_delta_log=np.stack([delta, delta]),
                   fold_parameters=np.tile([0.,1.1], (6,1)),
                   raw_source_fine=np.tile(common, (2,1)),
                   half_raw_source_fine=np.tile(common, (2,2,1)),
                   gate_group_target_fine=np.tile(target, (3,1)),
                   repeat_target_fine=np.tile(target, (2,1)))
        original = dict(optimizer=dict(success=True, parameters=[0.,1.1]),
                        folds=[dict(block=i, parameters=[0.,1.1], optimizer_success=True) for i in range(6)])
        bounded, report = bound_raw_common_texture(e, raw, original)
        self.assertTrue(report["selection"]["original_proposal_eligible"])
        self.assertFalse(report["selection"]["bounded_all_domains_nonworsening"])
        self.assertFalse(report["selection"]["proposal_eligible"])
        self.assertTrue(np.any(bounded["proposed_delta_log"] != 0))
        self.assertTrue(np.all(bounded["applied_delta_log"] == 0))
        self.assertTrue(any(f["domains"]["eligible"]["relative_RMS_change"] > 0 for f in report["folds"]))

    def test_bounded_same_transform_reports_gate_conflict_and_half_errors(self):
        raw = {name:value.copy() for name,value in self.raw_arrays.items()}
        # Opposite-direction evidence in one fixed group must not disappear
        # behind the global target's successful bounded CV.
        raw["gate_group_target_fine"][0] = raw["raw_common_fine"]-raw["proposed_delta_log"]
        bounded, report = bound_raw_common_texture(self.inputs[0], raw, self.raw_report)
        checks = report["same_transform_checks"]
        self.assertTrue(checks["uses_proposed_not_applied"])
        self.assertFalse(checks["refitted"])
        conflicting = checks["gate_groups"][0]["domains"]
        for domain in ("full", "eligible", "covered"):
            self.assertGreater(conflicting[domain]["relative_RMS_change"], 0)
        self.assertEqual(len(conflicting["blocks"]), 6)
        before = raw["half_raw_common_fine"][0]
        after = before+bounded["half_proposed_delta_log"][0]
        covered = bounded["support_weight"] > 0
        expected = float(np.sqrt(np.mean((after[covered]-raw["target_train_fine"][covered])**2)))
        self.assertEqual(checks["pn_halves"][0]["common"]["covered"]["after"], expected)
        self.assertEqual(set(checks["sources"]), {"P", "N"})
        self.assertEqual(len(checks["training_repeats"]), 2)
        json.dumps(report, allow_nan=False)

    def test_protocol_is_json_safe_and_explicit_about_global_spline(self):
        json.dumps(BOUNDED_TEXTURE_PROTOCOL, allow_nan=False)
        self.assertFalse(BOUNDED_TEXTURE_PROTOCOL["refit"])
        self.assertTrue(any("global coefficients" in note for note in BOUNDED_TEXTURE_PROTOCOL["limitations"]))


if __name__ == "__main__":
    unittest.main()
