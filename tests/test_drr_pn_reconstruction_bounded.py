"""Raw-array orchestration and provenance tests, not scientific acceptance."""
import copy
import json
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

from core.drr_pn_reconstruction import pipeline
from core.drr_pn_reconstruction.render_bounded import HALF_LABELS, render_bounded
from core.drr_pn_reconstruction.render_conditioned import render_conditioned
from core.drr_pn_texture_compatibility import analyze_raw_common_texture
from core.drr_pn_texture_support import bound_raw_common_texture
from tests.test_drr_pn_reconstruction_generalized import fixture
from tests.test_drr_pn_texture_compatibility import synthetic_inputs


class BoundedReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.energy, baseline = fixture()
        _, cls.frames, cls.samples, cls.eligible = synthetic_inputs(n=len(cls.energy))
        pixel = np.arange(len(cls.energy))
        # Distinct half spectra expose accidentally broadcasting the full delta.
        drift = .0004*np.sin(.61*pixel)
        cls.frames[:, :2] *= np.exp(drift)
        cls.frames[:, 2:] *= np.exp(-drift)
        cls.parents = {}
        for label, offset in (("full", 0.), ("first_half", .003), ("last_half", -.002)):
            current = {key: value.copy() for key, value in baseline.items()}
            current["original_corrected_source_log"] += offset
            current["original_aligned_source_log"] += offset
            for key in ("background", "baseline_reference_background", "raw_geometric_mean"):
                current[key] *= np.exp(offset)
            cls.parents[label] = render_conditioned(cls.energy, baseline=current)

    def context(self):
        n = len(self.energy)
        prepared = dict(energy=self.energy.copy(), source_feature_log=np.zeros((2, n)),
                        corrected_source_log=np.log(self.frames.mean(1)),
                        aligned_source_log=np.log(self.frames.mean(1)),
                        diagnostics={"test_source_fixture": True}, stages={})
        # Reversed dictionary order must not exchange first/last delta rows.
        deferred = np.zeros(n, bool)
        deferred[7:11] = True
        regions = dict(full={"label": "full", "deferred_exclusive_mask": deferred},
                       halves={"last_half": {"label": "last_half"}, "first_half": {"label": "first_half"}},
                       stages={"v45": {"arrays": {"v45_witness_eligible": self.eligible}}},
                       diagnostics={"test_region_fixture": True})
        return prepared, regions

    def run_helper(self, *, eligible=None, frames=None, samples=None, **kwargs):
        return render_bounded(self.energy, self.frames if frames is None else frames,
                              self.samples if samples is None else samples,
                              eligible_mask=self.eligible if eligible is None else eligible,
                              conditioned_results=self.parents, **kwargs)

    def test_pipeline_prepares_once_and_uses_current_raw_mask_without_io(self):
        prepared, regions = self.context()
        original_frames = self.frames.copy()
        original_samples = self.samples.copy()
        original_mask = self.eligible.copy()
        events = []
        with ExitStack() as stack:
            sources = stack.enter_context(patch(
                "core.drr_pn_reconstruction.source_pipeline.prepare_sources", return_value=prepared))
            classify = stack.enter_context(patch(
                "core.drr_pn_reconstruction.regions_pipeline.prepare_regions", return_value=regions))
            renderer = stack.enter_context(patch(
                "core.drr_pn_reconstruction.render_conditioned.render_conditioned",
                side_effect=lambda energy, *, baseline, **kw: self.parents[baseline["label"]]))
            analyze = stack.enter_context(patch(
                "core.drr_pn_reconstruction.render_bounded.analyze_raw_common_texture",
                wraps=analyze_raw_common_texture))
            bound = stack.enter_context(patch(
                "core.drr_pn_reconstruction.render_bounded.bound_raw_common_texture",
                wraps=bound_raw_common_texture))
            for name in ("builtins.open", "io.open", "os.open"):
                stack.enter_context(patch(name, side_effect=AssertionError("raw API must not read stored data")))
            arrays, report = pipeline.reconstruct_background(
                self.energy, self.frames.mean(1), self.frames, self.samples,
                method="conditioned_bounded", progress=lambda *event: events.append(event))
        self.assertEqual(sources.call_count, 1)
        self.assertEqual(classify.call_count, 1)
        self.assertIs(classify.call_args.kwargs["source_result"], prepared)
        self.assertEqual(renderer.call_count, 3)
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(bound.call_count, 1)
        self.assertIs(analyze.call_args.args[1], self.frames)
        self.assertIs(analyze.call_args.args[2], self.samples)
        self.assertIs(analyze.call_args.args[3], self.eligible)
        np.testing.assert_array_equal(arrays["selector_deferred_core_mask"],
                                      regions["full"]["deferred_exclusive_mask"])
        self.assertFalse(np.shares_memory(arrays["selector_deferred_core_mask"],
                                         regions["full"]["deferred_exclusive_mask"]))
        self.assertEqual(report["algorithm_version"], pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION)
        self.assertFalse(report["automatic_application"])
        self.assertFalse(report["scientifically_validated"])
        self.assertFalse(report["historical_background_used"])
        self.assertFalse(report["BN_used"])
        self.assertTrue(report["bounded_selection"]["proposal_eligible"])
        json.dumps(report, allow_nan=False)
        for stage in ("raw_common_texture", "bounded_texture", "apply_bounded_texture"):
            self.assertIn((stage, "start"), events)
            self.assertIn((stage, "complete"), events)
        for value, old in ((self.frames, original_frames), (self.samples, original_samples),
                           (self.eligible, original_mask)):
            np.testing.assert_array_equal(value, old)
        for index, label in enumerate(HALF_LABELS):
            expected = self.parents[label]["arrays"]["background"]*np.exp(
                arrays["bounded_half_applied_delta_log"][index])
            np.testing.assert_array_equal(arrays[f"conditional_{label}_background"], expected)

    def test_half_deltas_use_own_common_same_parameters_and_own_parent(self):
        result = self.run_helper()
        arrays = result["arrays"]
        shift, gain = arrays["bounded_parameters"]
        common = arrays["raw_common_texture_half_raw_common_fine"]
        pixel = np.arange(self.energy.size)
        expected = gain*CubicSpline(pixel, common, axis=-1)(pixel+shift)-common
        np.testing.assert_array_equal(arrays["raw_common_texture_half_proposed_delta_log"], expected)
        np.testing.assert_array_equal(arrays["bounded_half_applied_delta_log"],
                                      expected*arrays["bounded_support_weight"][None])
        self.assertGreater(np.max(abs(expected[0]-expected[1])), 1e-5)
        for label, delta in (("full", arrays["bounded_applied_delta_log"]), *zip(
                HALF_LABELS, arrays["bounded_half_applied_delta_log"])):
            key = "background" if label == "full" else f"conditional_{label}_background"
            parent = self.parents[label]["arrays"]["background"]
            np.testing.assert_array_equal(arrays[key], parent*np.exp(delta))
            np.testing.assert_array_equal(arrays[key][delta == 0], parent[delta == 0])

    def test_final_metadata_does_not_relabel_parent_polynomials_or_masks(self):
        parents_before = copy.deepcopy(self.parents)
        result = self.run_helper()
        arrays, report = result["arrays"], result["diagnostics"]
        final_names = {"background", "background_log", "energy", "method_name",
                       "construction_type", "diagnostic_metadata_revision"}
        for key, value in self.parents["full"]["arrays"].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(arrays[f"conditioned_parent_{key}"], value)
                if key not in final_names:
                    self.assertNotIn(key, arrays)
        self.assertNotIn("integrated_log_polynomial_coefficients", arrays)
        self.assertNotIn("final_curvature_polynomial_coefficients", arrays)
        self.assertFalse(report["final_array_contract"]["final_polynomial_available"])
        np.testing.assert_array_equal(arrays["background_log"], np.log(arrays["background"]))
        for label in self.parents:
            for key, value in self.parents[label]["arrays"].items():
                np.testing.assert_array_equal(value, parents_before[label]["arrays"][key])
            self.assertEqual(report["conditioned_parent_diagnostics"][label], self.parents[label]["diagnostics"])
            self.assertFalse(report["render_diagnostics"][label]["final_polynomial_available"])

    def test_insufficient_support_keeps_exact_parent_for_all_three(self):
        result = self.run_helper(eligible=np.zeros(self.energy.size, bool))
        arrays, report = result["arrays"], result["diagnostics"]
        self.assertFalse(report["bounded_selection"]["proposal_eligible"])
        self.assertTrue(report["bounded_selection"]["reasons"])
        self.assertTrue(np.all(arrays["bounded_applied_delta_log"] == 0))
        self.assertTrue(np.all(arrays["bounded_half_applied_delta_log"] == 0))
        for label in self.parents:
            key = "background" if label == "full" else f"conditional_{label}_background"
            np.testing.assert_array_equal(arrays[key], self.parents[label]["arrays"]["background"])
        self.assertFalse(any(row["correction_applied"] for row in report["region_applicability"]["regions"]))
        json.dumps(report, allow_nan=False)

    def test_original_cv_rejection_cannot_be_promoted_by_bounding(self):
        _, frames, samples, _ = synthetic_inputs(n=len(self.energy), local_mismatch=True)
        result = self.run_helper(frames=frames, samples=samples)
        arrays, report = result["arrays"], result["diagnostics"]
        self.assertFalse(report["bounded_selection"]["original_proposal_eligible"])
        self.assertFalse(report["bounded_selection"]["proposal_eligible"])
        np.testing.assert_array_equal(arrays["background"], self.parents["full"]["arrays"]["background"])

    def test_cancellation_stops_before_bounding_and_inside_analysis(self):
        state = {"cancel": False}
        def progress(stage, event):
            if (stage, event) == ("raw_common_texture", "complete"):
                state["cancel"] = True
        with patch("core.drr_pn_reconstruction.render_bounded.bound_raw_common_texture") as bound:
            with self.assertRaises(InterruptedError):
                self.run_helper(cancelled=lambda: state["cancel"], progress=progress)
        bound.assert_not_called()
        calls = 0
        def cancel_during_analysis():
            nonlocal calls
            calls += 1
            return calls > 12
        with self.assertRaises(InterruptedError):
            self.run_helper(cancelled=cancel_during_analysis)
        self.assertGreater(calls, 12)

    def test_existing_four_methods_and_default_do_not_run_texture(self):
        prepared, regions = self.context()
        methods = ((None, "render_pipeline", "render_v66", pipeline.ALGORITHM_VERSION),
                   ("v66", "render_pipeline", "render_v66", pipeline.ALGORITHM_VERSION),
                   ("generalized", "render_generalized", "render_generalized", pipeline.GENERALIZED_ALGORITHM_VERSION),
                   ("local_evidence", "render_local_evidence", "render_local_evidence", pipeline.LOCAL_ALGORITHM_VERSION),
                   ("conditioned", "render_conditioned", "render_conditioned", pipeline.CONDITIONED_ALGORITHM_VERSION))
        for method, module, function, version in methods:
            with self.subTest(method=method), ExitStack() as stack:
                stack.enter_context(patch("core.drr_pn_reconstruction.source_pipeline.prepare_sources", return_value=prepared))
                stack.enter_context(patch("core.drr_pn_reconstruction.regions_pipeline.prepare_regions", return_value=regions))
                renderer = stack.enter_context(patch(f"core.drr_pn_reconstruction.{module}.{function}",
                                                     return_value=self.parents["full"]))
                stack.enter_context(patch("core.drr_pn_reconstruction.render_bounded.render_bounded",
                                          side_effect=AssertionError("existing method must not run texture")))
                kwargs = {} if method is None else {"method": method}
                arrays, report = pipeline.reconstruct_background(
                    self.energy, self.frames.mean(1), self.frames, self.samples, **kwargs)
            self.assertEqual(renderer.call_count, 3)
            self.assertEqual(report["algorithm_version"], version)
            self.assertNotIn("bounded_selection", report)
            np.testing.assert_array_equal(arrays["background"], self.parents["full"]["arrays"]["background"])
            np.testing.assert_array_equal(arrays["integrated_log_polynomial_coefficients"],
                                          self.parents["full"]["arrays"]["integrated_log_polynomial_coefficients"])


if __name__ == "__main__":
    unittest.main()
