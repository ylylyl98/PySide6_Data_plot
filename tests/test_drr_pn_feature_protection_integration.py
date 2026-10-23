"""Protection request identity, orchestration and saved provenance."""
import copy
import json
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from core.drr_pn_reconstruction import app, pipeline
from core.drr_pn_reconstruction.feature_protection import apply_feature_protection
from core.drr_pn_region_report import describe_pn_regions
from core.drr_pn_store import DrrPnStore
from tests import test_drr_pn_reconstruction_app as fixtures


class ProtectionRequestTests(unittest.TestCase):
    setUp = fixtures.ReconstructionAppTests.setUp
    write = fixtures.ReconstructionAppTests.write

    def prepare(self, protection=()):
        return app.prepare_reconstruction_request(self.root, self.pair, self.files,
            training_files=self.training, method="conditioned_bounded", feature_protection=protection)

    def test_range_and_source_enter_key_but_check_measurements_do_not(self):
        plain = self.prepare()
        self.assertNotIn("feature_protection", plain.settings)
        ranges = [{"xmin_ev": 1.7, "xmax_ev": 1.75, "source": "P"}]
        protected = self.prepare(ranges)
        self.assertNotEqual(plain.key.fingerprint, protected.key.fingerprint)
        self.write(self.files[1], 97, 3, sample=True)
        self.assertEqual(protected.key.fingerprint, self.prepare(ranges).key.fingerprint)
        ranges[0]["source"] = "N"
        self.assertNotEqual(protected.key.fingerprint, self.prepare(ranges).key.fingerprint)
        self.assertEqual(protected.settings["feature_protection"][0]["source"], "P")

    def test_settings_are_forwarded_and_survive_save_and_read(self):
        ranges = [{"xmin_ev": 1.7, "xmax_ev": 1.75, "source": "P"}]
        request = self.prepare(ranges)
        captured = []
        def reconstruct(e, pn, frames, samples, **kwargs):
            captured.append(kwargs)
            return {"energy": e.copy(), "background": pn.mean(0)}, {
                "algorithm_version": pipeline.CONDITIONED_BOUNDED_ALGORITHM_VERSION,
                "settings": {**pipeline.CONDITIONED_BOUNDED_PROTOCOL, "feature_protection": ranges},
                "scientifically_validated": False,
            }
        with patch.object(pipeline, "reconstruct_background", side_effect=reconstruct):
            result = app.generate_reconstruction_background(self.root, self.pair, request=request)
        self.assertEqual(captured[0]["feature_protection"], ranges)
        store = DrrPnStore(self.root)
        version = store.save(result.document, request.key)
        loaded = store.list_versions(self.pair)[0]
        self.assertEqual(loaded.version_id, version.version_id)
        self.assertEqual(loaded.settings["feature_protection"], ranges)
        self.assertFalse(result.report["scientifically_validated"])

    def test_unsupported_method_rejects_instead_of_silently_ignoring(self):
        with self.assertRaisesRegex(ValueError, "conditioned_bounded"):
            app.prepare_reconstruction_request(self.root, self.pair, self.files,
                training_files=self.training, method="generalized",
                feature_protection=[{"xmin_ev": 1.7, "xmax_ev": 1.75, "source": "P"}])


class ProtectionPipelineTests(unittest.TestCase):
    def test_regional_report_keeps_prior_evidence_separate_from_explicit_choice(self):
        e = np.linspace(1., 2., 501)
        report = dict(scientifically_validated=False,
            region_applicability=dict(schema_version=1, regions=[dict(
                energy_bounds=[1.3, 1.7], correction_applied=True, source="N")]),
            feature_protection=dict(full=dict(regions=[dict(source="P",
                requested_bounds_eV=[1.4, 1.5], plateau_bounds_eV=[1.36, 1.54],
                support_bounds_eV=[1.32, 1.58])])) )
        original = copy.deepcopy(report)
        result = describe_pn_regions(e, report)
        self.assertEqual(original, report)
        fine = next(row for row in result.rows if row.operation == "Pre-protection fine correction")
        self.assertIn("Included before feature protection", fine.reason)
        protection = next(row for row in result.rows if row.operation == "Explicit feature protection")
        self.assertEqual(protection.source, "Raw P")
        self.assertEqual(protection.support, "User selected / requires review")
        self.assertEqual(len([r for r in result.rows if r.operation == "Feature protection transition"]), 2)
        self.assertTrue(any(r.record_path == "metadata_gap" for r in result.rows))
        self.assertIn("Not scientifically validated", result.scientific_review)

    def test_final_background_and_halves_use_raw_sources_and_frozen_gauge(self):
        e = np.linspace(1., 2., 501)
        frames = np.array([[100*np.exp(.03*np.sin(15*e)+drift*e) for drift in (0., .002, .01, .012)],
                           [104*np.exp(.08*np.sin(40*e)+drift*e) for drift in (0., -.002, -.01, -.012)]])
        samples = np.ones((2, 3, len(e))) * 120
        parents = {"background": 102*np.exp(.02*np.cos(10*e)),
                   "conditional_first_half_background": 101*np.exp(.021*np.cos(10*e)),
                   "conditional_last_half_background": 103*np.exp(.019*np.cos(10*e))}
        prepared = dict(energy=e, source_feature_log=np.zeros((2, len(e))),
            corrected_source_log=np.log(frames.mean(1)), aligned_source_log=np.log(frames.mean(1)),
            diagnostics={}, stages={})
        regions = dict(full={"deferred_exclusive_mask": np.zeros(len(e), bool)},
            halves={"first_half": {}, "last_half": {}},
            stages={"v45": {"arrays": {"v45_witness_eligible": np.zeros(len(e), bool)}}})
        bounded = {"arrays": copy.deepcopy(parents), "diagnostics": {
            "final_array_contract": {}, "render_diagnostics": {"full": {}, "first_half": {}, "last_half": {}}}}
        ranges = [{"xmin_ev": 1.4, "xmax_ev": 1.5, "source": "P"}]
        events = []
        with ExitStack() as stack:
            stack.enter_context(patch("core.drr_pn_reconstruction.source_pipeline.prepare_sources", return_value=prepared))
            stack.enter_context(patch("core.drr_pn_reconstruction.regions_pipeline.prepare_regions", return_value=regions))
            stack.enter_context(patch("core.drr_pn_reconstruction.render_conditioned.render_conditioned", return_value={"diagnostics": {}}))
            stack.enter_context(patch("core.drr_pn_reconstruction.render_bounded.render_bounded", return_value=bounded))
            arrays, report = pipeline.reconstruct_background(e, frames.mean(1), frames, samples,
                method="conditioned_bounded", feature_protection=ranges,
                progress=lambda *args: events.append(args))
        full, _extra, detail = apply_feature_protection(e, parents["background"], frames.mean(1), ranges)
        np.testing.assert_array_equal(arrays["background"], full)
        for label, group in (("first_half", frames[:, :2]), ("last_half", frames[:, 2:])):
            key = f"conditional_{label}_background"
            expected, _, _ = apply_feature_protection(e, parents[key], group.mean(1), ranges,
                frozen_parameters=detail["frozen_parameters"])
            np.testing.assert_array_equal(arrays[key], expected)
            np.testing.assert_array_equal(arrays[f"feature_protection_parent_{key}"], parents[key])
        self.assertEqual(report["settings"]["feature_protection"], ranges)
        self.assertFalse(report["final_array_contract"]["final_polynomial_available"])
        self.assertIn(("feature_protection", "complete"), events)
        self.assertNotIn("integrated_log_polynomial_coefficients", arrays)
        self.assertFalse(report["scientifically_validated"])
        json.dumps(report, allow_nan=False)
        rows = describe_pn_regions(e, report).rows
        self.assertTrue(any(row.operation == "Explicit feature protection" for row in rows))


if __name__ == "__main__":
    unittest.main()
