"""A fitted but unapplied source feature must not align into the other source."""
from contextlib import ExitStack
import copy
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_reconstruction import pipeline, source_pipeline, source_v11, source_v12
from core.drr_pn_reconstruction.source_cache import SourcePreparationCache


class RejectedFeatureAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.energy = np.linspace(1.3, 2.2, 121)
        cls.common = 2.4 + .07 * (cls.energy - 1.7)
        cls.envelope = .018 + .012 * (cls.energy - 1.7)
        smooth = np.exp(np.array([cls.common + cls.envelope / 2,
                                  cls.common - cls.envelope / 2]))
        # Generate all native rendering/support metadata, then substitute the
        # fit outcome under test. Optimizer acceptance is deliberately fixed.
        cls.template = source_v11.fit_bundle(cls.energy, smooth)

    def fixture(self, *, feature_source=0, applied=False, ambiguous=False):
        previous = copy.deepcopy(self.template)
        model = previous["variants"]["bandpass_binary_full_tail"]
        feature = .12 / (1 + ((self.energy - 1.75) / .012) ** 2)
        differential = feature if feature_source == 0 else -feature
        logs = np.array([self.common + self.envelope / 2,
                         self.common - self.envelope / 2])
        logs[feature_source] += feature
        pn = np.exp(logs)
        raw_ratio = np.log(pn[0] / pn[1])
        source = -1 if ambiguous else feature_source
        row = dict(parameters=[1.75, .012, .12, 0.], source=source,
                   applied=bool(applied), fit_usable=bool(applied or ambiguous),
                   maximum_feature_log=.12,
                   fit_rejection_reasons=[] if applied or ambiguous else ["parameter_boundary"],
                   attribution_reason="bandpass_source_ambiguous_or_fit_unsupported")
        source_full = np.zeros_like(pn)
        if applied:
            source_full[feature_source] = feature
        model.update(pn=pn, logs=np.log(pn), raw_ratio=raw_ratio,
                     ratio=np.round(raw_ratio, 12), components=differential[None, :],
                     nuisance=self.envelope.copy(), model=self.envelope+differential,
                     error=raw_ratio-self.envelope-differential,
                     records=[row], source=np.array([source]),
                     tapers=np.ones((1, len(feature))), source_full=source_full,
                     source_local=source_full.copy(),
                     attribution_anchors=np.zeros((1, len(feature)), bool),
                     source_local_residuals=np.zeros((1, 2, len(feature))),
                     weight_anchors=np.zeros((1, len(feature)), bool),
                     model_sha="deterministic-rejected-component")
        model["diag"].update(components=copy.deepcopy(model["records"]),
                             detected_component_count=1,
                             applied_component_count=int(applied))
        return previous, pn, feature

    def v12(self, previous, pn, family):
        with patch.object(source_v12._v11, "fit_bundle", return_value=previous):
            return source_v12.generate(self.energy, pn, family=family)

    def test_unapplied_p_or_n_feature_stays_in_residual_and_out_of_other_source(self):
        for source in (0, 1):
            for ambiguous in (False, True):
                with self.subTest(source=source, ambiguous=ambiguous):
                    previous, pn, feature = self.fixture(feature_source=source,
                                                         ambiguous=ambiguous)
                    rendered = self.v12(previous, pn, "accepted_residual_weighted")
                    corrected = np.log(pn)-rendered["source_feature_log"]
                    residual = rendered["residual_intervention_native_log"]
                    alignment = corrected[0]-corrected[1]-residual
                    aligned = corrected+np.array([-.5, .5])[:, None]*alignment
                    # Envelope alignment is expected; an isolated feature in the
                    # opposite raw source is not part of that envelope.
                    np.testing.assert_allclose(aligned[1-source], self.common,
                                               rtol=0, atol=2e-15)
                    np.testing.assert_allclose(aligned[source], self.common+feature,
                                               rtol=0, atol=2e-15)
                    np.testing.assert_allclose(alignment, self.envelope,
                                               rtol=0, atol=2e-15)
                    np.testing.assert_array_equal(rendered["source_feature_log"],
                                                  np.zeros_like(pn))

    def test_applied_component_is_removed_without_changing_alignment(self):
        for source in (0, 1):
            with self.subTest(source=source):
                previous, pn, feature = self.fixture(feature_source=source, applied=True)
                rendered = self.v12(previous, pn, "accepted_residual_weighted")
                corrected = np.log(pn)-rendered["source_feature_log"]
                residual = rendered["residual_intervention_native_log"]
                q = corrected[0]-corrected[1]-residual
                np.testing.assert_allclose(residual, 0, rtol=0, atol=2e-15)
                np.testing.assert_allclose(q, self.envelope, rtol=0, atol=2e-15)
                np.testing.assert_allclose(corrected+np.array([-.5, .5])[:, None]*q,
                                           np.broadcast_to(self.common, pn.shape),
                                           rtol=0, atol=2e-15)
                expected = np.zeros_like(pn)
                expected[source] = feature
                np.testing.assert_array_equal(rendered["source_feature_log"], expected)

    def test_legacy_family_reproduces_the_half_feature_leak_for_comparison(self):
        previous, pn, feature = self.fixture()
        rendered = self.v12(previous, pn, "native_residual_weighted")
        corrected = np.log(pn)-rendered["source_feature_log"]
        q = corrected[0]-corrected[1]-rendered["residual_intervention_native_log"]
        aligned = corrected+np.array([-.5, .5])[:, None]*q
        np.testing.assert_allclose(aligned[1]-self.common, feature/2,
                                   rtol=0, atol=2e-15)

    def test_native_ratio_corrected_sources_residual_and_model_all_close(self):
        for source in (0, 1):
            for applied in (False, True):
                with self.subTest(source=source, applied=applied):
                    previous, pn, _feature = self.fixture(feature_source=source, applied=applied)
                    rendered = self.v12(previous, pn, "accepted_residual_weighted")
                    raw_ratio = np.log(pn[0]/pn[1])
                    np.testing.assert_allclose(
                        rendered["differential_model_log"]+rendered["differential_residual_log"],
                        raw_ratio, rtol=0, atol=2e-15)
                    np.testing.assert_array_equal(rendered["differential_residual_log"],
                                                  rendered["residual_intervention_native_log"])
                    corrected = np.log(pn)-rendered["source_feature_log"]
                    q = corrected[0]-corrected[1]-rendered["residual_intervention_native_log"]
                    np.testing.assert_allclose(q, rendered["differential_nuisance_log"],
                                               rtol=0, atol=2e-15)
                    np.testing.assert_allclose(np.exp(corrected)*np.exp(rendered["source_feature_log"]),
                                               pn, rtol=2e-15, atol=0)

    def test_rejected_feature_remains_local_with_another_accepted_component_and_risk_weights(self):
        for rejected_source in (0, 1):
            with self.subTest(rejected_source=rejected_source):
                previous, pn, rejected = self.fixture(feature_source=rejected_source)
                accepted_source = 1-rejected_source
                accepted = .08/(1+((self.energy-1.91)/.018)**2)
                logs = np.log(pn)
                logs[accepted_source] += accepted
                pn = np.exp(logs)
                model = previous["variants"]["bandpass_binary_full_tail"]
                differential = accepted if accepted_source == 0 else -accepted
                record = dict(parameters=[1.91, .018, .08, 0.], source=accepted_source,
                              applied=True, fit_usable=True, maximum_feature_log=.08,
                              fit_rejection_reasons=[], attribution_reason="matched_bandpass")
                model["records"].append(record)
                model["source_full"][accepted_source] = accepted
                model.update(pn=pn, logs=logs, raw_ratio=np.log(pn[0]/pn[1]),
                             ratio=np.round(np.log(pn[0]/pn[1]), 12),
                             components=np.vstack([model["components"], differential]),
                             model=model["model"]+differential,
                             error=np.log(pn[0]/pn[1])-model["model"]-differential,
                             source=np.array([rejected_source, accepted_source]),
                             source_local=model["source_full"].copy(),
                             tapers=np.ones((2, len(self.energy))),
                             attribution_anchors=np.zeros((2, len(self.energy)), bool),
                             source_local_residuals=np.zeros((2, 2, len(self.energy))),
                             weight_anchors=np.zeros((2, len(self.energy)), bool))
                model["diag"].update(components=copy.deepcopy(model["records"]),
                                     detected_component_count=2, applied_component_count=1)
                rendered = self.v12(previous, pn, "accepted_residual_weighted")
                weights = rendered["source_weights"]
                self.assertGreater(np.max(abs(weights-.5)), .1)
                corrected = np.log(pn)-rendered["source_feature_log"]
                q = corrected[0]-corrected[1]-rendered["residual_intervention_native_log"]
                aligned = corrected+np.array([-.5, .5])[:, None]*q
                expected = np.array([self.common, self.common])
                expected[rejected_source] += rejected
                np.testing.assert_allclose(q, self.envelope, rtol=0, atol=3e-15)
                np.testing.assert_allclose(aligned, expected, rtol=0, atol=3e-15)
                np.testing.assert_allclose(rendered["background"],
                                           np.exp(np.sum(weights*expected, axis=0)),
                                           rtol=2e-15, atol=0)

    def prepare_frozen_events(self, previous, pn, frames, **kwargs):
        # Keep the original rejected decision fixed: no new fit candidates are
        # introduced in later passes. Every render and all q/r/C/half algebra
        # still run through the production source pipeline.
        zeros = np.zeros_like(self.energy)
        with ExitStack() as stack:
            stack.enter_context(patch.object(source_v12._v11, "fit_bundle", return_value=previous))
            stack.enter_context(patch.object(source_pipeline.source_v20, "detect_events",
                                             return_value=([], [], zeros, zeros)))
            stack.enter_context(patch.object(source_pipeline.source_v24, "detect",
                                             return_value=([], [], {}, zeros, zeros)))
            stack.enter_context(patch.object(source_pipeline.source_v31, "detect_events",
                return_value=([], {}, np.zeros_like(pn), zeros,
                              np.zeros(len(self.energy), bool))))
            return source_pipeline.prepare_sources(self.energy, pn, frames, **kwargs)

    def test_prepared_and_conditional_half_sources_keep_unapplied_feature_local(self):
        drift = np.array([-.003, -.002, -.001, .001, .002, .003])
        source_scale = np.array([1., -.5])
        for source in (0, 1):
            with self.subTest(source=source):
                previous, pn, feature = self.fixture(feature_source=source)
                frames = pn[:, None, :]*(1+source_scale[:, None, None]*drift[None, :, None])
                prepared = self.prepare_frozen_events(previous, pn, frames)
                np.testing.assert_allclose(prepared["alignment_q_log"], self.envelope,
                                           rtol=0, atol=2e-15)
                np.testing.assert_allclose(prepared["aligned_source_log"][1-source], self.common,
                                           rtol=0, atol=2e-15)
                np.testing.assert_allclose(prepared["aligned_source_log"][source], self.common+feature,
                                           rtol=0, atol=2e-15)
                np.testing.assert_allclose(prepared["corrected_source_log"][0]-
                    prepared["corrected_source_log"][1],
                    prepared["residual_log"]+prepared["alignment_q_log"], rtol=0, atol=2e-15)
                for half_index, half_drift in enumerate((drift[:3].mean(), drift[3:].mean())):
                    expected = np.array([self.common, self.common])
                    expected[source] += feature
                    expected += np.log1p(source_scale*half_drift)[:, None]
                    np.testing.assert_allclose(prepared["half_aligned_source_log"][half_index],
                                               expected, rtol=0, atol=2e-15)
                legacy = self.prepare_frozen_events(previous, pn, frames,
                                                     alignment_policy="legacy_all_fitted")
                np.testing.assert_allclose(legacy["aligned_source_log"][1-source]-self.common,
                                           feature/2, rtol=0, atol=2e-15)


class AlignmentPolicyCacheTests(unittest.TestCase):
    def test_only_bounded_uses_accepted_policy_and_cached_policies_are_isolated(self):
        energy = np.linspace(1.4, 2., 31)
        frames = np.full((2, 4, len(energy)), 10.)
        pn = frames.mean(1)
        samples = np.full((2, 3, len(energy)), 20.)
        cache = SourcePreparationCache()
        policies = []

        def prepare(e, mean, _frames, *, alignment_policy, **_kwargs):
            policies.append(alignment_policy)
            marker = 1. if alignment_policy == "accepted_only" else 2.
            return dict(energy=e.copy(), source_feature_log=np.zeros_like(mean),
                        corrected_source_log=np.log(mean),
                        aligned_source_log=np.full_like(mean, marker),
                        stages={}, diagnostics={"raw_mean_refitted": True,
                                                "alignment_policy": alignment_policy})

        def regions(e, _pn, _frames, _samples, *, source_result, **_kwargs):
            return {"full": {"background": np.exp(source_result["aligned_source_log"][0]),
                             "deferred_exclusive_mask": np.zeros(len(e), bool)},
                    "halves": {},
                    "stages": {"v45": {"arrays": {"v45_witness_eligible": np.ones(len(e), bool)}}}}

        def render(_e, *, baseline, **_kwargs):
            return {"arrays": baseline, "diagnostics": {}}

        def bounded(_e, _frames, _samples, *, conditioned_results, **_kwargs):
            return {"arrays": conditioned_results["full"]["arrays"], "diagnostics": {}}

        with ExitStack() as stack:
            stack.enter_context(patch.object(source_pipeline, "prepare_sources", side_effect=prepare))
            stack.enter_context(patch("core.drr_pn_reconstruction.regions_pipeline.prepare_regions",
                                       side_effect=regions))
            for path in ("render_generalized.render_generalized", "render_conditioned.render_conditioned",
                         "render_local_evidence.render_local_evidence", "render_pipeline.render_v66"):
                stack.enter_context(patch("core.drr_pn_reconstruction."+path, side_effect=render))
            stack.enter_context(patch("core.drr_pn_reconstruction.render_bounded.render_bounded",
                                       side_effect=bounded))
            results = []
            for method in ("conditioned_bounded", "generalized", "conditioned", "local_evidence",
                           "v66", "conditioned_bounded", "generalized"):
                results.append(pipeline.reconstruct_background(
                    energy, pn, frames, samples, method=method, source_cache=cache,
                    source_cache_identity="identical-raw-input-and-code"))
        self.assertEqual(policies, ["accepted_only", "legacy_all_fitted"])
        for index, (arrays, report) in enumerate(results):
            is_bounded = index in (0, 5)
            np.testing.assert_array_equal(arrays["aligned_source_log"],
                                          np.full_like(pn, 1. if is_bounded else 2.))
            self.assertEqual(report["execution"]["source_preparation"]["reused"], index >= 2)
        self.assertNotEqual(results[0][1]["execution"]["source_preparation"]["cache_key"],
                            results[1][1]["execution"]["source_preparation"]["cache_key"])


if __name__ == "__main__":
    unittest.main()
