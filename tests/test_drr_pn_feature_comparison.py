"""Synthetic ranking/retention regressions; no experimental CSV is opened."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from core.drr_pn_reconstruction import feature_comparison as comparison


class FeatureComparisonTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1., 2., 301)
        self.x = self.e-1.5
        self.g = np.linspace(-1., 1., 5)
        self.samples = self.sample_pair(.035, .035)
        self.parent = np.exp(.05*self.x**2)
        self.pn = np.array([np.ones(len(self.e)), np.exp(.15*self.x**2)])
        self.feature = dict(id=7, branch="dynamic", bounds_eV=[1.43, 1.57],
                            core_bounds_eV=[1.45, 1.55], anchor_bounds_eV=[1.43, 1.57],
                            center_ev=1.5, status="detected")

    def sample_pair(self, first_width, second_width):
        h = [np.exp(-.5*(self.x/width)**2) for width in (first_width, second_width)]
        return np.array([1000*(1+.03*self.g[:, None]*h[0]),
                         1002*(1+.0301*self.g[:, None]*h[1])])

    def compare(self, **overrides):
        values = dict(energy=self.e, training_samples=self.samples, pn_mean=self.pn,
                      parent_background=self.parent, features=[self.feature])
        values.update(overrides)
        return comparison.compare_feature_backgrounds(**values)

    def test_real_constructor_positive_proposal_and_public_scores(self):
        arrays, report = self.compare()
        row = report["comparison_features"][0]
        self.assertEqual(row["suggested_source"], "P")
        self.assertEqual(row["status"], "local_proposal_unvalidated")
        self.assertFalse(row["retain_parent"])
        self.assertEqual(report["suggested_feature_ids"], [7])
        self.assertEqual(report["suggested_regions"], [dict(xmin_ev=1.43, xmax_ev=1.57, source="P")])
        self.assertEqual(arrays["local_candidate_backgrounds"].shape, (1, 3, 301))
        self.assertEqual(arrays["transport_scores"].shape, (1, 5, 2, 2))
        self.assertEqual(arrays["stationary_signed_chords"].shape, (1, 5, 2))
        self.assertEqual(row["numerical_best_source_by_repeat"], ["P", "P"])
        self.assertTrue(row["numerical_best_source_consistent"])
        np.testing.assert_array_equal(arrays["candidate_local_ids"], [7])
        self.assertTrue(arrays["local_candidate_valid"].all())
        self.assertFalse(np.array_equal(arrays["local_candidate_backgrounds"][0, 1], self.pn[0]))
        for method in range(3):
            for domain, bounds in enumerate((row["core_bounds_eV"], row["full_support_bounds_eV"])):
                actual = comparison.score_dynamic_background(self.e, self.samples,
                    arrays["local_candidate_backgrounds"][0, method], bounds)
                np.testing.assert_allclose(actual, arrays["transport_scores"][0, method, :, domain], atol=0, rtol=0)
        self.assertFalse(report["scientifically_validated"])
        self.assertFalse(report["automatic_application"])
        self.assertFalse(report["heldout_used"])
        self.assertFalse(report["BN_used"])
        json.dumps(report, allow_nan=False)

    def test_rejected_compact_fit_keeps_diagnostics_but_blocks_otherwise_winning_candidate(self):
        # Isolate the acceptance hand-off from the renderer: the legacy fixture
        # is a proven numeric P winner, while this renderer rejects its fit.
        def rejected_renderer(*args, **kwargs):
            bg, arrays, detail = comparison.apply_feature_protection(*args, **kwargs)
            return bg, arrays, {**detail, 'fit_accepted': False,
                                'fit_rejection_reasons': ['incompatible curvature target']}
        with patch.object(comparison, '_constructor_for_mode', return_value=rejected_renderer):
            arrays, report = self.compare(construction_mode='compact_bounded')
        row = report['comparison_features'][0]
        self.assertEqual(row['numerical_best_source_by_repeat'], ['P', 'P'])
        self.assertEqual(row['suggested_source'], 'Parent')
        self.assertIn('numerical_best_candidate_curvature_fit_rejected', row['reasons'])
        self.assertEqual(report['suggested_regions'], [])
        np.testing.assert_array_equal(arrays['local_candidate_constructed'], [[True, True, True]])
        np.testing.assert_array_equal(arrays['local_candidate_accepted_fit'], [[True, False, False]])
        self.assertFalse(np.array_equal(arrays['local_candidate_backgrounds'][0, 1], self.parent))
        self.assertTrue(np.isfinite(arrays['transport_scores'][0, 1:3]).all())
        self.assertEqual(row['candidates'][1]['fit_rejection_reasons'], ['incompatible curvature target'])

    def test_real_compact_candidates_preserve_exterior_and_keep_rejected_fit_diagnostics(self):
        span = 20*(self.e[-1]-self.e[0])/(len(self.e)-1)
        left = self.feature["bounds_eV"][0]-2*span
        right = self.feature["bounds_eV"][1]+2*span
        u = (self.e-left)/(right-left)
        h = np.zeros_like(u)
        inside = (u > 0) & (u < 1)
        h[inside] = .64*(u[inside]*(1-u[inside]))**3
        # P is a compatible compact source; N exceeds the declared gain cap.
        # Both go through the real constructor and full-axis sample processing.
        pn = self.parent[None, :]*np.exp(np.array([h, 40*h]))
        arrays, report = self.compare(pn_mean=pn, construction_mode="compact_bounded")
        row = report["comparison_features"][0]
        self.assertEqual(report["protocol"]["construction_mode"], "compact_bounded")
        np.testing.assert_array_equal(arrays["local_candidate_constructed"], [[True, True, True]])
        np.testing.assert_array_equal(arrays["local_candidate_accepted_fit"], [[True, True, False]])
        self.assertEqual(row["fit_rejected_sources"], ["N"])
        self.assertNotEqual(row["suggested_source"], "N")
        for method in (1, 2):
            candidate = arrays["local_candidate_backgrounds"][0, method]
            detail = row["candidates"][method]["constructor_report"]
            self.assertEqual(candidate[~inside].tobytes(), self.parent[~inside].tobytes())
            self.assertTrue(detail["outside_support_native_parent_exact"])
            self.assertFalse(np.array_equal(candidate, self.parent))
            self.assertTrue(np.isfinite(arrays["transport_scores"][0, method]).all())
            self.assertTrue(np.isfinite(arrays["stationary_signed_chords"][0, method]).all())
            for domain, bounds in enumerate((row["core_bounds_eV"], row["full_support_bounds_eV"])):
                expected = comparison.score_dynamic_background(self.e, self.samples, candidate, bounds)
                np.testing.assert_array_equal(expected, arrays["transport_scores"][0, method, :, domain])
        rejected = row["candidates"][2]
        self.assertEqual(rejected["status"], "evaluated")
        self.assertFalse(rejected["fit_accepted"])
        self.assertTrue(rejected["fit_rejection_reasons"])
        self.assertGreater(rejected["constructor_report"]["regions"][0]["active_bound_count"], 0)
        json.dumps(report, allow_nan=False)

    def test_unknown_constructor_mode_is_not_silently_used_as_legacy(self):
        with self.assertRaisesRegex(ValueError, 'construction mode'):
            self.compare(construction_mode='compact_typo')

    def test_parent_can_win_even_when_raw_p_beats_raw_n(self):
        parent = np.ones(len(self.e))
        pn = np.exp(np.array([.05, .1])[:, None]*self.x**2)
        arrays, report = self.compare(parent_background=parent, pn_mean=pn)
        scores = arrays["transport_scores"][0]
        self.assertTrue(np.all(scores[3] < scores[4]))
        self.assertTrue(np.all(scores[0] < scores[1:3]))
        self.assertEqual(report["comparison_features"][0]["suggested_source"], "Parent")
        self.assertEqual(report["suggested_regions"], [])
        np.testing.assert_array_equal(arrays["local_candidate_backgrounds"][0, 0], parent)

    def test_raw_source_win_does_not_hide_actual_constructed_support_regression(self):
        # Flat P has zero raw transport. The local curvature edit retains an
        # affine integration tilt and worsens the complete support score.
        parent = np.exp(self.x+.3*self.x**2)
        pn = np.array([np.ones(len(self.e)), np.exp(5*self.x)])
        feature = {**self.feature, "bounds_eV": [1.4, 1.6]}
        arrays, report = self.compare(parent_background=parent, pn_mean=pn,
                                      training_samples=self.sample_pair(.06, .06), features=[feature])
        scores = arrays["transport_scores"][0]
        self.assertTrue(np.all(scores[3] < scores[0]))
        self.assertTrue(np.all(scores[3] < scores[4]))
        self.assertTrue(np.all(scores[1, :, 0] < scores[0, :, 0]))
        self.assertTrue(np.all(scores[1, :, 1] > scores[0, :, 1]))
        self.assertEqual(report["comparison_features"][0]["suggested_source"], "Parent")
        self.assertFalse(report["suggested_regions"])

    def test_pn_swap_changes_only_source_identity_and_candidate_order(self):
        first, first_report = self.compare()
        second, second_report = self.compare(pn_mean=self.pn[::-1])
        self.assertEqual(first_report["suggested_regions"][0]["source"], "P")
        self.assertEqual(second_report["suggested_regions"][0]["source"], "N")
        np.testing.assert_array_equal(first["local_candidate_backgrounds"][:, 0], second["local_candidate_backgrounds"][:, 0])
        np.testing.assert_array_equal(first["local_candidate_backgrounds"][:, 1], second["local_candidate_backgrounds"][:, 2])
        np.testing.assert_array_equal(first["local_candidate_backgrounds"][:, 2], second["local_candidate_backgrounds"][:, 1])
        np.testing.assert_array_equal(first["transport_scores"][:, [0, 1, 2, 3, 4]],
                                      second["transport_scores"][:, [0, 2, 1, 4, 3]])
        np.testing.assert_array_equal(first["stationary_signed_chords"][:, [0, 1, 2, 3, 4]],
                                      second["stationary_signed_chords"][:, [0, 2, 1, 4, 3]])
        self.assertEqual(second_report["comparison_features"][0]["numerical_best_source_by_repeat"], ["N", "N"])

    def test_equal_candidates_are_not_forced_to_p(self):
        arrays, report = self.compare(pn_mean=np.array([np.ones(len(self.e))]*2))
        self.assertEqual(report["comparison_features"][0]["suggested_source"], "Parent")
        self.assertIn("source_preference_is_tied_within_repeat_spread", report["comparison_features"][0]["reasons"])
        self.assertTrue(arrays["local_candidate_valid"].all())
        self.assertEqual(report["comparison_features"][0]["numerical_best_source_by_repeat"], [None, None])
        self.assertEqual(report["comparison_features"][0]["numerical_tied_sources_by_repeat"], [["P", "N"]]*2)

    def test_repeat_discrepancy_is_a_margin_not_a_truth_or_ci(self):
        self.assertAlmostEqual(comparison.improvement_margin([.3, .4], [.1, .1]), .05)
        arrays, report = self.compare(training_samples=self.sample_pair(.035, .12))
        row = report["comparison_features"][0]
        self.assertEqual(row["suggested_source"], "Parent")
        self.assertTrue(any("repeat_spread" in reason for reason in row["reasons"]))
        self.assertIn("not a statistical confidence interval", report["protocol"]["improvement_margin"])
        self.assertTrue(arrays["local_candidate_valid"].all())

    def test_stationary_anchor_in_transition_allows_diagnostics_but_blocks_adoption(self):
        # Dynamic core ends at 1.55. Its source support reaches 1.7033;
        # a stationary core beyond it has an anchor touching the transition.
        stationary = dict(id=8, branch="stationary", bounds_eV=[1.72, 1.74],
                          core_bounds_eV=[1.72, 1.74], anchor_bounds_eV=[1.69, 1.76],
                          center_ev=1.73, status="ambiguous_stationary")
        arrays, report = self.compare(features=[self.feature, stationary])
        row = report["comparison_features"][0]
        self.assertEqual(row["suggested_source"], "Parent")
        self.assertIn("full_support_intersects_ambiguous_stationary", row["reasons"])
        self.assertEqual(row["stationary_conflicts"][0]["interval"], "anchor_bounds_eV")
        self.assertEqual(row["construction_blockers"], [])
        self.assertEqual(row["adoption_blockers"], ["full_support_intersects_ambiguous_stationary"])
        self.assertEqual(row["numerical_best_source_by_repeat"], ["P", "P"])
        self.assertTrue(row["numerical_best_source_consistent"])
        self.assertEqual(report["constructor_calls"], 4)
        self.assertEqual(report["suggested_regions"], [])
        self.assertTrue(np.isfinite(arrays["transport_scores"][0, 1:3]).all())
        self.assertTrue(np.isfinite(arrays["stationary_signed_chords"]).all())
        np.testing.assert_array_equal(arrays["local_candidate_valid"], [[True, True, True]]*2)
        for i in range(2):
            self.assertFalse(np.array_equal(arrays["local_candidate_backgrounds"][i, 1], self.parent))
            self.assertEqual(report["comparison_features"][i]["suggested_source"], "Parent")
        self.assertIn("stationary_source_not_identified", report["comparison_features"][1]["adoption_blockers"])

    def test_stationary_chord_retains_two_repeat_signs_on_fixed_geometry(self):
        from core.processing import _sg_derivative_energy_rows
        h = np.exp(-.5*(self.x/.035)**2)
        samples = self.samples+20*np.array([1., -1.])[:, None, None]*h[None, None]
        arrays, report = self.compare(training_samples=samples, parent_background=np.ones(len(self.e)))
        row = report["comparison_features"][0]
        geometry = row["stationary_chord_geometry"]
        a, center, b = geometry["indices"]
        fraction = (self.e[center]-self.e[a])/(self.e[b]-self.e[a])
        backgrounds = [*arrays["local_candidate_backgrounds"][0], *self.pn]
        for method, bg in enumerate(backgrounds):
            maps = _sg_derivative_energy_rows(samples.mean(axis=1)/bg-1, self.e, 21, 2, 2)
            expected = maps[:, center]-(1-fraction)*maps[:, a]-fraction*maps[:, b]
            np.testing.assert_array_equal(arrays["stationary_signed_chords"][0, method], expected)
        self.assertLess(arrays["stationary_signed_chords"][0, 0, 0], 0)
        self.assertGreater(arrays["stationary_signed_chords"][0, 0, 1], 0)
        self.assertEqual(row["candidates"][0]["training_stationary_signed_chords"],
                         arrays["stationary_signed_chords"][0, 0].tolist())

    def test_collapsed_native_chord_anchors_are_undefined_not_zero(self):
        feature = {**self.feature, "anchor_bounds_eV": [1.5, 1.5001]}
        arrays, report = self.compare(features=[feature])
        row = report["comparison_features"][0]
        self.assertFalse(row["stationary_chord_geometry"]["valid"])
        self.assertTrue(np.isnan(arrays["stationary_signed_chords"]).all())
        self.assertEqual(row["candidates"][1]["training_stationary_signed_chords"], [None, None])
        self.assertEqual(row["suggested_source"], "P")  # New diagnostic does not alter the old selection gates.
        json.dumps(report, allow_nan=False)

    def test_invalid_geometry_and_constructor_failure_keep_parent_placeholders(self):
        edge = {**self.feature, "bounds_eV": [1.02, 1.06], "core_bounds_eV": [1.03, 1.05],
                "anchor_bounds_eV": [1.02, 1.06], "center_ev": 1.04}
        arrays, report = self.compare(features=[edge])
        self.assertEqual(report["constructor_calls"], 0)
        self.assertEqual(report["comparison_features"][0]["suggested_source"], "Parent")
        with patch.object(comparison, "apply_feature_protection", side_effect=ValueError("synthetic constructor failure")):
            arrays, report = self.compare()
        self.assertFalse(arrays["local_candidate_valid"][0, 1:].any())
        self.assertTrue(np.isnan(arrays["transport_scores"][0, 1:3]).all())
        self.assertTrue(np.isnan(arrays["stationary_signed_chords"][0, 1:3]).all())
        self.assertIn("construction_failed", report["comparison_features"][0]["candidates"][1]["reasons"][0])
        self.assertEqual(report["suggested_regions"], [])

    def test_same_geometry_reuses_candidates_but_preserves_both_ids(self):
        arrays, report = self.compare(features=[self.feature, {**self.feature, "id": 19}])
        self.assertEqual(report["constructor_calls"], 2)
        self.assertEqual(report["suggested_feature_ids"], [7, 19])
        np.testing.assert_array_equal(arrays["candidate_local_ids"], [7, 19])
        np.testing.assert_array_equal(arrays["local_candidate_backgrounds"][0], arrays["local_candidate_backgrounds"][1])
        self.assertFalse(report["features_merged"])

    def test_positive_finite_validation_cancellation_and_immutable_inputs(self):
        originals = [value.copy() for value in (self.e, self.samples, self.pn, self.parent)]
        with self.assertRaisesRegex(ValueError, "positive finite"):
            self.compare(pn_mean=np.zeros_like(self.pn))
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            self.compare(training_samples=np.full_like(self.samples, np.nan))
        with self.assertRaises(InterruptedError):
            self.compare(cancelled=lambda: True)
        count = 0
        def cancel_during_work():
            nonlocal count
            count += 1
            return count >= 5
        with self.assertRaises(InterruptedError):
            self.compare(cancelled=cancel_during_work)
        old_feature = copy.deepcopy(self.feature)
        self.compare()
        for actual, expected in zip((self.e, self.samples, self.pn, self.parent), originals):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.feature, old_feature)

    def test_gate_constant_samples_have_undefined_dynamic_scores_and_retain(self):
        samples = np.full_like(self.samples, 1000.)
        arrays, report = self.compare(training_samples=samples)
        self.assertTrue(np.isnan(arrays["transport_scores"]).all())
        self.assertEqual(report["comparison_features"][0]["suggested_source"], "Parent")
        self.assertIn("parent_transport_undefined", report["comparison_features"][0]["reasons"])
        self.assertEqual(report["comparison_features"][0]["numerical_best_source_by_repeat"], [None, None])
        self.assertFalse(report["comparison_features"][0]["numerical_best_source_consistent"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
