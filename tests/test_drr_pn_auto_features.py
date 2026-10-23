"""Data-role isolation and final-selection guards, not scientific truth tests."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import erf

from core.drr_pn_reconstruction import auto_features


class LocalFeatureIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1., 2., 301)
        self.parent = np.full_like(self.e, 100.)
        self.frames = np.broadcast_to(self.parent, (2, 5, len(self.e))).copy()
        self.samples = np.broadcast_to(self.parent, (2, 5, len(self.e))).copy()

    def run_analysis(self, **kwargs):
        return auto_features.analyze_local_feature_sources(self.e, self.frames, self.samples, self.parent, **kwargs)

    def test_flat_inputs_retain_parent_exact_and_all_odd_frames(self):
        before = [value.copy() for value in (self.e, self.frames, self.samples, self.parent)]
        arrays, report = self.run_analysis()
        np.testing.assert_array_equal(arrays['background'], self.parent)
        self.assertEqual(report['automatic_regions'], [])
        self.assertEqual(report['pn_frame_count'], 5)
        self.assertEqual(report['pn_half_frame_counts'], [2, 3])
        self.assertAlmostEqual(report['pn_half_noise_factor'], np.sqrt(6)/5)
        self.assertFalse(report['heldout_used'])
        self.assertFalse(report['production_default_allowed'])
        self.assertFalse(report['scientifically_validated'])
        json.dumps(report, allow_nan=False)
        for old, value in zip(before, (self.e, self.frames, self.samples, self.parent)):
            np.testing.assert_array_equal(old, value)

    def test_last_odd_frame_is_in_mean_and_hash(self):
        self.frames[:, -1] *= 1.02
        received = {}
        def compare(e, samples, pn, parent, features, **kwargs):
            received['pn'] = pn.copy()
            return {}, {'suggested_regions': [], 'suggested_feature_ids': [], 'comparison_features': []}
        with patch('core.drr_pn_reconstruction.feature_comparison.compare_feature_backgrounds', side_effect=compare):
            self.run_analysis()
        np.testing.assert_allclose(received['pn'], self.frames.mean(axis=1))
        self.assertFalse(np.array_equal(received['pn'], self.frames[:, :4].mean(axis=1)))

    def test_real_compact_flat_pipeline_returns_exact_parent_with_empty_regions(self):
        arrays, report = self.run_analysis(construction_mode='compact_bounded')
        self.assertEqual(arrays['background'].tobytes(), self.parent.tobytes())
        self.assertEqual(arrays['proposed_combined_background'].tobytes(), self.parent.tobytes())
        self.assertEqual(report['protocol']['construction_mode'], 'compact_bounded')
        self.assertEqual(report['automatic_regions'], [])
        self.assertEqual(report['local_features'], [])
        self.assertEqual(report['comparison']['constructor_calls'], 0)
        self.assertEqual(arrays['local_candidate_backgrounds'].shape, (0, 3, len(self.e)))
        self.assertEqual(arrays['local_candidate_accepted_fit'].shape, (0, 3))
        self.assertFalse(arrays['construction_support_mask'].any())
        self.assertFalse(np.any(arrays['construction_delta_log']))
        for key in ('constructor_report', 'proposed_constructor_report'):
            detail = report['experiment_construction'][key]
            self.assertEqual(detail['construction_mode'], 'compact_bounded')
            self.assertEqual(detail['regions'], [])
            self.assertFalse(detail['enabled'])
            self.assertTrue(detail['fit_accepted'])
            self.assertTrue(detail['outside_support_native_parent_exact'])
        self.assertFalse(report['production_default_allowed'])
        self.assertFalse(report['scientifically_validated'])
        json.dumps(report, allow_nan=False)

    def test_whole_pipeline_selects_dynamic_candidate_and_is_source_symmetric(self):
        # Analytic twice-integrated Gaussian gives a resolved, gate-changing D2
        # feature. Both sources are smooth and contain no stationary peaks.
        x = self.e - 1.5
        width = .035
        integrated = (x * width * np.sqrt(np.pi / 2) * erf(x / (np.sqrt(2) * width))
                      + width**2 * np.exp(-.5 * (x / width)**2))
        gates = np.linspace(-1., 1., 5)[:, None]
        samples = np.array([1000 * (1 + .03 * gates * integrated),
                            1002 * (1 + .0301 * gates * integrated)])
        sources = np.array([np.ones_like(x), np.exp(.15 * x**3)])
        frames = np.repeat(sources[:, None, :], 5, axis=1)
        parent = np.exp(.05 * x**2)
        arrays, report = auto_features.analyze_local_feature_sources(self.e, frames, samples, parent)
        explicit, explicit_report = auto_features.analyze_local_feature_sources(
            self.e, frames, samples, parent, construction_mode='integrated_curvature')
        swapped, swapped_report = auto_features.analyze_local_feature_sources(
            self.e, frames[::-1], samples, parent)
        # The new opt-in mode must leave the existing nonempty default path
        # byte-for-byte identical to explicitly requesting the legacy renderer.
        self.assertEqual(report, explicit_report)
        self.assertEqual(arrays.keys(), explicit.keys())
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], explicit[name], err_msg=name)
        self.assertEqual(len(report['automatic_regions']), 1)
        self.assertEqual(report['automatic_regions'][0]['source'], 'P')
        self.assertEqual(swapped_report['automatic_regions'][0]['source'], 'N')
        self.assertTrue(report['experiment_construction']['combination_check']['passed'])
        self.assertFalse(np.array_equal(arrays['background'], parent))
        np.testing.assert_array_equal(arrays['background'], swapped['background'])
        self.assertTrue(np.all(np.isfinite(arrays['background']) & (arrays['background'] > 0)))
        json.dumps(report, allow_nan=False)

    def test_rejects_extra_repeat_and_unmatched_parent_axis(self):
        with self.assertRaises(ValueError):
            auto_features.analyze_local_feature_sources(self.e, self.frames, np.repeat(self.samples, 2, axis=0), self.parent)
        with self.assertRaises(ValueError):
            auto_features.analyze_local_feature_sources(self.e, self.frames, self.samples, self.parent[:-1])

    def test_cancellation_before_processing_and_invalid_callback(self):
        with self.assertRaises(InterruptedError):
            self.run_analysis(cancelled=lambda: True)
        with self.assertRaises(TypeError):
            self.run_analysis(cancelled=True)

    def test_overlapping_supports_keep_feature_ids_and_reject_cluster(self):
        proposed = [dict(xmin_ev=a, xmax_ev=b, source='P') for a,b in ((1.3,1.32),(1.4,1.42),(1.8,1.82))]
        original = copy.deepcopy(proposed)
        regions, ids, excluded = auto_features._independent_supports(self.e, proposed, [11, 12, 19])
        self.assertEqual(ids, [19])
        self.assertEqual({row['feature_id'] for row in excluded}, {11,12})
        self.assertEqual(regions, [proposed[-1]])
        self.assertEqual(proposed, original)

    def test_combination_failure_returns_exact_parent_and_keeps_failed_candidate(self):
        region = dict(xmin_ev=1.45,xmax_ev=1.48,source='P')
        local = dict(id=7,branch='dynamic',bounds_eV=[1.45,1.48],core_bounds_eV=[1.45,1.48],
                     center_ev=1.465,anchor_bounds_eV=[1.45,1.48],status='detected_dynamic')
        self.frames[0] *= np.exp(.04*(self.e-1.5)**2)[None]
        with patch('core.drr_pn_reconstruction.feature_detection.detect_local_features', return_value=({}, {'local_features':[local]})), \
             patch('core.drr_pn_reconstruction.feature_comparison.compare_feature_backgrounds', return_value=({}, {'suggested_regions':[region],'suggested_feature_ids':[7]})), \
             patch.object(auto_features, '_combined_checks', return_value={'passed':False,'failures':[{'feature_id':7,'reason':'combined regression'}]}):
            arrays, report = self.run_analysis()
        np.testing.assert_array_equal(arrays['background'], self.parent)
        self.assertFalse(np.array_equal(arrays['proposed_combined_background'], self.parent))
        self.assertEqual(report['experiment_construction']['applied_regions'], [])
        self.assertEqual(report['experiment_construction']['proposed_regions'], [region])
        self.assertTrue(report['experiment_construction']['proposed_constructor_report']['enabled'])
        self.assertFalse(report['experiment_construction']['constructor_report']['enabled'])

    def test_undefined_combined_transport_is_rejected_without_nan_json(self):
        feature = dict(id=3,branch='dynamic',bounds_eV=[1.4,1.5],core_bounds_eV=[1.42,1.48],
                       center_ev=1.45,anchor_bounds_eV=[1.4,1.5])
        result = auto_features._combined_checks(self.e,self.samples,self.parent,self.parent,
            [feature],[3],np.ones(2),None)
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['failures']),2)
        json.dumps(result,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
