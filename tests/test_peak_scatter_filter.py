"""Applied trajectory cleanup is conservative, explainable and reversible."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from core.peak_candidates import filter_heatmap
from core.peak_profiles import recommend_drr_filters
from tests.test_peak_auto_selection import _dataset


def settings(dataset, **overrides):
    return {**dataset['settings'], 'min_snr': 0., 'prominence': 0.,
            'min_width_mev': 0., 'min_distance_mev': 0., 'neighbor_support': 0,
            'max_width_mev': 0., 'max_per_row': 0, **overrides}


class ScatterFilterTests(unittest.TestCase):
    def test_cleanup_restores_cache_and_explains_each_removed_point(self):
        dataset = _dataset(np.arange(12.))
        points = dataset['candidates']['points']
        for p in points:
            if p['candidate_id'].startswith('isolated:'):
                p['snr'] = 4.
        next(p for p in points if p['candidate_id'] == 'isolated:0')['snr'] = 12.  # strong short feature
        before = deepcopy(dataset['candidates'])
        with patch('core.drr_peak_analysis.find_peaks', side_effect=AssertionError('redetected')):
            off = filter_heatmap(dataset, settings(dataset, remove_scatter=False))
            on = filter_heatmap(dataset, settings(dataset, remove_scatter=True))
            again = filter_heatmap(dataset, settings(dataset, remove_scatter=False))
        off_ids = {p['candidate_id'] for p in off['points']}
        on_ids = {p['candidate_id'] for p in on['points']}
        self.assertEqual(off_ids, {p['candidate_id'] for p in again['points']})
        self.assertTrue(off_ids - on_ids)
        self.assertIn('isolated:0', on_ids)
        self.assertTrue(all(p['candidate_id'] in on_ids for p in points if p['candidate_id'].startswith(('strong:', 'weak:'))))
        self.assertEqual({i for i, reason in on['filter_preview']['reasons'].items() if reason == 'trajectory'}, off_ids - on_ids)
        self.assertEqual(dataset['candidates'], before)
        self.assertTrue(on['scatter_filter']['applied'])
        self.assertEqual(on['filter_counts']['trajectory_removed'], len(off_ids - on_ids))

    def test_short_or_noise_only_and_unknown_snr_are_preserved(self):
        for gates in (np.arange(4.), np.arange(16.)):
            dataset = _dataset(gates)
            dataset['candidates']['points'] = [p for p in dataset['candidates']['points'] if p['candidate_id'].startswith('isolated:')]
            for p in dataset['candidates']['points']:
                p['snr'] = None
            off = filter_heatmap(dataset, settings(dataset))
            on = filter_heatmap(dataset, settings(dataset, remove_scatter=True))
            self.assertEqual(len(off['points']), len(on['points']))

    def test_noisy_broad_weak_feature_survives_alongside_scatter(self):
        dataset = _dataset(np.arange(24.))
        dataset['candidates']['points'] = [p for p in dataset['candidates']['points']
            if not (p['candidate_id'].startswith('weak:') and p['row_index'] in (4, 5, 13, 18))]
        for p in dataset['candidates']['points']:
            if p['candidate_id'].startswith('weak:'):
                i = p['row_index']
                p['energy'] = 1.06 + .00005 * i + (.00025 if i % 2 else -.00025)
                p['width_mev'] = 6.
                p['snr'] = 5.
            if p['candidate_id'].startswith('isolated:'):
                p['snr'] = 4.
        off = filter_heatmap(dataset, settings(dataset))
        on = filter_heatmap(dataset, settings(dataset, remove_scatter=True))
        weak_off = {p['candidate_id'] for p in off['points'] if p['candidate_id'].startswith('weak:')}
        weak_on = {p['candidate_id'] for p in on['points'] if p['candidate_id'].startswith('weak:')}
        self.assertGreaterEqual(len(weak_on), .85 * len(weak_off))
        self.assertLess(len(on['points']), len(off['points']))

    def test_mixed_noise_with_varied_widths_still_removes_scatter(self):
        dataset = _dataset(np.arange(40.))
        rng = np.random.default_rng(917)
        template = dataset['candidates']['points'][0]
        track = [p for p in dataset['candidates']['points'] if p['candidate_id'].startswith('weak:')]
        noise = [dict(template, candidate_id=f'random:{i}:{j}', row_index=i, y=float(i),
                      energy=1 + .1 * rng.random(), snr=4.,
                      width_mev=float(rng.uniform(1, 15)))
                 for i in range(40) for j in range(15)]
        dataset['candidates']['points'] = track + noise
        on = filter_heatmap(dataset, settings(dataset, remove_scatter=True))
        self.assertTrue(on['scatter_filter']['applied'])
        retained = {p['candidate_id'] for p in on['points']}
        self.assertGreaterEqual(sum(p['candidate_id'] in retained for p in track), 35)
        self.assertLess(sum(p['candidate_id'] in retained for p in noise), len(noise) * .6)

    def test_local_guard_does_not_use_quality_rejected_neighbors(self):
        dataset = _dataset(np.arange(12.))
        points = [p for p in dataset['candidates']['points'] if p['candidate_id'].startswith('strong:')]
        prototype = points[0]
        for row in (3, 4, 5):
            points.append(dict(prototype, candidate_id=f'candidate:{row}', row_index=row,
                y=float(row), energy=1.09, snr=4. if row == 4 else 2., width_mev=5.))
        dataset['candidates']['points'] = points
        result = filter_heatmap(dataset, settings(dataset, min_snr=3., remove_scatter=True))
        self.assertNotIn('candidate:4', {p['candidate_id'] for p in result['points']})
        self.assertEqual(result['filter_preview']['reasons']['candidate:4'], 'trajectory')

    def test_auto_proposals_ignore_existing_scatter_setting(self):
        dataset = _dataset(np.arange(12.))
        dataset['settings']['remove_scatter'] = False
        first, _ = recommend_drr_filters(dataset, dataset['candidates'])
        dataset['settings']['remove_scatter'] = True
        second, _ = recommend_drr_filters(dataset, dataset['candidates'])
        self.assertEqual(first, second)

    def test_unknown_snr_is_protected_when_reference_exists(self):
        dataset = _dataset(np.arange(12.))
        for p in dataset['candidates']['points']:
            if p['candidate_id'].startswith('isolated:'):
                p['snr'] = None
        result = filter_heatmap(dataset, settings(dataset, remove_scatter=True))
        self.assertTrue(result['scatter_filter']['applied'])
        self.assertIn('isolated:0', {p['candidate_id'] for p in result['points']})

    def test_random_population_falls_back_without_tracks(self):
        dataset = _dataset(np.arange(40.))
        rng = np.random.default_rng(123)
        points = dataset['candidates']['points'][:1]
        sample = points[0]
        dataset['candidates']['points'] = [dict(sample, row_index=row, y=float(row),
            energy=1 + .1 * rng.random(), snr=4., candidate_id=f'{row}:{j}')
            for row in range(40) for j in range(20)]
        result = filter_heatmap(dataset, settings(dataset, remove_scatter=True))
        self.assertFalse(result['scatter_filter']['applied'])
        self.assertEqual(result['scatter_filter']['reason'], 'insufficient_trajectory_evidence')
        self.assertEqual(result['filter_counts']['trajectory_removed'], 0)

    def test_range_uses_full_map_and_single_spectrum_disables_cleanup(self):
        dataset = _dataset(np.arange(12.))
        for p in dataset['candidates']['points']:
            if p['candidate_id'].startswith('isolated:'):
                p['snr'] = 4.
        roi = settings(dataset, remove_scatter=True, y_min=5., y_max=5.)
        result = filter_heatmap(dataset, roi)
        self.assertIn('weak:5', {p['candidate_id'] for p in result['points']})
        self.assertNotIn('isolated:5', {p['candidate_id'] for p in result['points']})
        dataset['analysis_scope'] = 'spectrum'
        result = filter_heatmap(dataset, roi)
        self.assertIn('isolated:5', {p['candidate_id'] for p in result['points']})
        recommended, _ = recommend_drr_filters(dataset, dataset['candidates'])
        self.assertFalse(recommended['remove_scatter'])


if __name__ == '__main__':
    unittest.main()
