"""Behavioral checks for selecting cached DRR filter proposals."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from core.loader import DataCube
from core.peak_candidates import filter_heatmap
from core.peak_profiles import recommend_drr_filters
from core.peak_workspace import create_dataset


def _dataset(gates):
    energy = np.linspace(1., 1.1, 501)
    cube = DataCube(energy, np.asarray(gates, dtype=float),
                    np.zeros((len(gates), len(energy))), 'Gate', 'Synthetic', 'DRR')
    dataset = create_dataset(cube, 'DRR', 'raw', 'synthetic', 'Synthetic')
    dataset['analysis_scope'] = 'map'
    points = []

    def add(row, energy, snr, label, polarity='peak'):
        points.append(dict(candidate_id=f'{label}:{row}', row_index=row,
            y=float(dataset['cube'].gate[row]), energy=energy, polarity=polarity,
            snr=snr, noise_sigma=1., prominence=snr, prominence_fraction=1.,
            width_mev=2.))

    for row, gate in enumerate(gates):
        add(row, 1.02 + gate * .0001, 12., 'strong')
        add(row, 1.06 - gate * .0001, 4., 'weak', 'dip')
    for row in range(len(gates)):
        # Isolated high-SNR artifacts should not be promoted by amplitude alone.
        add(row, 1.075 + row * .003, 8., 'isolated')
    dataset['candidates'] = dict(dataset_key=dataset['key'], points=points,
        settings=dict(detection_method='local', smoothing_window=11, candidate_floor=0.),
        method='Local', quality=dict(version=1))
    return dataset


class AutoSelectionTests(unittest.TestCase):
    def test_recommendation_preserves_weak_track_while_rejecting_isolated_artifacts(self):
        dataset = _dataset(np.arange(9.))
        before_pool = deepcopy(dataset['candidates'])
        before_settings = deepcopy(dataset['settings'])
        with patch('core.drr_peak_analysis.find_peaks', side_effect=AssertionError('redetected')):
            selected, summary = recommend_drr_filters(dataset, dataset['candidates'])
            points = filter_heatmap(dataset, {**dataset['settings'], **selected})['points']
        ids = {p['candidate_id'].split(':')[0] for p in points}
        assert {'strong', 'weak'} <= ids
        assert sum(p['candidate_id'].startswith('weak:') for p in points) >= 6
        assert sum(p['candidate_id'].startswith('isolated:') for p in points) <= 2
        assert dataset['candidates'] == before_pool
        assert dataset['settings'] == before_settings
        assert 'Automatic starting filters' in summary


    def test_recommendation_is_deterministic_with_reverse_scan_order_and_no_candidates(self):
        dataset = _dataset(np.arange(8., -1., -1.))
        first, _ = recommend_drr_filters(dataset, dataset['candidates'])
        second, _ = recommend_drr_filters(dataset, dataset['candidates'])
        assert first == second
        points = filter_heatmap(dataset, {**dataset['settings'], **first})['points']
        assert sum(p['candidate_id'].startswith('weak:') for p in points) >= 6
        empty = {**dataset['candidates'], 'points': []}
        settings, summary = recommend_drr_filters(dataset, empty)
        assert all(np.isfinite(value) for value in settings.values())
        assert summary


    def test_recommendation_can_be_cancelled_without_mutating_the_cache(self):
        dataset = _dataset(np.arange(9.))
        before = deepcopy(dataset['candidates'])
        with self.assertRaisesRegex((RuntimeError, InterruptedError), 'cancel'):
            recommend_drr_filters(dataset, dataset['candidates'], cancelled=lambda: True)
        assert dataset['candidates'] == before


    def test_missing_snr_and_single_spectrum_use_valid_fallback(self):
        dataset = _dataset([0.])
        dataset['analysis_scope'] = 'spectrum'
        for point in dataset['candidates']['points']:
            point['snr'] = None
        settings, summary = recommend_drr_filters(dataset, dataset['candidates'])
        self.assertEqual(settings['min_snr'], 0.)
        self.assertEqual(settings['neighbor_support'], 0)
        self.assertIn('Insufficient', summary)
        self.assertTrue(filter_heatmap(dataset, {**dataset['settings'], **settings})['points'])

    def test_narrow_roi_uses_neighbors_outside_view(self):
        dataset = _dataset(np.arange(9.))
        dataset['settings'].update(y_min=4., y_max=4.)
        settings, _ = recommend_drr_filters(dataset, dataset['candidates'])
        points = filter_heatmap(dataset, {**dataset['settings'], **settings})['points']
        self.assertTrue(any(p['candidate_id'] == 'weak:4' for p in points))
        self.assertTrue(all(p['row_index'] == 4 for p in points))

    def test_legacy_quality_cache_is_enriched_without_redetection(self):
        from ui_qt.quick.peak_analysis_workflow import analyze_dataset
        dataset = _dataset(np.arange(5.))
        dataset['candidates'].pop('quality')
        for point in dataset['candidates']['points']:
            point.pop('snr')
            point.pop('noise_sigma')
        before = deepcopy(dataset['candidates'])
        with patch('core.drr_peak_analysis.find_peaks', side_effect=AssertionError('redetected')):
            result = analyze_dataset(dataset, 'auto-filter')
        self.assertEqual(dataset['candidates'], before)
        self.assertTrue(all('snr' in point for point in result['candidates']['points']))

    def test_reference_does_not_identify_every_point_kept_by_one_proposal(self):
        from core.peak_auto_selection import _continuity_reference
        dataset = _dataset(np.arange(9.))
        points = dataset['candidates']['points']
        # A weak sloping line moves faster than the tight position proposals.
        for point in points:
            if point['candidate_id'].startswith('weak:'):
                point['energy'] = 1.05 + point['row_index'] * .0008
        reference = _continuity_reference(points, dataset['cube'].gate, .2)
        self.assertGreaterEqual(sum(key.startswith('weak:') for key in reference), 7)
        selected, _ = recommend_drr_filters(dataset, dataset['candidates'])
        kept = filter_heatmap(dataset, {**dataset['settings'], **selected})['points']
        self.assertGreaterEqual(sum(p['candidate_id'].startswith('weak:') for p in kept), 7)

    def test_reference_does_not_bridge_scan_gaps_or_mix_polarities(self):
        from core.peak_auto_selection import _continuity_reference
        dataset = _dataset([0., 1., 20., 21., 40., 41.])
        self.assertEqual(_continuity_reference(dataset['candidates']['points'], dataset['cube'].gate, .2), set())
        dataset = _dataset(np.arange(9.))
        points = [p for p in dataset['candidates']['points'] if p['candidate_id'].startswith('strong:')]
        for point in points:
            point['polarity'] = 'peak' if point['row_index'] % 2 else 'dip'
        self.assertEqual(_continuity_reference(points, dataset['cube'].gate, .2), set())
