"""Full-range detection and reversible filtering of DRR extrema."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from core.loader import DataCube
from core.peak_candidates import detect_heatmap, filter_heatmap
from core.peak_workspace import create_dataset, load_workspace, save_workspace


class PeakFilterWorkflowTests(unittest.TestCase):
    def test_find_applies_suggested_and_explicit_filters_without_discarding_cache(self):
        from ui_qt.quick.peak_analysis_workflow import analyze_dataset, prepare_sources
        energy = np.linspace(1., 2., 501)
        # Broad repeated signal plus narrow sample-to-sample extrema.
        row = np.sin(energy * 8 * np.pi) + .12 * np.where(np.arange(501) % 2, 1., -1.)
        cube = DataCube(energy, np.arange(4.), np.tile(row, (4, 1)),
                        gate_label='Gate', title='Noisy DRR', cbar_label='DRR')
        dataset = prepare_sources([dict(cube=cube, kind='DRR', channel='second', source='sample', name='sample')])[0]
        found = analyze_dataset(dataset, 'detect')
        self.assertEqual(found['candidates']['detected_rows'], [0, 1, 2, 3])
        self.assertGreater(len(found['candidates']['points']), len(found['result']['points']))
        self.assertFalse(found['filters_pending'])
        self.assertTrue(found['auto_filters'])
        self.assertIn('Automatic starting filters', found['auto_filter_summary'])
        found['settings'].update(min_snr=0., prominence=0., min_width_mev=0., min_distance_mev=0.,
                                 neighbor_support=0, max_width_mev=0., max_per_row=0)
        restored = analyze_dataset(found, 'filter')
        self.assertEqual(len(restored['result']['points']), len(found['candidates']['points']))
        # Explicit settings also apply to a fresh Find, rather than being reset.
        restored['settings']['min_width_mev'] = 500.
        restored['auto_filters'] = False
        narrowed = analyze_dataset(restored, 'detect')
        self.assertEqual(narrowed['result']['points'], [])
        self.assertEqual(narrowed['result']['settings']['min_width_mev'], 500.)
        self.assertEqual(len(narrowed['candidates']['points']), len(found['candidates']['points']))
        self.assertFalse(narrowed['filters_pending'])

    def test_auto_filter_adapts_to_cached_metrics_and_preserves_source(self):
        from core.peak_profiles import recommend_drr_filters
        from ui_qt.quick.peak_analysis_workflow import analyze_dataset, prepare_sources
        def make_dataset(step, snr):
            energy = 1. + np.arange(201) * step
            cube = DataCube(energy, np.arange(3.), np.tile(np.sin(np.arange(201) * .2), (3, 1)),
                            gate_label='Gate', title='DRR', cbar_label='DRR')
            dataset = prepare_sources([dict(cube=cube, kind='DRR', channel='second', source='s', name='s')])[0]
            pool = detect_heatmap(dataset, dataset['settings'])
            for point in pool['points']:
                point['snr'] = snr
            dataset['candidates'] = pool
            return dataset
        fine, coarse = make_dataset(.0001, 2.), make_dataset(.001, 10.)
        low, _ = recommend_drr_filters(fine, fine['candidates'])
        high, _ = recommend_drr_filters(coarse, coarse['candidates'])
        self.assertLess(low['min_snr'], high['min_snr'])
        self.assertLess(low['min_distance_mev'], high['min_distance_mev'])
        before_settings, before_pool = deepcopy(fine['settings']), deepcopy(fine['candidates'])
        filtered = analyze_dataset(fine, 'auto-filter')
        self.assertEqual(fine['settings'], before_settings)
        self.assertEqual(fine['candidates'], before_pool)
        self.assertEqual(filtered['settings']['min_snr'], low['min_snr'])
        self.assertEqual(filtered['settings']['prominence'], 0.)
        self.assertFalse(filtered['filters_pending'])
        self.assertTrue(filtered['auto_filters'])

    def test_fresh_drr_handoff_uses_suggested_and_workspace_keeps_choices(self):
        from core.peak_profiles import drr_initial_filter_settings, filter_presets
        from ui_qt.quick.peak_analysis_workflow import prepare_sources
        energy = np.linspace(1., 2., 101)
        cube = DataCube(energy, np.array([0., 1.]), np.array([np.sin(energy * 30)] * 2),
                        gate_label='Gate', title='Sample', cbar_label='Signal')
        with TemporaryDirectory() as folder:
            for kind, channel in (('DRR', 'raw'), ('PL', 'PL')):
                dataset = prepare_sources([dict(cube=cube, kind=kind, channel=channel,
                                                source=kind, name=kind)])[0]
                path = Path(folder) / (kind + '.npz')
                save_workspace(path, [dataset], dataset['key'])
                fresh = load_workspace(path, fresh_snapshot=True)[0][0]
                if kind == 'DRR':
                    self.assertEqual(fresh['settings']['candidate_floor'], 0)
                    for field, value in drr_initial_filter_settings(fresh['cube']).items():
                        self.assertEqual(fresh['settings'][field], value, field)
                else:
                    from core.pl_peak_profiles import pl_initial_filter_settings
                    preset = pl_initial_filter_settings(fresh['cube'])
                    for field, value in preset.items():
                        self.assertEqual(fresh['settings'][field], value, field)
                dataset['settings']['prominence'] = .37
                dataset['settings']['candidate_floor'] = .23
                save_workspace(path, [dataset], dataset['key'])
                restored = load_workspace(path, fresh_snapshot=False)[0][0]
                self.assertEqual(restored['settings']['prominence'], .37)
                self.assertEqual(restored['settings']['candidate_floor'], .23)

    def test_full_map_cache_restores_ranges_and_polarities_without_detection(self):
        energy = np.linspace(1.0, 2.0, 501)
        gates = np.arange(4, dtype=float)
        values = np.array([np.sin(energy * 40 * np.pi + row * .02) for row in gates])
        cube = DataCube(energy, gates, values, gate_label='Gate', title='DRR', cbar_label='DRR')
        dataset = create_dataset(cube, 'DRR', 'raw', 'source', 'source')
        dataset['analysis_scope'] = 'map'
        settings = {**dataset['settings'], 'detection_method': 'local', 'candidate_floor': 0.,
                    'x_min': 1.2, 'x_max': 1.3, 'y_min': 1., 'y_max': 1.,
                    'polarity': 'peaks', 'prominence': 0., 'min_width_mev': 0.,
                    'max_width_mev': 0., 'min_snr': 0., 'neighbor_support': 0,
                    'min_distance_mev': 0., 'max_per_row': 0}
        before_values = dataset['cube'].Z.copy()
        before_settings = deepcopy(settings)
        pool = detect_heatmap(dataset, settings)
        dataset['candidates'] = pool
        before_pool = deepcopy(pool)
        self.assertEqual(pool['detected_rows'], [0, 1, 2, 3])
        self.assertEqual({p['polarity'] for p in pool['points']}, {'peak', 'dip'})
        self.assertTrue(any(p['energy'] < 1.2 for p in pool['points']))
        self.assertTrue(any(p['energy'] > 1.3 for p in pool['points']))
        self.assertGreater(sum(p['row_index'] == 0 and p['polarity'] == 'peak' for p in pool['points']), 8)
        with patch('core.drr_peak_analysis.find_peaks', side_effect=AssertionError('Detection ran again')):
            narrow = filter_heatmap(dataset, settings)
            dips = filter_heatmap(dataset, {**settings, 'polarity': 'dips'})
            full = filter_heatmap(dataset, {**settings, 'x_min': 1., 'x_max': 2.,
                                            'y_min': 0., 'y_max': 3., 'polarity': 'both'})
        self.assertTrue(narrow['points'])
        self.assertTrue(dips['points'])
        self.assertEqual({p['polarity'] for p in narrow['points']}, {'peak'})
        self.assertEqual({p['polarity'] for p in dips['points']}, {'dip'})
        self.assertEqual({p['candidate_id'] for p in full['points']},
                         {p['candidate_id'] for p in pool['points']})
        self.assertEqual(pool, before_pool)
        self.assertEqual(settings, before_settings)
        np.testing.assert_array_equal(dataset['cube'].Z, before_values)


if __name__ == '__main__':
    unittest.main()
