import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
from matplotlib.figure import Figure

from core.peak_workspace import create_dataset
from core.peak_plotting import draw_overlay, overlay_payload, resize_overlay
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_workflow import analyze_dataset


class PLAnalysisContractTests(unittest.TestCase):
    def test_automatic_filters_preserve_a_fast_moving_prominent_branch(self):
        from core.loader import DataCube
        from ui_qt.quick.peak_analysis_workflow import prepare_sources
        x = np.linspace(1., 1.15, 1501)
        gates = np.arange(15.)
        y = np.array([np.exp(-.5 * ((x - 1.04 - gate * .001) / .003) ** 2)
                      + .02 * np.exp(-.5 * ((x - 1.10) / .002) ** 2) for gate in gates])
        cube = DataCube(x, gates, y, 'Gate', 'PL', 'Intensity')
        d = prepare_sources([dict(cube=cube, kind='PL', channel='PL', source='moving', name='moving')])[0]
        out = analyze_dataset(d, 'detect')
        for row, gate in enumerate(gates):
            self.assertTrue(any(p['row_index'] == row and abs(p['energy'] - 1.04 - gate * .001) < .0003
                                for p in out['result']['points']), row)

    def test_auto_seed_budget_keeps_detection_and_fits_the_supported_subset(self):
        from core.loader import DataCube
        from ui_qt.quick.peak_analysis_workflow import prepare_sources
        x = np.linspace(1., 1.2, 2001)
        signal = np.exp(-.5 * ((x - 1.02) / .001) ** 2)
        for center in np.linspace(1.04, 1.18, 9):
            signal += .02 * np.exp(-.5 * ((x - center) / .001) ** 2)
        cube = DataCube(x, np.arange(5.), np.tile(signal, (5, 1)), 'Gate', 'PL', 'Intensity')
        d = prepare_sources([dict(cube=cube, kind='PL', channel='PL', source='many', name='many')])[0]
        def inspect_fit(dataset, result, settings, *args):
            from collections import Counter
            counts = Counter(p['row_index'] for p in result['points'] if p['status'] == 'accepted')
            self.assertTrue(counts)
            self.assertLessEqual(max(counts.values()), 8)
            return result
        with patch('core.pl_peak_analysis.fit', side_effect=inspect_fit):
            out = analyze_dataset(d, 'find-fit')
        self.assertGreater(len(out['result']['points']), 8 * 5)
        deferred = [p for p in out['result']['points'] if p.get('fit_status') == 'not fitted: automatic seed budget']
        self.assertTrue(deferred)
        self.assertTrue(out['result']['notices'])
        self.assertIsNone(d.get('result'))

    def dataset(self):
        d = create_dataset(sample(), 'PL', 'PL', 'fixture', 'PL')
        d['analysis_scope'] = 'spectrum'
        d['row'] = 4
        return d

    def test_changing_noise_and_detection_mode_requires_a_new_pool(self):
        d = analyze_dataset(self.dataset(), 'detect')
        original = deepcopy(d['candidates'])
        d['settings'].update(pl_weak_features=True, candidate_floor=0.)
        for action in ('filter', 'auto-filter'):
            with self.assertRaisesRegex(ValueError, 'Find candidates again'):
                analyze_dataset(d, action)
            self.assertEqual(d['candidates'], original)
        rebuilt = analyze_dataset(d, 'find-fit')
        self.assertTrue(rebuilt['candidates']['settings']['pl_weak_features'])

    def test_fit_reuse_keeps_quality_and_invalidates_older_algorithm(self):
        d = analyze_dataset(self.dataset(), 'find-fit')
        self.assertTrue(d['result']['fits'])
        with patch('core.pl_peak_analysis.fit', side_effect=AssertionError('unexpected refit')):
            again = analyze_dataset(d, 'find-fit')
        self.assertEqual(again['result']['fits'], d['result']['fits'])
        self.assertEqual(again['result']['points'][0]['fit_quality_status'], d['result']['points'][0]['fit_quality_status'])
        d['result']['pl_fit_version'] = -1
        with patch('core.pl_peak_analysis.fit', side_effect=RuntimeError('refit requested')):
            with self.assertRaisesRegex(RuntimeError, 'refit requested'):
                analyze_dataset(d, 'find-fit')

    def test_reviewed_fit_retains_detected_position_in_map_and_packet(self):
        d = analyze_dataset(self.dataset(), 'find-fit')
        p = d['result']['points'][0]
        p['fit_status'] = 'review: structured residual'
        p['fit_center'] = p['energy'] + .002
        packet = overlay_payload(d)
        ax = Figure().subplots()
        artists = draw_overlay(ax, packet, transform_y=lambda y: y * 2)
        review = next(a for a in artists if a.get_gid() == 'peak-fit-review')
        np.testing.assert_allclose(review.get_offsets()[0], [p['energy'], p['y'] * 2])
        resize_overlay(ax, 16)
        self.assertEqual(review.get_sizes()[0], 16)
        self.assertTrue(any(a.get_gid() == 'peak-detected-reference' for a in artists))


if __name__ == '__main__':
    unittest.main()
