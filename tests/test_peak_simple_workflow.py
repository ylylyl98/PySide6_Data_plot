import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from core.peak_workspace import create_dataset
from tests.test_peak_workspace import sample
from ui_qt.quick.peak_analysis_workflow import analyze_dataset, add_seed


class SimpleWorkflowTests(unittest.TestCase):
    def dataset(self, kind='PL'):
        d = create_dataset(sample(kind), kind, 'PL', 'test', 'test')
        d['analysis_scope'] = 'spectrum'
        d['row'] = 4
        d['settings'].update(detection_method='local', min_snr=0, neighbor_support=5)
        return d

    def test_single_both_and_source_unchanged(self):
        d = self.dataset('DRR'); original = d['cube'].Z.copy(); settings = deepcopy(d['settings'])
        out = analyze_dataset(d, 'detect')
        self.assertEqual(out['candidates']['detected_rows'], [4])
        self.assertEqual({p['row_index'] for p in out['candidates']['points']}, {4})
        self.assertEqual({p['polarity'] for p in out['result']['points']}, {'peak', 'dip'})
        self.assertEqual(out['result']['settings']['neighbor_support'], 0)
        self.assertEqual(out['result']['settings']['y_min'], d['cube'].gate[4])
        self.assertEqual(out['settings'], settings)
        np.testing.assert_array_equal(d['cube'].Z, original)

    def test_find_fit_reuses_cache_and_refinds_on_row_change(self):
        d = analyze_dataset(self.dataset(), 'detect')
        with patch('core.peak_candidates.detect_heatmap', side_effect=AssertionError('unexpected detection')):
            fitted = analyze_dataset(d, 'find-fit')
        self.assertTrue(fitted['result']['fits'])
        self.assertAlmostEqual(fitted['result']['points'][0]['fit_center'], 1.65, places=3)
        fitted['row'] = 5
        switched = analyze_dataset(fitted, 'find-fit')
        self.assertEqual(switched['candidates']['detected_rows'], [5])
        self.assertEqual({p['row_index'] for p in switched['result']['points']}, {5})
        switched['analysis_scope'] = 'map'
        mapped = analyze_dataset(switched, 'find-fit')
        self.assertEqual(mapped['candidates']['detected_rows'], list(range(9)))

    def test_disabled_and_no_peaks(self):
        d = analyze_dataset(self.dataset(), 'detect')
        for b in d['branches']: b['enabled'] = False
        out = analyze_dataset(d, 'find-fit')
        self.assertFalse(out['result']['fits'])
        self.assertTrue(out['candidates']['points'])
        self.assertTrue(out['result']['notices'])
        self.assertTrue(all(not b['enabled'] for b in out['branches']))
        d = self.dataset(); d['settings']['prominence'] = 2
        out = analyze_dataset(d, 'find-fit')
        self.assertFalse(out['result']['fits'])
        self.assertTrue(out['result']['notices'])

    def test_manual_seed_before_detection_does_not_duplicate_fit_component(self):
        d = add_seed(self.dataset(), 4, 1.65)
        manual_id = d['branches'][0]['id']
        out = analyze_dataset(d, 'find-fit')
        near = [p for p in out['result']['points'] if abs(p['energy'] - 1.65) < .001]
        self.assertEqual(len(near), 1)
        self.assertEqual(near[0]['branch_id'], manual_id)
        self.assertTrue(out['result']['fits'])
        again = analyze_dataset(out, 'find-fit')
        self.assertEqual(len([p for p in again['result']['points'] if abs(p['energy'] - 1.65) < .001]), 1)

    def test_more_than_eight_peaks_retains_detection_without_fitting(self):
        from dataclasses import replace
        d = self.dataset()
        x = d['cube'].energy
        signal = sum(1 / (1 + (2 * (x - center) / .003) ** 2)
                     for center in np.linspace(1.615, 1.785, 10))
        d['cube'] = replace(d['cube'], Z=np.tile(signal, (len(d['cube'].gate), 1)))
        d['settings'].update(min_snr=0, prominence=.05, min_width_mev=.1, min_distance_mev=1)
        out = analyze_dataset(d, 'find-fit')
        self.assertEqual(len(out['result']['points']), 10)
        self.assertFalse(out['result']['fits'])
        self.assertTrue(any('More than 8' in notice for notice in out['result']['notices']))

    def test_cancel_preserves_input(self):
        d = analyze_dataset(self.dataset(), 'detect'); old = deepcopy(d['result'])
        with self.assertRaises(RuntimeError):
            analyze_dataset(d, 'find-fit', cancelled=lambda: True)
        self.assertEqual(d['result'], old)
        self.assertEqual(d['candidates']['detected_rows'], [4])

if __name__ == '__main__': unittest.main()
