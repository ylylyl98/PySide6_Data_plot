import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from core.peak_quality import prepare_quality_cache
from tests import test_peak_simple_workflow as fixtures
from ui_qt.quick.peak_analysis_workflow import analyze_dataset

class PeakPerformanceCorrectnessTests(unittest.TestCase):
    def test_repeat_fit_reuse_and_invalidation(self):
        d = analyze_dataset(fixtures.SimpleWorkflowTests().dataset(), 'find-fit')
        with patch('core.pl_peak_analysis.fit', side_effect=AssertionError('fit should be reused')):
            repeated = analyze_dataset(d, 'find-fit')
        self.assertEqual(d['result']['fits'], repeated['result']['fits'])
        self.assertEqual([(p.get('fit_center'), p.get('area')) for p in d['result']['points']],
                         [(p.get('fit_center'), p.get('area')) for p in repeated['result']['points']])
        repeated['settings']['model'] = 'Gaussian'
        with patch('core.pl_peak_analysis.fit', wraps=__import__('core.pl_peak_analysis', fromlist=['fit']).fit) as fit:
            analyze_dataset(repeated, 'find-fit')
            fit.assert_called_once()
        changed_range = deepcopy(d)
        changed_range['settings']['x_min'] += .005
        with patch('core.pl_peak_analysis.fit', wraps=__import__('core.pl_peak_analysis', fromlist=['fit']).fit) as fit:
            analyze_dataset(changed_range, 'find-fit')
            fit.assert_called_once()
        excluded = deepcopy(d)
        for branch in excluded['branches']:
            branch['enabled'] = False
        out = analyze_dataset(excluded, 'find-fit')
        self.assertEqual(out['result']['fits'], [])

    def test_quality_nearest_index_legacy_interpolated_and_input_unchanged(self):
        d = fixtures.SimpleWorkflowTests().dataset()
        from core.peak_candidates import detect_heatmap
        pool = detect_heatmap(d, d['settings'])
        pool.pop('quality'); p = pool['points'][0]; x = d['cube'].energy
        index = np.searchsorted(x, p['energy']); p['energy'] = float((x[index] + x[index+1]) / 2)
        before = deepcopy(pool)
        progress = []
        result = prepare_quality_cache(d, pool, progress=progress.append)
        self.assertTrue(progress)
        self.assertTrue(all(0 <= value <= 100 for value in progress))
        self.assertEqual(progress, sorted(progress))
        self.assertEqual(pool, before)
        target = result['points'][0]
        self.assertIsNotNone(target['noise_sigma'])
        self.assertEqual(int(np.argmin(abs(x-p['energy']))),
                         int(np.searchsorted(x, p['energy']) - (abs(x[np.searchsorted(x,p['energy'])-1]-p['energy']) <= abs(x[np.searchsorted(x,p['energy'])]-p['energy']))))

if __name__ == '__main__': unittest.main()
