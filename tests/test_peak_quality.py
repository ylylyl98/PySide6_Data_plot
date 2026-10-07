import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.loader import DataCube
from core.peak_workspace import create_dataset, load_workspace, save_workspace
from core.peak_candidates import detect_heatmap, filter_heatmap


class QualityTests(unittest.TestCase):
    def noisy_map(self):
        x = np.linspace(1.6, 1.8, 801)
        y = np.arange(15.)
        z = np.random.default_rng(72).normal(0, .035, (len(y), len(x)))
        for row in range(3, 12):
            z[row] += np.exp(-((x - 1.65 - row * .0002) / .002)**2)
            z[row] -= .8 * np.exp(-((x - 1.75 + row * .0002) / .002)**2)
        z[7] += .9 * np.exp(-((x - 1.70) / .002)**2)  # Strong but isolated artifact.
        return create_dataset(DataCube(x, y, z, 'Gate', 'Noisy map', 'DR/R'),
                              'DRR', 'second', 'noise.csv', 'Noise')

    def test_default_quality_rejects_noise_only_rows_and_isolated_artifact(self):
        d = self.noisy_map()
        d['candidates'] = detect_heatmap(d, d['settings'])
        r = filter_heatmap(d, d['settings'])
        self.assertLessEqual(len(r['points']), 22)
        # Four-of-five support trims the onset/end of this synthetic branch:
        # rows 3 and 11 have only three supporting spectra in their windows.
        self.assertEqual({p['row_index'] for p in r['points']}, set(range(4, 11)))
        for row in range(4, 11):
            for polarity, expected in [('peak', 1.65 + row*.0002), ('dip', 1.75 - row*.0002)]:
                self.assertTrue(any(p['row_index'] == row and p['polarity'] == polarity
                                    and abs(p['energy'] - expected) < .0006 for p in r['points']))
        self.assertFalse(any(p['row_index'] == 7 and p['polarity'] == 'peak'
                             and abs(p['energy'] - 1.70) < .003 for p in r['points']))

    def test_support_tolerance_allows_accumulated_drift_over_two_scan_steps(self):
        from core.peak_quality import supported_candidates
        points = [dict(row_index=i, energy=1.65+i*.0007, polarity='peak') for i in range(7)]
        kept = supported_candidates(points, np.arange(7.), 4, 1.)
        self.assertEqual(len(kept), 7)

    def test_legacy_cache_is_enriched_without_detection_or_mutating_input(self):
        from importlib.util import find_spec
        self.assertIsNotNone(find_spec('core.peak_quality'), 'Candidate quality cache is missing')
        from core import peak_quality
        self.assertTrue(hasattr(peak_quality, 'prepare_quality_cache'))
        d = self.noisy_map()
        pool = detect_heatmap(d, d['settings'])
        legacy = {k: copy.deepcopy(v) for k, v in pool.items() if k != 'quality'}
        for p in legacy['points']:
            p.pop('snr', None); p.pop('noise_sigma', None)
        before = copy.deepcopy(legacy)
        with patch('core.drr_peak_analysis.find_peaks', side_effect=AssertionError('Detector reran')):
            enriched = peak_quality.prepare_quality_cache(d, legacy)
            self.assertEqual(legacy, before)
            self.assertEqual([p['candidate_id'] for p in legacy['points']],
                             [p['candidate_id'] for p in enriched['points']])
            self.assertTrue(all(np.isfinite(p['snr']) for p in enriched['points']))
            self.assertIs(peak_quality.prepare_quality_cache(d, enriched), enriched)
            d['candidates'] = enriched
            strict = filter_heatmap(d, {**d['settings'], 'min_snr': 8})
            relaxed = filter_heatmap(d, {**d['settings'], 'min_snr': 0, 'neighbor_support': 0})
            self.assertLess(len(strict['points']), len(relaxed['points']))

    def test_neighbor_support_uses_scan_order_polarity_and_survives_range_crop(self):
        from importlib.util import find_spec
        self.assertIsNotNone(find_spec('core.peak_quality'), 'Neighbor support filtering is missing')
        from core import peak_quality
        self.assertTrue(hasattr(peak_quality, 'supported_candidates'))
        gates = np.array([2., 0., 4., 1., 3.])
        points = [dict(row_index=i, energy=1.65 + gate*.0002, polarity='peak')
                  for i, gate in enumerate(gates)]
        points += [dict(row_index=0, energy=1.7, polarity='dip'),
                   dict(row_index=3, energy=1.7, polarity='peak'),
                   dict(row_index=4, energy=1.7, polarity='peak')]
        selected = peak_quality.supported_candidates(points, gates, 3, .8)
        self.assertEqual(len(selected), 5)
        self.assertEqual({p['row_index'] for p in selected}, set(range(5)))
        self.assertTrue(all(p['energy'] < 1.66 for p in selected))
        # Scope cropping is applied after support so a one-row view retains
        # branches supported by the adjacent spectra outside the viewport.
        d = self.noisy_map(); d['candidates'] = detect_heatmap(d, d['settings'])
        cropped = filter_heatmap(d, {**d['settings'], 'y_min': 7, 'y_max': 7})
        self.assertEqual(len(cropped['points']), 2)

    def test_single_spectrum_is_not_discarded_for_missing_neighbors(self):
        d = self.noisy_map(); cube = d['cube']
        d = create_dataset(DataCube(cube.energy, np.array([7.]), cube.Z[7:8],
                                    'Gate', 'Single', 'DR/R'), 'DRR', 'second', 'single', 'Single')
        d['candidates'] = detect_heatmap(d, d['settings'])
        self.assertGreaterEqual(len(filter_heatmap(d, d['settings'])['points']), 3)

    def test_noise_enrichment_cancellation_keeps_original_cache_unchanged(self):
        from core.peak_quality import prepare_quality_cache
        d=self.noisy_map();pool=detect_heatmap(d,d['settings']);pool.pop('quality')
        before=copy.deepcopy(pool);progress=[]
        with self.assertRaisesRegex(RuntimeError,'cancel'):
            prepare_quality_cache(d,pool,cancelled=lambda:bool(progress),progress=progress.append)
        self.assertEqual(pool,before)

    def test_noise_enrichment_uses_cached_method_and_survives_restore(self):
        from core.peak_quality import prepare_quality_cache
        d=self.noisy_map();d['settings']['detection_method']='sg'
        pool=detect_heatmap(d,d['settings']);before=copy.deepcopy(pool)
        pool.pop('quality');d['settings']['detection_method']='local'
        enriched=prepare_quality_cache(d,pool)
        self.assertEqual(enriched['points'],before['points'])
        d['candidates']=enriched
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cached.npz';save_workspace(path,[d],d['key'])
            restored,_=load_workspace(path)
        self.assertEqual(restored[0]['candidates'],enriched)
        self.assertEqual(restored[0]['settings'],d['settings'])

    def test_neighbor_support_does_not_bridge_large_scan_gaps(self):
        from core.peak_quality import supported_candidates
        gates=np.array([0.,.1,.2,10.,10.1,10.2])
        points=[dict(row_index=i,energy=1.65,polarity='peak') for i in (2,3)]
        self.assertEqual(supported_candidates(points,gates,3,1.5),[])

    def test_restore_preserves_legacy_filtering_but_fresh_handoff_uses_quality(self):
        d = self.noisy_map()
        for key in ('min_snr', 'neighbor_support', 'neighbor_tolerance_mev'):
            d['settings'].pop(key, None)
        d['settings'].update(prominence=.05, min_width_mev=.5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.npz'; save_workspace(path, [d], d['key'])
            restored, _ = load_workspace(path)
            fresh, _ = load_workspace(path, fresh_snapshot=True)
        self.assertEqual(restored[0]['settings'].get('min_snr'), 0)
        self.assertEqual(restored[0]['settings'].get('neighbor_support'), 0)
        self.assertGreater(fresh[0]['settings'].get('min_snr', 0), 0)
        self.assertGreater(fresh[0]['settings'].get('neighbor_support', 0), 0)
