from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.peak_workspace import create_dataset, load_workspace, save_workspace
from tests.test_peak_workspace import sample


class PeakProfileTests(unittest.TestCase):
    def test_width_adaptive_defaults_restore_and_automatic_recommendation(self):
        from core.peak_candidates import detect_heatmap
        from core.peak_profiles import recommend_drr_filters
        from ui_qt.quick.peak_analysis_workflow import prepare_sources
        d = prepare_sources([dict(cube=sample('DRR'), kind='DRR', channel='second',
                                  source='adaptive.csv', name='Adaptive')])[0]
        self.assertTrue(d['settings']['neighbor_adaptive'])
        d['candidates'] = detect_heatmap(d, d['settings'])
        recommended, _ = recommend_drr_filters(d, d['candidates'])
        self.assertTrue(recommended['neighbor_adaptive'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'adaptive.npz'
            save_workspace(path, [d], d['key'])
            restored, _ = load_workspace(path)
            self.assertTrue(restored[0]['settings']['neighbor_adaptive'])
            # Legacy cached sessions retain strict matching even on a handoff.
            d['settings'].pop('neighbor_adaptive')
            save_workspace(path, [d], d['key'])
            for fresh in (False, True):
                restored, _ = load_workspace(path, fresh_snapshot=fresh)
                self.assertFalse(restored[0]['settings']['neighbor_adaptive'])
        d['analysis_scope'] = 'spectrum'
        recommended, _ = recommend_drr_filters(d, d['candidates'])
        self.assertFalse(recommended['neighbor_adaptive'])
        pl = create_dataset(sample(), 'PL', 'PL', 'pl.csv', 'PL')
        self.assertFalse(pl['settings']['neighbor_adaptive'])

    def test_new_data_uses_product_specific_balanced_filters(self):
        for kind, channel, snr, prominence, support in (
                ('PL', 'PL', 5., .05, 3), ('PL', 'KK', 5., .05, 3),
                ('DRR', 'raw', 5., .08, 3), ('DRR', 'second', 7., .10, 4)):
            with self.subTest(kind=kind, channel=channel):
                d = create_dataset(sample(kind), kind, channel, 'profile.csv', 'Profile')
                s = d['settings']
                self.assertEqual((s['min_snr'], s['prominence'], s['neighbor_support']),
                                 (snr, prominence, support))
                self.assertEqual(s['candidate_floor'], .01)
                self.assertEqual(s['smoothing_window'], 11)

    def test_default_width_and_spacing_follow_sampling_instead_of_fixed_mev(self):
        cube = sample()
        for x, width, spacing in ((np.linspace(1.6, 1.8, 501), 1.2, .8),
                                  (np.linspace(1.8, 1.7, 501), .6, .4)):
            d = create_dataset(replace(cube, energy=x), 'PL', 'PL', 'sampling.csv', 'Sampling')
            self.assertAlmostEqual(d['settings']['min_width_mev'], width)
            self.assertAlmostEqual(d['settings']['min_distance_mev'], spacing)

    def test_handoff_gets_new_profile_but_saved_manual_settings_survive(self):
        for channel, product in (
                ('raw', 'ΔR/R'), ('second', '2nd derivative'),
                ('raw', '1st derivative'), ('second', 'Mixed derivative')):
            with self.subTest(product=product), tempfile.TemporaryDirectory() as folder:
                d = create_dataset(sample('DRR'), 'DRR', channel, 'old-app.csv', 'Old app')
                d['product_label'] = product
                d['settings'].update(min_snr=6.2, prominence=.123, neighbor_support=2,
                                     min_width_mev=2.3, min_distance_mev=3.4,
                                     smoothing_window=17, candidate_floor=.023)
                original = deepcopy(d['settings'])
                path = Path(folder) / 'snapshot.npz'; save_workspace(path, [d], d['key'])
                saved, _ = load_workspace(path)
                self.assertEqual(saved[0]['settings'], original)
                fresh, _ = load_workspace(path, fresh_snapshot=True)
                self.assertEqual((fresh[0]['settings']['min_snr'], fresh[0]['settings']['prominence'],
                                  fresh[0]['settings']['neighbor_support']), (3., 0., 3))
                self.assertAlmostEqual(fresh[0]['settings']['min_width_mev'], .8)
                self.assertAlmostEqual(fresh[0]['settings']['min_distance_mev'], .8)
                self.assertEqual(fresh[0]['settings']['smoothing_window'], 17)
                self.assertEqual(fresh[0]['settings']['candidate_floor'], .023)

    def test_existing_candidate_cache_does_not_receive_new_defaults_on_handoff(self):
        from core.peak_candidates import detect_heatmap
        d = create_dataset(sample('DRR'), 'DRR', 'second', 'cached.csv', 'Cached')
        d['settings'].update(min_snr=3.2, prominence=.027, neighbor_support=2, min_width_mev=.2)
        d['candidates'] = detect_heatmap(d, d['settings'])
        original = deepcopy(d['settings'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'snapshot.npz'; save_workspace(path, [d], d['key'])
            loaded, _ = load_workspace(path, fresh_snapshot=True)
        self.assertEqual(loaded[0]['settings'], original)
        self.assertEqual(loaded[0]['candidates'], d['candidates'])
