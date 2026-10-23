"""Automatic references preserve measurements and label model-only intervals."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from core import drr_auto_composite as A
from core.drr_composite_background import CompositeBackground, CompositeRegion, CompositeSource, compose_background
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class AutoCompositeTests(unittest.TestCase):
    def setUp(self):
        self.e = np.linspace(1.4, 2., 601)
        self.base = 10000 * np.exp(.4 * (self.e - 1.7) - .7 * (self.e - 1.7)**2)
        self.p = self.source('P', self.base * np.exp(-.4 * np.exp(-((self.e - 1.61) / .003)**2)))
        self.n = self.source('N', self.base * np.exp(.3 * np.exp(-((self.e - 1.74) / .004)**2)))
        self.document = CompositeBackground((self.p, self.n), 'P')

    def source(self, key, values, **kwargs):
        return CompositeSource(key, key, (), 'all', self.e, tuple(format(v, '.17g') for v in self.e), values, **kwargs)

    def generated(self, suggestion):
        return next(s for s in suggestion.document.sources if s.key == suggestion.document.default_source)

    def test_selects_away_from_positive_and_negative_local_features(self):
        suggestion = A.suggest_composite_background(self.document)
        result = self.generated(suggestion)
        for center, source in ((1.61, self.n), (1.74, self.p)):
            index = np.argmin(abs(self.e - center))
            self.assertEqual(result.intensity[index], source.intensity[index])
        self.assertTrue(np.all((result.intensity == self.p.intensity) | (result.intensity == self.n.intensity)))
        self.assertFalse(suggestion.report['estimated_regions'])
        self.assertEqual(json.loads(result.generation_json)['algorithm'], 'adaptive_masked_regions_v1')
        self.assertEqual(len(suggestion.document.sources), 3)
        self.assertTrue(result.boundaries_ev)

    def test_three_sources_and_equal_ties_are_deterministic(self):
        clean = self.source('C', self.base)
        document = CompositeBackground((self.p, self.n, clean), 'C')
        first = A.suggest_composite_background(document)
        second = A.suggest_composite_background(document)
        np.testing.assert_array_equal(self.generated(first).intensity, clean.intensity)
        self.assertEqual(first.report, second.report)
        self.assertEqual(first.document.default_source, second.document.default_source)
        identical = CompositeBackground((clean, replace(clean, key='D')), 'D')
        same = A.suggest_composite_background(identical)
        self.assertEqual({segment['source_key'] for segment in same.report['source_segments']}, {'D'})

    def test_manual_regions_are_preserved_and_do_not_become_training_windows(self):
        region = CompositeRegion(1.72, 1.76, 'N')
        document = replace(self.document, regions=(region,))
        result = A.suggest_composite_background(document, settings=A.AutoCompositeSettings(reconstruct_gaps=True))
        self.assertEqual(result.document.regions, (region,))
        mask = (self.e >= region.xmin_ev) & (self.e < region.xmax_ev)
        np.testing.assert_array_equal(compose_background(result.document)[1][mask], self.n.intensity[mask])
        self.assertTrue(all(hi <= region.xmin_ev or lo >= region.xmax_ev for lo, hi in result.report['estimated_regions']))
        self.assertIn(region.xmin_ev, self.generated(result).boundaries_ev)
        self.assertIn(region.xmax_ev, self.generated(result).boundaries_ev)
        unlocked = A.suggest_composite_background(document, preserve_regions=False)
        self.assertFalse(unlocked.document.regions)

    def test_second_run_uses_original_sources_without_adding_derived_layers(self):
        first = A.suggest_composite_background(self.document)
        again = A.suggest_composite_background(first.document)
        self.assertEqual(first.document.default_source, again.document.default_source)
        self.assertEqual(len(again.document.sources), 3)
        self.assertEqual(again.report['input_source_keys'], ['P', 'N'])
        self.assertEqual(again.report['original_default_source'], 'P')
        np.testing.assert_array_equal(self.generated(first).intensity, self.generated(again).intensity)

    def test_missing_and_nonpositive_samples_are_not_interpolated(self):
        p, n = self.base.copy(), self.base.copy()
        p[100:130], n[300:320] = np.nan, -1.
        p[220:230], n[220:230] = np.nan, np.nan
        result = A.suggest_composite_background(CompositeBackground((self.source('P', p), self.source('N', n)), 'P'))
        y = self.generated(result).intensity
        np.testing.assert_array_equal(y[100:130], n[100:130])
        np.testing.assert_array_equal(y[300:320], p[300:320])
        self.assertTrue(np.isnan(y[220:230]).all())
        self.assertTrue(result.report['unsupported_regions'])
        self.assertFalse(result.report['estimated_regions'])

    def test_shared_feature_is_reported_as_unresolved(self):
        twin = replace(self.p, key='N')
        result = A.suggest_composite_background(CompositeBackground((self.p, twin), 'P'))
        self.assertTrue(any(lo <= 1.61 <= hi for lo, hi in result.report['unresolved_regions']))
        np.testing.assert_array_equal(self.generated(result).intensity, self.p.intensity)

    def test_source_seams_are_preserved_without_cross_seam_smoothing(self):
        sources = tuple(replace(s, intensity=self.base, boundaries_ev=(1.8,)) for s in (self.p, self.n))
        result = A.suggest_composite_background(CompositeBackground(sources, 'P'))
        self.assertIn(1.8, self.generated(result).boundaries_ev)
        self.assertTrue(any(lo <= 1.8 <= hi for lo, hi in result.report['unsupported_regions']))
        np.testing.assert_array_equal(self.generated(result).intensity, self.base)

    def test_unsafe_native_join_is_explicit_and_not_gain_corrected(self):
        high = replace(self.n, intensity=self.n.intensity * 1.2)
        result = A.suggest_composite_background(CompositeBackground((self.p, high), 'P'))
        self.assertTrue(any(item['unsafe'] for item in result.report['joins']))
        self.assertTrue(result.report['unresolved_regions'])
        self.assertFalse(result.report['estimated_regions'])
        values = self.generated(result).intensity
        self.assertTrue(np.all((values == self.p.intensity) | (values == high.intensity)))

    def test_invalid_settings_and_insufficient_sources(self):
        for kwargs in ({'max_gap_ev': 0}, {'anchor_width_ev': np.nan}, {'max_join_fraction': True},
                       {'reconstruct_gaps': 1}, {'widths_ev': (.07, .035)}, {'widths_ev': ()}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                A.AutoCompositeSettings(**kwargs)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            A.suggest_composite_background(CompositeBackground((self.p,), 'P'))

    def test_cancellation_is_observed_during_screening(self):
        cancellation = LoadCancellation()
        original = A._masked_trend
        def cancel(*args, **kwargs):
            cancellation.cancel()
            return original(*args, **kwargs)
        with load_scope(cancellation), patch.object(A, '_masked_trend', side_effect=cancel):
            with self.assertRaises(LoadCancelled):
                A.suggest_composite_background(self.document)

    def test_bridge_does_not_read_hidden_interior_and_retains_endpoints(self):
        l, r = 220, 260
        values = self.base.copy()
        hidden = values.copy()
        hidden[l + 1:r] = np.nan
        first, diagnostics = A._bridge_gap(self.e, values, l, r)
        second, other = A._bridge_gap(self.e, hidden, l, r)
        self.assertIsNotNone(first, diagnostics)
        self.assertEqual(diagnostics, other)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, self.base, rtol=1e-12)
        self.assertEqual(first[l], values[l])
        self.assertEqual(first[r], values[r])
        np.testing.assert_array_equal(first[:l], values[:l])
        np.testing.assert_array_equal(first[r + 1:], values[r + 1:])

    def test_bridge_rejects_wide_gaps_missing_anchors_and_source_boundaries(self):
        cases = [dict(start_idx=200, end_idx=280), dict(start_idx=2, end_idx=30),
                 dict(start_idx=220, end_idx=260, boundaries=(1.64,))]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                result, report = A._bridge_gap(self.e, self.base, **kwargs)
                self.assertIsNone(result)
                self.assertEqual(report['status'], 'unresolved')
        mask = np.ones(len(self.e), dtype=bool)
        mask[205:220] = False
        result, report = A._bridge_gap(self.e, self.base, 220, 260, anchor_mask=mask)
        self.assertIsNone(result)
        self.assertIn('anchor', report['reason'])

    def test_bridge_rejects_adjacent_endpoints_and_level_mismatch(self):
        result, report = A._bridge_gap(self.e, self.base, 280, 281)
        self.assertIsNone(result)
        self.assertIn('interior samples', report['reason'])
        stepped = self.base.copy()
        stepped[281:] *= 1.03
        result, report = A._bridge_gap(self.e, stepped, 276, 286)
        self.assertIsNone(result)
        self.assertIn('slopes', report['reason'])

    def texture_case(self):
        fine = .004 * np.sin(2 * np.pi * (self.e - self.e[0]) / .007)
        screen = dict(source_keys=['P', 'N'])
        texture = np.vstack((fine, fine))
        usable = np.ones_like(texture, dtype=bool)
        return screen, texture, usable

    def test_measured_texture_retains_phase_and_amplitude_when_switching_carriers(self):
        screen, texture, usable = self.texture_case()
        usable[0, 225:245] = False
        usable[1, 250:265] = False
        carrier, support, report = A._measured_texture(self.e, screen, texture, usable, 220, 270,
                                                       A.AutoCompositeSettings())
        self.assertEqual(report['status'], 'measured')
        self.assertTrue(support[220:271].all())
        np.testing.assert_allclose(carrier[220:271], texture[0, 220:271], atol=1e-17)
        self.assertAlmostEqual(np.corrcoef(carrier[220:271], texture[0, 220:271])[0, 1], 1.)
        self.assertAlmostEqual(np.std(carrier[220:271]) / np.std(texture[0, 220:271]), 1.)
        self.assertEqual({item['source_key'] for item in report['source_segments']}, {'P', 'N'})

    def test_measured_texture_requires_every_point_and_independent_sides(self):
        screen, texture, usable = self.texture_case()
        usable[:, 240:245] = False
        carrier, support, report = A._measured_texture(self.e, screen, texture, usable, 220, 270,
                                                       A.AutoCompositeSettings())
        self.assertIsNone(carrier)
        self.assertIsNone(support)
        self.assertTrue(report['texture_missing_regions'])
        for change in ('phase', 'amplitude', 'one_side'):
            with self.subTest(change=change):
                screen, texture, usable = self.texture_case()
                if change == 'phase':
                    texture[1] *= -1
                elif change == 'amplitude':
                    texture[1] *= 1.5
                else:
                    usable[1, :220] = False
                carrier, _, report = A._measured_texture(self.e, screen, texture, usable, 220, 270,
                                                          A.AutoCompositeSettings())
                self.assertIsNone(carrier)
                self.assertIn('both outer sides', report['reason'])

    def test_fine_component_excludes_weaker_feature_with_full_window_guard(self):
        fine = .004 * np.sin(2 * np.pi * self.e / .007)
        source = self.source('P', self.base * np.exp(fine))
        screen = A._screen(self.e, (source, replace(source, key='N')), A.AutoCompositeSettings(), ())
        screen['strong'][:] = False
        screen['score'][:] = 0.
        screen['score'][0, 240] = 2.6
        _, usable = A._texture_components(self.e, screen)
        guard = abs(self.e - self.e[240]) <= .010 - 1e-12
        self.assertFalse(usable[0, guard].any())
        self.assertTrue(usable[1, guard].all())

    def test_texture_missing_report_excludes_estimates_and_manual_locks(self):
        document = CompositeBackground((self.p, replace(self.n, intensity=self.n.intensity * 1.2)),
                                       'P', (CompositeRegion(1.64, 1.65, 'N'),))
        def estimate(e, raw, chosen, screen, locked, barriers, seeds, settings):
            estimated = (e >= 1.62) & (e <= 1.63)
            attempts = [dict(status='unresolved', texture=dict(texture_missing_regions=[[1.60, 1.66]])),
                dict(status='estimated', region_ev=[1.62, 1.63],
                     texture=dict(source_segments=[], texture_missing_regions=[]))]
            return raw, estimated, attempts
        with patch.object(A, '_estimate', side_effect=estimate):
            suggestion = A.suggest_composite_background(document,
                settings=A.AutoCompositeSettings(reconstruct_gaps=True))
        report = suggestion.report
        def covered(regions, energy):
            return any(lo <= energy <= hi for lo, hi in regions)
        for energy in (1.605, 1.635, 1.655):
            self.assertTrue(covered(report['texture_missing_regions'], energy))
            self.assertTrue(covered(report['unresolved_regions'], energy))
        for energy in (1.625, 1.645):
            self.assertFalse(covered(report['texture_missing_regions'], energy))
        self.assertIn('local envelope plus validated measured texture', report['note'])

    def test_sparse_texture_window_cannot_reach_past_feature_guard(self):
        energy = np.linspace(1.4, 2., 121)
        raw = np.full((2, len(energy)), 10000.)
        raw[:, 50] *= np.exp(.1)
        strong = np.zeros_like(raw, dtype=bool)
        strong[:, 50] = True
        screen = dict(raw=raw, valid=np.ones_like(strong), strong=strong, score=np.zeros_like(raw),
                      chunks=[((0, len(energy)),)] * 2)
        fine, usable = A._texture_components(energy, screen)
        self.assertGreater(abs(fine[0, 53]), .005)
        self.assertFalse(usable[:, 53].any())

    def test_estimate_restores_measured_texture_and_rejects_unchanged_seed(self):
        texture = .003 * np.sin(2 * np.pi * self.e / .007)
        truth = self.base * np.exp(texture)
        raw = np.vstack((truth.copy(), truth.copy()))
        raw[0, 240:250] *= np.exp(.05 * np.sin(np.linspace(0, np.pi, 10))**2)
        strong = np.zeros_like(raw, dtype=bool)
        strong[0, 240:250] = True
        screen = dict(raw=raw, valid=np.ones_like(strong), strong=strong, score=strong.astype(float) * 3,
                      supported=np.ones_like(strong), chunks=[((0, len(self.e)),)] * 2, source_keys=['P', 'N'])
        seeds = np.zeros(len(self.e), dtype=bool)
        seeds[244:246] = True
        settings = A.AutoCompositeSettings(reconstruct_gaps=True)
        values, estimated, attempts = A._estimate(self.e, raw[0], np.zeros(len(self.e), dtype=int),
            screen, np.zeros(len(self.e), dtype=bool), (), seeds, settings)
        self.assertTrue(estimated.any(), attempts)
        self.assertTrue(all(a['status'] == 'estimated' for a in attempts))
        np.testing.assert_allclose(values[estimated], truth[estimated], rtol=.0005)
        restored = np.log(values[estimated] / self.base[estimated])
        expected = texture[estimated]
        # The fixed 20 meV decomposition has a finite frequency response. It
        # must retain the oscillation, unlike a pure smooth-envelope bridge.
        self.assertGreater(np.corrcoef(restored, expected)[0, 1], .99)
        self.assertAlmostEqual(np.std(restored) / np.std(expected), 1., delta=.10)
        screen['raw'][:] = self.base
        screen['strong'][:] = False
        screen['score'][:] = 0.
        # An adjacent-sample seed expands to a physical window, but an already
        # matching spectrum must not be reported as a successful estimate.
        values, estimated, attempts = A._estimate(self.e, self.base, np.zeros(len(self.e), dtype=int),
            screen, np.zeros(len(self.e), dtype=bool), (), seeds, settings)
        self.assertFalse(estimated.any())
        self.assertGreaterEqual(np.diff(attempts[0]['requested_ev'])[0], .008 - 1e-12)
        self.assertIn('substantive change', attempts[0]['reason'])
        np.testing.assert_array_equal(values, self.base)


if __name__ == '__main__':
    unittest.main()
