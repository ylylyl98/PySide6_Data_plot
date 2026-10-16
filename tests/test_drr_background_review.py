from dataclasses import asdict, replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from core.drr_fitted_background import estimate_background, load_fitted_group, settings_for_source
from core.drr_background_cache import fit_cached
from core.drr_background_quality import suggest_regions, merge_regions
from core.drr_background_review import check_group, compare_methods
from tests.test_drr_fitted_background import SETTINGS, synthetic, write_csv


class BackgroundReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        environment = patch.dict(os.environ, DPTK_FITTED_CACHE_DIR=str(self.root/'cache'))
        environment.start()
        self.addCleanup(environment.stop)
        self.raw, self.background, _ = synthetic()
        self.signals = SimpleNamespace(emit=lambda *_: None)

    def fit(self, raw=None, settings=SETTINGS, axis='bg'):
        return fit_cached(raw or self.raw, settings, source='a.csv', signature=(1, 2),
                          source_path=self.root/'a.csv', y_axis=axis)

    def test_polynomial_recovers_known_background_and_reports_method(self):
        result, ref = estimate_background(self.raw, replace(SETTINGS, method='polynomial', polynomial_degree=2))
        np.testing.assert_allclose(ref.background(), self.background, rtol=1e-5)
        self.assertIn('polynomial', ref.metadata()['method'])
        self.assertEqual(ref.diagnostics['screened_gates'], len(self.raw.gate))

    def test_automatic_protection_fits_and_cache_retains_actual_regions(self):
        settings = replace(SETTINGS, protected_regions=(), automatic_protection=True)
        first, ref = self.fit(settings=settings)
        self.assertTrue(ref.settings.protected_regions)
        self.assertTrue(ref.settings.automatic_protection)
        for center in (1.6, 1.8):
            self.assertTrue(any(lo <= center <= hi for lo, hi in ref.settings.protected_regions))
        second, restored = self.fit(settings=settings)
        self.assertTrue(restored.cache_hit)
        self.assertEqual(restored.settings, ref.settings)
        np.testing.assert_array_equal(restored.fit_mask, ref.fit_mask)
        np.testing.assert_array_equal(first.Z, second.Z)
        altered = replace(self.raw, Z=self.raw.Z*(1+.25*np.exp(-.5*((self.raw.energy-1.7)/.004)**2)))
        _, new_ref = self.fit(raw=altered, settings=settings)
        self.assertTrue(any(lo <= 1.7 <= hi for lo, hi in new_ref.settings.protected_regions))

    def test_automatic_protection_handles_smooth_spectra_and_heldout_validation(self):
        smooth = replace(self.raw, Z=self.background)
        settings = replace(SETTINGS, protected_regions=(), automatic_protection=True)
        result, ref = self.fit(raw=smooth, settings=settings)
        self.assertEqual(ref.settings.protected_regions, ())
        self.assertTrue(ref.fit_mask.all())
        self.assertTrue(ref.diagnostics['warnings'])
        np.testing.assert_allclose(result.Z, 0, atol=1e-4)  # Spline curvature penalty permits a small envelope bias.
        write_csv(self.root/'smooth.csv', smooth)
        comparison = compare_methods(self.root, 'smooth.csv', 'bg', settings, 1, 21, 2,
                                     progress=self.signals, log=self.signals)
        self.assertIsNone(comparison['metrics']['background_max_difference_percent'])
        self.assertTrue(all(score['blocks'] > 0 for score in comparison['metrics']['validation']))

    def test_screening_detects_both_signs_on_unsampled_gate_without_shrinking_manual_regions(self):
        energy = self.raw.energy
        gate = np.arange(25.)
        background = np.repeat(self.background[:1], len(gate), axis=0)
        values = background.copy()
        for i, center in enumerate(np.linspace(1.65, 1.70, len(gate))):
            values[i] *= 1+.25*np.exp(-.5*((energy-center)/.004)**2)
        # Gate 13 would be missed by nine uniformly spaced representative gates.
        values[13] *= 1-.3*np.exp(-.5*((energy-1.87)/.004)**2)
        cube = replace(self.raw, gate=gate, Z=values)
        original = ((1.50, 1.54),)
        report = suggest_regions(cube, original)
        merged = report['merged_regions']
        for center in (1.65, 1.70, 1.87):
            self.assertTrue(any(lo <= center <= hi for lo, hi in merged), (center, merged))
        self.assertIn([1.5, 1.54], merged)
        self.assertEqual(report['screened_gates'], 25)
        self.assertEqual(suggest_regions(replace(self.raw, Z=self.background))['suggested_regions'], [])

    def test_cache_survives_calls_and_invalidates_on_values_axis_and_settings(self):
        with patch('core.drr_background_cache.estimate_background', wraps=estimate_background) as calculate:
            first, ref = self.fit()
            self.assertFalse(ref.cache_hit)
            second, cached = self.fit()
            self.assertTrue(cached.cache_hit)
            np.testing.assert_array_equal(first.Z, second.Z)
            self.assertEqual(calculate.call_count, 1)
            # Identical stat fields are insufficient: content is part of the key.
            self.fit(replace(self.raw, Z=self.raw.Z*1.01))
            self.fit(axis='tg')
            self.fit(settings=replace(SETTINGS, method='polynomial'))
            self.assertEqual(calculate.call_count, 4)
            with patch('core.drr_background_cache.CACHE_VERSION', 999):
                self.fit()
            self.assertEqual(calculate.call_count, 5)

    def test_dim_wing_noise_is_not_accumulated_into_protected_bands(self):
        rng = np.random.default_rng(672)
        values = np.repeat(self.background[:1], 201, axis=0)
        sigma = np.where(self.raw.energy < 1.55, .025, .001)
        values *= np.exp(rng.normal(size=values.shape)*sigma)
        cube = replace(self.raw, gate=np.arange(201.), Z=values)
        self.assertEqual(suggest_regions(cube)['suggested_regions'], [])

    def test_corrupt_or_unwritable_cache_does_not_block_a_fit(self):
        self.fit()
        path = next((self.root/'cache').glob('*.npz'))
        path.write_bytes(b'not a zip file')
        _, recovered = self.fit()
        self.assertFalse(recovered.cache_hit)
        file_as_directory = self.root/'blocked'
        file_as_directory.write_text('file')
        with patch.dict(os.environ, DPTK_FITTED_CACHE_DIR=str(file_as_directory)):
            _, ref = self.fit()
            self.assertFalse(ref.cache_hit)

    def test_cache_is_bounded_without_removing_unrelated_files(self):
        from core.drr_background_cache import _prune
        directory = self.root/'bounded'
        directory.mkdir()
        sentinel = directory/'keep.npz'
        sentinel.write_bytes(b'unrelated')
        for index in range(257):
            (directory/f'{index:064x}.npz').write_bytes(b'entry')
        _prune(directory)
        self.assertTrue(sentinel.exists())
        self.assertEqual(len(list(directory.glob('*.npz'))), 257)  # 256 cache entries + sentinel.

    def test_group_uses_file_overrides_and_checks_errors_without_losing_valid_files(self):
        write_csv(self.root/'a.csv', self.raw)
        write_csv(self.root/'b.csv', self.raw)
        recipe = {**asdict(SETTINGS), 'protected_regions_by_file': {'b.csv': ((1.55, 1.65), (1.75, 1.85))}}
        self.assertNotEqual(settings_for_source(recipe, 'a.csv'), settings_for_source(recipe, 'b.csv'))
        _, refs = load_fitted_group(self.root, ['a.csv', 'b.csv'], recipe, y_axis='bg')
        self.assertEqual(refs[1].settings.protected_regions, ((1.55, 1.65), (1.75, 1.85)))
        payload = check_group(self.root, ['a.csv', 'missing.csv', 'b.csv'], 'bg', recipe, 3,
                              progress=self.signals, log=self.signals)
        self.assertEqual(payload['generation'], 3)
        self.assertEqual(payload['results'][1]['report']['status'], 'Error')
        self.assertTrue(payload['results'][2]['reference'].cache_hit)
        no_mask = check_group(self.root, ['a.csv'], 'bg', {}, 4, progress=self.signals, log=self.signals)
        self.assertEqual(no_mask['results'][0]['report']['status'], 'Needs regions')
        self.assertTrue(no_mask['results'][0]['report']['suggested_regions'])

    def test_comparison_preserves_primary_and_uses_fixed_sg_with_heldout_bands(self):
        write_csv(self.root/'a.csv', self.raw)
        result = compare_methods(self.root, 'a.csv', 'bg', SETTINGS, 1, 21, 2,
                                 progress=self.signals, log=self.signals)
        self.assertEqual(result['reference'].settings, SETTINGS)
        self.assertEqual(result['alternate'].settings.method, 'polynomial')
        self.assertEqual(result['metrics']['sg_window'], 21)
        self.assertEqual(len(result['derivatives']), 2)
        for score in result['metrics']['validation']:
            self.assertGreater(score['blocks'], 0)
            self.assertLess(score['rms_percent'], .1)
        np.testing.assert_array_equal(result['derivatives'][0].energy, result['derivatives'][1].energy)


if __name__ == '__main__':
    unittest.main()
