"""Numerical equivalence and safe reuse; deliberately no wall-clock assertions."""
from dataclasses import replace
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from core import drr_background_quality as quality
from core import drr_background_review as review
from core.drr_background_memo import ReviewMemo
from core.drr_fitted_background import _robust_fit, _robust_gain_tilt, estimate_background
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope
from tests.test_drr_fitted_background import SETTINGS, synthetic, write_csv


class BackgroundPerformanceTests(unittest.TestCase):
    def setUp(self):
        quality._screen_memo.clear()
        review._validation_memo.clear()
        self.addCleanup(quality._screen_memo.clear)
        self.addCleanup(review._validation_memo.clear)

    def test_batched_solver_matches_independent_huber_fits_across_batch_boundaries(self):
        rng = np.random.default_rng(724)
        x = np.sort(rng.uniform(-1, 1, 893))
        values = rng.normal(size=(137, 1)) + rng.normal(size=(137, 1))*x
        values += rng.normal(size=values.shape)*np.linspace(0, .05, len(values))[:, None]
        values[1::3, 10:20] += 1.5
        values[2::3, 600:610] -= 2.
        values[0] = 0.  # Converges before its neighbours.
        design = np.column_stack((np.ones_like(x), x))
        expected = np.array([_robust_fit(design, row) for row in values])
        original = values.copy()
        for subset in (slice(None), slice(0, 1), slice(3, 5)):
            np.testing.assert_allclose(_robust_gain_tilt(x, values[subset]), expected[subset], rtol=1e-10, atol=1e-11)
        np.testing.assert_array_equal(values, original)

    def test_batch_solver_remains_cancellable_between_iterations(self):
        with patch('core.drr_fitted_background.check_load_cancelled',
                   side_effect=[None, None, LoadCancelled()]):
            with self.assertRaises(LoadCancelled):
                _robust_gain_tilt(np.linspace(-1, 1, 100), np.ones((70, 100)))

    def test_screening_reuses_data_but_reapplies_changed_protection(self):
        raw, _, _ = synthetic()
        with patch.object(quality, '_screen_features', wraps=quality._screen_features) as screen:
            first = quality.suggest_regions(raw)
            expected = quality.suggest_regions(raw)
            self.assertTrue(first['suggested_regions'])
            first['suggested_regions'].clear()
            covered = quality.suggest_regions(raw, expected['suggested_regions'])
            self.assertEqual(covered['suggested_regions'], [])
            self.assertEqual(quality.suggest_regions(raw), expected)
            self.assertEqual(screen.call_count, 1)
            changed = raw.Z.copy()
            changed[0, 40] *= 1.001
            quality.suggest_regions(replace(raw, Z=changed))
            quality.suggest_regions(replace(raw, energy=raw.energy+.001))
            with patch.object(quality, 'SCREEN_VERSION', 999):
                quality.suggest_regions(raw)
            self.assertEqual(screen.call_count, 4)

    def test_validation_reuse_is_independent_of_sg_but_tracks_data_and_settings(self):
        raw, _, _ = synthetic()
        signal = SimpleNamespace(emit=lambda *_: None)
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', DPTK_FITTED_CACHE_DIR=directory+'/cache'):
            write_csv(Path(directory)/'a.csv', raw)
            with patch.object(review, '_blocked_validation', wraps=review._blocked_validation) as validate:
                first = review.compare_methods(directory, 'a.csv', 'bg', SETTINGS, 1, 21, 2, progress=signal, log=signal)
                next_sg = review.compare_methods(directory, 'a.csv', 'bg', SETTINGS, 2, 31, 2, progress=signal, log=signal)
                self.assertEqual(validate.call_count, 2)  # One per method, not per view/SG window.
                self.assertEqual(first['metrics']['validation'], next_sg['metrics']['validation'])
                self.assertFalse(np.allclose(first['derivatives'][0].Z, next_sg['derivatives'][0].Z))
                # Full contents, rather than names/stat fields, decide reuse.
                ref = first['reference']
                review.blocked_validation(replace(raw, Z=raw.Z*1.001), ref)
                review.blocked_validation(raw, replace(ref, settings=replace(SETTINGS, smoothing=.002)))
                review.blocked_validation(replace(raw, gate=raw.gate+1), ref)
                mask = ref.fit_mask.copy()
                mask[5] = False
                review.blocked_validation(raw, replace(ref, fit_mask=mask))
                self.assertEqual(validate.call_count, 6)

    def test_temporary_validation_fits_skip_review_without_changing_predictions(self):
        raw, _, _ = synthetic()
        with patch.object(quality, 'inspect_background', wraps=quality.inspect_background) as inspect:
            full, ref = estimate_background(raw, SETTINGS)
            temporary, _ = estimate_background(raw, SETTINGS, assess_quality=False)
            score = review.blocked_validation(raw, ref)
            self.assertGreater(score['blocks'], 0)
            self.assertEqual(inspect.call_count, 1)
            self.assertEqual(ref.diagnostics['screened_gates'], len(raw.gate))
            np.testing.assert_array_equal(full.Z, temporary.Z)

    def test_memo_bounds_results_and_never_exposes_mutable_cached_state(self):
        memo = ReviewMemo(max_entries=2, max_bytes=100)
        first = memo.get_or_compute('a', lambda: dict(values=[1]))
        first['values'][0] = 999
        hit = memo.get_or_compute('a', lambda: self.fail('Unexpected recompute'))
        self.assertEqual(hit['values'], [1])
        hit['values'].clear()
        self.assertEqual(memo.get_or_compute('a', lambda: None)['values'], [1])
        memo.get_or_compute('b', lambda: dict(values=[2]))
        memo.get_or_compute('a', lambda: None)  # Keep a most recently used.
        memo.get_or_compute('c', lambda: dict(values=[3]))
        self.assertEqual(list(memo._entries), ['a', 'c'])
        self.assertLessEqual(memo._bytes, 100)
        memo.get_or_compute('large', lambda: dict(values='x'*200))
        self.assertNotIn('large', memo._entries)
        memo.clear()
        self.assertEqual(memo._bytes, 0)

    def test_cancelled_memo_work_is_not_published_and_hits_honour_cancellation(self):
        memo = ReviewMemo()
        cancellation = LoadCancellation()
        def cancelled_result():
            cancellation.cancel()
            return dict(value=1)
        with load_scope(cancellation), self.assertRaises(LoadCancelled):
            memo.get_or_compute('a', cancelled_result)
        self.assertFalse(memo._entries)
        memo.get_or_compute('a', lambda: dict(value=2))
        cancellation = LoadCancellation()
        with load_scope(cancellation):
            cancellation.cancel()
            with self.assertRaises(LoadCancelled):
                memo.get_or_compute('a', lambda: self.fail('Cached work must not run'))


if __name__ == '__main__':
    unittest.main()
