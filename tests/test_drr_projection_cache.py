from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

import numpy as np

from core import drr_background_structure as structure
from core.load_cancellation import LoadCancellation, LoadCancelled, load_scope


class ProjectionCacheTests(unittest.TestCase):
    def setUp(self):
        for name, value in (('_projection_cache', OrderedDict()), ('_projection_cache_bytes', 0)):
            change = patch.object(structure, name, value)
            change.start()
            self.addCleanup(change.stop)
        self.energy = np.sort(1239.841984 / np.linspace(580, 870, 81))
        self.values = np.random.default_rng(81).normal(size=(13, len(self.energy)))

    def test_reuses_projection_for_new_values_without_changing_results(self):
        expected = (structure._build_local_quadratic_projection(self.energy, .035) @ self.values.T).T
        with patch.object(structure, '_build_local_quadratic_projection',
                          wraps=structure._build_local_quadratic_projection) as build:
            first = structure._local_quadratic_trend(self.energy, self.values, .035)
            second = structure._local_quadratic_trend(self.energy.copy(), -self.values, .035)
            self.assertEqual(build.call_count, 1)
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, -expected)
        self.assertFalse(np.shares_memory(first, second))

    def test_complete_coordinates_width_and_dtype_invalidate_projection(self):
        reference_projection = structure._build_local_quadratic_projection
        with patch.object(structure, '_build_local_quadratic_projection',
                          wraps=reference_projection) as build:
            for energy, width in ((self.energy, .035), (self.energy, .070),
                                  (self.energy.astype(np.float32), .035)):
                actual = structure._local_quadratic_trend(energy, self.values, width)
                expected = (reference_projection(energy, width) @ self.values.T).T
                np.testing.assert_array_equal(actual, expected)
            changed = self.energy.copy()
            changed[40] += .0001  # Same shape and endpoints, different calibration.
            actual = structure._local_quadratic_trend(changed, self.values, .035)
            expected = (reference_projection(changed, .035) @ self.values.T).T
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(build.call_count, 4)

    def test_lru_entry_and_byte_bounds_and_oversized_projection(self):
        with patch.object(structure, '_PROJECTION_CACHE_MAX_ENTRIES', 2):
            first = structure._local_quadratic_projection(self.energy, .035)
            second = structure._local_quadratic_projection(self.energy, .040)
            self.assertIs(structure._local_quadratic_projection(self.energy.copy(), .035), first)
            structure._local_quadratic_projection(self.energy, .045)
            self.assertEqual(len(structure._projection_cache), 2)
            self.assertIs(structure._local_quadratic_projection(self.energy, .035), first)
            self.assertIsNot(structure._local_quadratic_projection(self.energy, .040), second)
        budget = max(size for _, size in structure._projection_cache.values())
        with patch.object(structure, '_PROJECTION_CACHE_MAX_BYTES', budget):
            structure._local_quadratic_projection(self.energy + .001, .035)
            self.assertLessEqual(structure._projection_cache_bytes, budget)
        before = tuple(structure._projection_cache)
        with patch.object(structure, '_PROJECTION_CACHE_MAX_BYTES', 1):
            structure._local_quadratic_projection(self.energy, .080)
        self.assertEqual(tuple(structure._projection_cache), before)
        self.assertEqual(structure._projection_cache_bytes,
                         sum(size for _, size in structure._projection_cache.values()))

    def test_cancelled_build_is_not_published_and_hits_honour_cancellation(self):
        cancellation = LoadCancellation()
        build = structure._build_local_quadratic_projection
        def cancel_after_build(*args):
            result = build(*args)
            cancellation.cancel()
            return result
        with load_scope(cancellation), patch.object(structure, '_build_local_quadratic_projection',
                                                    side_effect=cancel_after_build):
            with self.assertRaises(LoadCancelled):
                structure._local_quadratic_trend(self.energy, self.values, .035)
        self.assertFalse(structure._projection_cache)
        self.assertEqual(structure._projection_cache_bytes, 0)
        structure._local_quadratic_trend(self.energy, self.values, .035)
        cancellation = LoadCancellation()
        with load_scope(cancellation):
            cancellation.cancel()
            with self.assertRaises(LoadCancelled):
                structure._local_quadratic_trend(self.energy, self.values, .035)

    def test_concurrent_requests_publish_one_accounted_entry(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            outputs = list(executor.map(lambda _: structure._local_quadratic_trend(
                self.energy, self.values, .035), range(6)))
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])
        self.assertEqual(len(structure._projection_cache), 1)
        self.assertEqual(structure._projection_cache_bytes,
                         sum(size for _, size in structure._projection_cache.values()))


if __name__ == '__main__':
    unittest.main()
